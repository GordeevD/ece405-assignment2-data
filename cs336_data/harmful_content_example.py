from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType  # pyright: ignore[reportMissingImports]

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech


def normalize_text(text: str) -> str:
	return " ".join(text.split())


@dataclass
class HarmfulPrediction:
	doc_id: int
	uri: str
	text: str
	nsfw_label: str
	nsfw_score: float
	toxic_label: str
	toxic_score: float


@dataclass
class ManualJudgment:
	nsfw: str | None
	toxic: str | None


def parse_binary_label(value: str | None, *, positive: str, negative: str) -> str | None:
	if value is None:
		return None

	clean = value.strip().lower()
	if clean in {"", "skip", "na", "n/a", "?"}:
		return None
	if clean in {positive, "yes", "y", "1", "true", "t", "pos", "positive"}:
		return positive
	if clean in {negative, "no", "n", "0", "false", "f", "neg", "negative"}:
		return negative
	return None


def iter_warc_extracted_documents(warc_path: Path, max_records: int):
	seen = 0
	with warc_path.open("rb") as warc_file:
		for record in ArchiveIterator(warc_file, parse_http=True):
			if record.record_type != WarcRecordType.response:
				continue

			if seen >= max_records:
				return
			seen += 1

			uri = record.headers.get("WARC-Target-URI") or ""
			html_bytes = record.reader.read()
			extracted = extract_text_from_html_bytes(html_bytes)
			if extracted is None:
				continue

			normalized = normalize_text(extracted)
			if normalized:
				yield uri, normalized


def build_predictions(warc_path: Path, max_records: int) -> list[HarmfulPrediction]:
	predictions: list[HarmfulPrediction] = []
	for doc_id, (uri, text) in enumerate(iter_warc_extracted_documents(warc_path, max_records=max_records)):
		nsfw_label, nsfw_score = classify_nsfw(text)
		toxic_label, toxic_score = classify_toxic_speech(text)
		predictions.append(
			HarmfulPrediction(
				doc_id=doc_id,
				uri=uri,
				text=text,
				nsfw_label=nsfw_label,
				nsfw_score=nsfw_score,
				toxic_label=toxic_label,
				toxic_score=toxic_score,
			)
		)
	return predictions


def sample_predictions(predictions: list[HarmfulPrediction], sample_size: int, seed: int) -> list[HarmfulPrediction]:
	if sample_size >= len(predictions):
		return list(predictions)
	rng = random.Random(seed)
	return rng.sample(predictions, sample_size)


def write_samples_jsonl(samples: list[HarmfulPrediction], output_path: Path, preview_chars: int) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		for sample in samples:
			row = {
				"doc_id": sample.doc_id,
				"uri": sample.uri,
				"nsfw_pred": sample.nsfw_label,
				"nsfw_score": sample.nsfw_score,
				"toxic_pred": sample.toxic_label,
				"toxic_score": sample.toxic_score,
				"manual_nsfw": "",
				"manual_toxic": "",
				"text_preview": sample.text[:preview_chars],
			}
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_labels(path: Path) -> dict[int, ManualJudgment]:
	judgments: dict[int, ManualJudgment] = {}

	if path.suffix.lower() == ".csv":
		with path.open("r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				doc_id_raw = row.get("doc_id")
				if doc_id_raw is None or not str(doc_id_raw).strip():
					continue

				nsfw = parse_binary_label(
					row.get("manual_nsfw"),
					positive="nsfw",
					negative="non-nsfw",
				)
				toxic = parse_binary_label(
					row.get("manual_toxic"),
					positive="toxic",
					negative="non-toxic",
				)
				judgments[int(doc_id_raw)] = ManualJudgment(nsfw=nsfw, toxic=toxic)
		return judgments

	with path.open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			row = json.loads(line)
			doc_id_raw = row.get("doc_id")
			if doc_id_raw is None:
				continue

			nsfw = parse_binary_label(
				str(row.get("manual_nsfw", "")),
				positive="nsfw",
				negative="non-nsfw",
			)
			toxic = parse_binary_label(
				str(row.get("manual_toxic", "")),
				positive="toxic",
				negative="non-toxic",
			)
			judgments[int(doc_id_raw)] = ManualJudgment(nsfw=nsfw, toxic=toxic)

	return judgments


def interactive_labels(samples: list[HarmfulPrediction], preview_chars: int) -> dict[int, ManualJudgment]:
	judgments: dict[int, ManualJudgment] = {}
	print("Enter manual labels for each sample. Use y/n (or press Enter to skip).")

	for sample in samples:
		print("-" * 90)
		print(f"doc_id={sample.doc_id} uri={sample.uri}")
		print(f"NSFW pred={sample.nsfw_label} score={sample.nsfw_score:.4f}")
		print(f"Toxic pred={sample.toxic_label} score={sample.toxic_score:.4f}")
		print(f"text_preview={sample.text[:preview_chars]}")

		nsfw_raw = input("manual_nsfw (y/n/skip)> ")
		toxic_raw = input("manual_toxic (y/n/skip)> ")

		nsfw = parse_binary_label(nsfw_raw, positive="nsfw", negative="non-nsfw")
		toxic = parse_binary_label(toxic_raw, positive="toxic", negative="non-toxic")
		judgments[sample.doc_id] = ManualJudgment(nsfw=nsfw, toxic=toxic)

	print("-" * 90)
	return judgments


def harmful_fraction(predictions: list[HarmfulPrediction], nsfw_threshold: float, toxic_threshold: float) -> float:
	if not predictions:
		return 0.0

	harmful = 0
	for p in predictions:
		nsfw_hit = p.nsfw_label == "nsfw" and p.nsfw_score >= nsfw_threshold
		toxic_hit = p.toxic_label == "toxic" and p.toxic_score >= toxic_threshold
		if nsfw_hit or toxic_hit:
			harmful += 1
	return harmful / len(predictions)


def find_classifier_errors(
	predictions: list[HarmfulPrediction],
	judgments: dict[int, ManualJudgment],
	mode: str,
) -> list[tuple[HarmfulPrediction, str]]:
	errors: list[tuple[HarmfulPrediction, str]] = []
	for p in predictions:
		manual = judgments.get(p.doc_id)
		if manual is None:
			continue

		if mode == "nsfw":
			if manual.nsfw is None:
				continue
			pred_label = p.nsfw_label
			manual_label = manual.nsfw
		else:
			if manual.toxic is None:
				continue
			pred_label = p.toxic_label
			manual_label = manual.toxic

		if pred_label != manual_label:
			errors.append((p, manual_label))
	return errors


def recommend_threshold(
	predictions: list[HarmfulPrediction],
	judgments: dict[int, ManualJudgment],
	mode: str,
	target_precision: float,
) -> tuple[float | None, float | None, float | None]:
	rows: list[tuple[float, bool]] = []
	manual_positive_count = 0

	for p in predictions:
		manual = judgments.get(p.doc_id)
		if manual is None:
			continue

		if mode == "nsfw":
			manual_label = manual.nsfw
			pred_positive = p.nsfw_label == "nsfw"
			score = p.nsfw_score
			positive_label = "nsfw"
		else:
			manual_label = manual.toxic
			pred_positive = p.toxic_label == "toxic"
			score = p.toxic_score
			positive_label = "toxic"

		if manual_label is None:
			continue

		if manual_label == positive_label:
			manual_positive_count += 1

		if pred_positive:
			rows.append((score, manual_label == positive_label))

	if not rows:
		return None, None, None

	thresholds = sorted({score for score, _ in rows})
	best_threshold: float | None = None
	best_precision: float | None = None
	best_recall: float | None = None

	for threshold in thresholds:
		selected = [is_true_positive for score, is_true_positive in rows if score >= threshold]
		if not selected:
			continue

		true_positive = sum(1 for value in selected if value)
		precision = true_positive / len(selected)
		recall = (true_positive / manual_positive_count) if manual_positive_count > 0 else 0.0

		if precision >= target_precision:
			best_threshold = threshold
			best_precision = precision
			best_recall = recall
			break

	if best_threshold is not None:
		return best_threshold, best_precision, best_recall

	# Fallback when target precision is not reachable.
	fallback = None
	for threshold in thresholds:
		selected = [is_true_positive for score, is_true_positive in rows if score >= threshold]
		if not selected:
			continue
		true_positive = sum(1 for value in selected if value)
		precision = true_positive / len(selected)
		recall = (true_positive / manual_positive_count) if manual_positive_count > 0 else 0.0
		f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
		rank = (f1, precision, recall, threshold)
		if fallback is None or rank > fallback:
			fallback = rank

	if fallback is None:
		return None, None, None

	_, precision, recall, threshold = fallback
	return threshold, precision, recall


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run NSFW and toxic-speech filters on extracted WARC text, sample random examples for manual review, "
			"and report classifier errors, harmful fraction, and confidence-threshold guidance."
		)
	)
	parser.add_argument(
		"--warc",
		type=Path,
		default=Path("cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"),
		help="Path to input WARC file (.warc.gz).",
	)
	parser.add_argument(
		"--max-records",
		type=int,
		default=2000,
		help="Maximum number of response records to process.",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=20,
		help="Number of random examples for manual inspection.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=405,
		help="Random seed for reproducible sampling.",
	)
	parser.add_argument(
		"--preview-chars",
		type=int,
		default=320,
		help="Maximum characters to print for each sample preview.",
	)
	parser.add_argument(
		"--sample-output",
		type=Path,
		default=Path("cs336_data/harmful_content_manual_samples.jsonl"),
		help="Where to write sampled examples for manual labeling.",
	)
	parser.add_argument(
		"--manual-labels",
		type=Path,
		default=None,
		help="Optional JSONL/CSV with doc_id, manual_nsfw, manual_toxic.",
	)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="Prompt for manual labels in the terminal.",
	)
	parser.add_argument(
		"--target-precision",
		type=float,
		default=0.90,
		help="Desired precision for threshold recommendation on harmful predictions.",
	)
	parser.add_argument(
		"--nsfw-threshold",
		type=float,
		default=0.50,
		help="Threshold used to estimate harmful fraction from NSFW classifier.",
	)
	parser.add_argument(
		"--toxic-threshold",
		type=float,
		default=0.50,
		help="Threshold used to estimate harmful fraction from toxic-speech classifier.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.warc.exists():
		raise SystemExit(f"WARC file not found: {args.warc}")

	predictions = build_predictions(args.warc, max_records=args.max_records)
	if not predictions:
		raise SystemExit("No extracted text was produced from scanned WARC responses.")

	print(f"Total documents scored: {len(predictions)}")
	base_harmful_fraction = harmful_fraction(
		predictions,
		nsfw_threshold=args.nsfw_threshold,
		toxic_threshold=args.toxic_threshold,
	)
	print(
		"Predicted harmful fraction "
		f"(nsfw_threshold={args.nsfw_threshold:.2f}, toxic_threshold={args.toxic_threshold:.2f}): "
		f"{base_harmful_fraction:.2%}"
	)

	samples = sample_predictions(predictions, args.sample_size, args.seed)
	write_samples_jsonl(samples, args.sample_output, args.preview_chars)
	print(f"Wrote {len(samples)} random samples to: {args.sample_output}")

	judgments: dict[int, ManualJudgment] = {}
	if args.manual_labels is not None:
		judgments = load_manual_labels(args.manual_labels)
		print(f"Loaded manual labels: {len(judgments)}")
	elif args.interactive:
		judgments = interactive_labels(samples, args.preview_chars)
		print(f"Collected manual labels interactively: {len(judgments)}")
	else:
		print("No manual labels provided yet.")
		print("Next step: inspect the 20 samples and rerun with --interactive or --manual-labels.")
		return

	nsfw_errors = find_classifier_errors(samples, judgments, mode="nsfw")
	toxic_errors = find_classifier_errors(samples, judgments, mode="toxic")

	print(f"\nNSFW errors in labeled sample: {len(nsfw_errors)}")
	if nsfw_errors:
		for p, manual in nsfw_errors:
			print(
				f"doc_id={p.doc_id} pred={p.nsfw_label} score={p.nsfw_score:.4f} "
				f"manual={manual} uri={p.uri}"
			)

	print(f"\nToxic-speech errors in labeled sample: {len(toxic_errors)}")
	if toxic_errors:
		for p, manual in toxic_errors:
			print(
				f"doc_id={p.doc_id} pred={p.toxic_label} score={p.toxic_score:.4f} "
				f"manual={manual} uri={p.uri}"
			)

	nsfw_threshold, nsfw_precision, nsfw_recall = recommend_threshold(
		samples,
		judgments,
		mode="nsfw",
		target_precision=args.target_precision,
	)
	toxic_threshold, toxic_precision, toxic_recall = recommend_threshold(
		samples,
		judgments,
		mode="toxic",
		target_precision=args.target_precision,
	)

	print("\nSuggested thresholds from labeled sample:")
	if nsfw_threshold is None:
		print("NSFW: insufficient labeled positives to recommend a threshold.")
	else:
		print(
			f"NSFW: threshold={nsfw_threshold:.4f}, precision={nsfw_precision:.2%}, "
			f"recall={nsfw_recall:.2%}"
		)

	if toxic_threshold is None:
		print("Toxic-speech: insufficient labeled positives to recommend a threshold.")
	else:
		print(
			f"Toxic-speech: threshold={toxic_threshold:.4f}, precision={toxic_precision:.2%}, "
			f"recall={toxic_recall:.2%}"
		)

	if nsfw_threshold is not None and toxic_threshold is not None:
		recommended_harmful_fraction = harmful_fraction(
			predictions,
			nsfw_threshold=nsfw_threshold,
			toxic_threshold=toxic_threshold,
		)
		print(
			"Estimated harmful fraction under suggested thresholds: "
			f"{recommended_harmful_fraction:.2%}"
		)


if __name__ == "__main__":
	main()
