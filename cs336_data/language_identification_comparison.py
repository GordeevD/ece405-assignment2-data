from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language


def normalize_text(text: str) -> str:
	return " ".join(text.split())


@dataclass
class Prediction:
	doc_id: int
	source: str
	uri: str
	text: str
	predicted_label: str
	score: float


def iter_warc_extracted_documents(warc_path: str):
	with open(warc_path, "rb") as warc_file:
		for record in ArchiveIterator(warc_file, parse_http=True):
			if record.record_type != WarcRecordType.response:
				continue

			uri = record.headers.get("WARC-Target-URI") or ""
			html_bytes = record.reader.read()
			extracted = extract_text_from_html_bytes(html_bytes)
			if extracted is None:
				continue

			normalized = normalize_text(extracted)
			if normalized:
				yield uri, normalized


def iter_wet_documents(wet_path: str):
	with open(wet_path, "rb") as wet_file:
		for record in ArchiveIterator(wet_file, parse_http=False):
			if record.record_type != WarcRecordType.conversion:
				continue

			uri = record.headers.get("WARC-Target-URI") or ""
			text = record.reader.read().decode("utf-8", errors="replace")
			normalized = normalize_text(text)
			if normalized:
				yield uri, normalized


def build_predictions(warc_path: str, wet_path: str) -> list[Prediction]:
	predictions: list[Prediction] = []
	doc_id = 0

	for uri, text in iter_warc_extracted_documents(warc_path):
		label, score = identify_language(text)
		predictions.append(
			Prediction(
				doc_id=doc_id,
				source="warc",
				uri=uri,
				text=text,
				predicted_label=label,
				score=score,
			)
		)
		doc_id += 1

	for uri, text in iter_wet_documents(wet_path):
		label, score = identify_language(text)
		predictions.append(
			Prediction(
				doc_id=doc_id,
				source="wet",
				uri=uri,
				text=text,
				predicted_label=label,
				score=score,
			)
		)
		doc_id += 1

	return predictions


def english_fraction(predictions: list[Prediction]) -> float:
	if not predictions:
		return 0.0
	english_count = sum(1 for p in predictions if p.predicted_label == "en")
	return english_count / len(predictions)


def sample_predictions(predictions: list[Prediction], sample_size: int, seed: int) -> list[Prediction]:
	if not predictions:
		return []

	rng = random.Random(seed)
	if sample_size >= len(predictions):
		return list(predictions)
	return rng.sample(predictions, sample_size)


def write_samples_jsonl(samples: list[Prediction], output_path: Path, preview_chars: int) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as f:
		for sample in samples:
			row = {
				"doc_id": sample.doc_id,
				"source": sample.source,
				"uri": sample.uri,
				"predicted_label": sample.predicted_label,
				"score": sample.score,
				"manual_label": "",
				"text_preview": sample.text[:preview_chars],
			}
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_labels(path: Path) -> dict[int, str]:
	labels: dict[int, str] = {}

	if path.suffix.lower() == ".csv":
		with path.open("r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				doc_id_raw = row.get("doc_id")
				label = (row.get("manual_label") or "").strip().lower()
				if not doc_id_raw or not label:
					continue
				labels[int(doc_id_raw)] = label
		return labels

	with path.open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			row = json.loads(line)
			doc_id_raw = row.get("doc_id")
			label = str(row.get("manual_label", "")).strip().lower()
			if doc_id_raw is None or not label:
				continue
			labels[int(doc_id_raw)] = label

	return labels


def interactive_labels(samples: list[Prediction], preview_chars: int) -> dict[int, str]:
	labels: dict[int, str] = {}
	print("Enter manual labels for each sample (for example: en, zh, es).")
	print("Press Enter with no input to skip an item.\n")

	for sample in samples:
		print("-" * 80)
		print(f"doc_id: {sample.doc_id} | source: {sample.source} | predicted: {sample.predicted_label} ({sample.score:.4f})")
		print(f"uri: {sample.uri}")
		print(f"text_preview: {sample.text[:preview_chars]}")
		value = input("manual_label> ").strip().lower()
		if value:
			labels[sample.doc_id] = value

	print("-" * 80)
	return labels


def evaluate(
	predictions: list[Prediction],
	labels: dict[int, str],
	target_precision: float,
) -> tuple[list[Prediction], float | None, float | None]:
	labeled = [p for p in predictions if p.doc_id in labels]
	if not labeled:
		return [], None, None

	errors = [p for p in labeled if p.predicted_label != labels[p.doc_id]]

	# Use labeled samples to choose a threshold where accepted precision
	# reaches target_precision, if possible.
	candidates = sorted({p.score for p in labeled})
	best_threshold = None
	best_coverage = None

	for threshold in candidates:
		accepted = [p for p in labeled if p.score >= threshold]
		if not accepted:
			continue

		correct = sum(1 for p in accepted if p.predicted_label == labels[p.doc_id])
		precision = correct / len(accepted)
		coverage = len(accepted) / len(labeled)

		if precision >= target_precision:
			best_threshold = threshold
			best_coverage = coverage
			break

	if best_threshold is None:
		# Fallback: threshold with highest precision, break ties by higher coverage.
		fallback = None
		for threshold in candidates:
			accepted = [p for p in labeled if p.score >= threshold]
			if not accepted:
				continue
			correct = sum(1 for p in accepted if p.predicted_label == labels[p.doc_id])
			precision = correct / len(accepted)
			coverage = len(accepted) / len(labeled)
			rank = (precision, coverage, threshold)
			if fallback is None or rank > fallback:
				fallback = rank
		if fallback is not None:
			_, best_coverage, best_threshold = fallback

	return errors, best_threshold, best_coverage


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run language identification on WARC and WET text, sample documents for manual labeling, "
			"and report classifier errors/threshold guidance."
		)
	)
	parser.add_argument(
		"--warc",
		default="cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz",
		help="Path to WARC file (.warc.gz).",
	)
	parser.add_argument(
		"--wet",
		default="cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz",
		help="Path to WET file (.warc.wet.gz).",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=20,
		help="Number of random samples for manual language labeling.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=405,
		help="Random seed for sampling.",
	)
	parser.add_argument(
		"--preview-chars",
		type=int,
		default=300,
		help="Number of characters to include in text previews.",
	)
	parser.add_argument(
		"--sample-output",
		default="cs336_data/langid_manual_samples.jsonl",
		help="Where to write sampled examples for manual labeling.",
	)
	parser.add_argument(
		"--manual-labels",
		default=None,
		help="Optional JSONL/CSV file containing doc_id + manual_label.",
	)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="Prompt for manual labels in the terminal.",
	)
	parser.add_argument(
		"--target-precision",
		type=float,
		default=0.95,
		help="Desired precision for threshold recommendation.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	predictions = build_predictions(args.warc, args.wet)
	print(f"Total documents scored: {len(predictions)}")
	print(f"English document fraction (predicted): {english_fraction(predictions):.2%}")

	samples = sample_predictions(predictions, args.sample_size, args.seed)
	sample_output = Path(args.sample_output)
	write_samples_jsonl(samples, sample_output, args.preview_chars)
	print(f"Wrote {len(samples)} random samples to: {sample_output}")

	labels: dict[int, str] = {}
	if args.manual_labels:
		labels = load_manual_labels(Path(args.manual_labels))
		print(f"Loaded {len(labels)} manual labels from: {args.manual_labels}")
	elif args.interactive:
		labels = interactive_labels(samples, args.preview_chars)
		print(f"Collected {len(labels)} manual labels interactively.")
	else:
		print("No manual labels provided yet.")
		print("Next step: label the sample file (manual_label field), then rerun with --manual-labels.")
		return

	errors, threshold, coverage = evaluate(samples, labels, args.target_precision)
	evaluated_count = sum(1 for sample in samples if sample.doc_id in labels)

	print(f"Evaluated labeled samples: {evaluated_count}")
	print(f"Classifier errors in labeled set: {len(errors)}")

	if errors:
		print("\nSample classifier errors:")
		for item in errors:
			manual = labels[item.doc_id]
			print(
				f"doc_id={item.doc_id} source={item.source} pred={item.predicted_label} "
				f"score={item.score:.4f} manual={manual} uri={item.uri}"
			)

	if threshold is not None and coverage is not None:
		print(
			f"Suggested confidence threshold: {threshold:.4f} "
			f"(estimated coverage on labeled set: {coverage:.2%}, target precision={args.target_precision:.0%})"
		)
	else:
		print("Could not determine a threshold recommendation (insufficient labeled data).")


if __name__ == "__main__":
	main()
