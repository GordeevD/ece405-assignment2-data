"""
Run Gopher quality filters on extracted text from WARC files.
Sample random examples for manual review and compare filter predictions to manual judgments.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType  # pyright: ignore[reportMissingImports]

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter


def normalize_text(text: str) -> str:
	return " ".join(text.split())


@dataclass
class GopherPrediction:
	doc_id: int
	uri: str
	text: str
	gopher_pass: bool


@dataclass
class ManualQualityJudgment:
	quality_pass: bool | None


def iter_warc_extracted_documents(warc_path: Path, max_records: int):
	"""Iterate through WARC file extracting text from response records."""
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


def build_predictions(warc_path: Path, max_records: int) -> list[GopherPrediction]:
	"""Build predictions by running gopher filter on all extracted documents."""
	predictions: list[GopherPrediction] = []
	for doc_id, (uri, text) in enumerate(iter_warc_extracted_documents(warc_path, max_records=max_records)):
		gopher_pass = gopher_quality_filter(text)
		predictions.append(
			GopherPrediction(
				doc_id=doc_id,
				uri=uri,
				text=text,
				gopher_pass=gopher_pass,
			)
		)
	return predictions


def sample_predictions(predictions: list[GopherPrediction], sample_size: int, seed: int) -> list[GopherPrediction]:
	"""Sample random predictions."""
	if sample_size >= len(predictions):
		return list(predictions)
	rng = random.Random(seed)
	return rng.sample(predictions, sample_size)


def interactive_labels(samples: list[GopherPrediction], preview_chars: int) -> dict[int, ManualQualityJudgment]:
	"""Prompt user to provide manual quality judgments for samples."""
	judgments: dict[int, ManualQualityJudgment] = {}
	print("\n" + "=" * 100)
	print("GOPHER QUALITY FILTER EVALUATION")
	print("=" * 100)
	print("For each sample, review the text and judge if it is HIGH QUALITY or LOW QUALITY.")
	print("Enter: y/yes for HIGH QUALITY (should pass), n/no for LOW QUALITY (should fail), or skip\n")

	for i, sample in enumerate(samples, 1):
		print("-" * 100)
		print(f"SAMPLE {i}/{len(samples)} | doc_id={sample.doc_id}")
		print(f"URI: {sample.uri}")
		print(f"Gopher filter predicts: {'PASS (high quality)' if sample.gopher_pass else 'FAIL (low quality)'}")
		print(f"\nText preview ({len(sample.text)} chars total):")
		print(f"{sample.text[:preview_chars]}")
		if len(sample.text) > preview_chars:
			print(f"... [truncated, {len(sample.text) - preview_chars} more chars]")

		while True:
			manual_raw = input("\nYour judgment - is this HIGH QUALITY text? (y/n/skip)> ").strip().lower()
			if manual_raw in {"", "skip", "s"}:
				print("[SKIPPED]")
				break
			elif manual_raw in {"y", "yes", "1", "true"}:
				judgments[sample.doc_id] = ManualQualityJudgment(quality_pass=True)
				print("[MARKED: HIGH QUALITY]")
				break
			elif manual_raw in {"n", "no", "0", "false"}:
				judgments[sample.doc_id] = ManualQualityJudgment(quality_pass=False)
				print("[MARKED: LOW QUALITY]")
				break
			else:
				print("Please enter y, n, or skip")

		print()

	print("=" * 100)
	return judgments


def find_filter_errors(
	predictions: list[GopherPrediction],
	judgments: dict[int, ManualQualityJudgment],
) -> list[tuple[GopherPrediction, bool, str]]:
	"""Find cases where filter prediction differs from manual judgment.
	
	Returns list of (prediction, manual_judgment, error_type) tuples.
	error_type is one of: "FP" (false positive - filter passed but should fail),
	"FN" (false negative - filter failed but should pass), "TP" (true positive), "TN" (true negative)
	"""
	errors: list[tuple[GopherPrediction, bool, str]] = []
	for p in predictions:
		manual = judgments.get(p.doc_id)
		if manual is None or manual.quality_pass is None:
			continue

		# Compare filter prediction to manual judgment
		if p.gopher_pass == manual.quality_pass:
			# Correct prediction
			error_type = "TP" if p.gopher_pass else "TN"
		else:
			# Incorrect prediction
			if p.gopher_pass and not manual.quality_pass:
				error_type = "FP"  # Filter passed but should fail
			else:
				error_type = "FN"  # Filter failed but should pass

		errors.append((p, manual.quality_pass, error_type))

	return errors


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Run Gopher quality filters on extracted WARC text, sample random examples for manual review, "
			"and report filter accuracy."
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
		default=500,
		help="Maximum characters to print for each sample preview.",
	)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="Prompt for manual quality judgments in the terminal.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.warc.exists():
		raise SystemExit(f"WARC file not found: {args.warc}")

	print(f"Processing WARC file: {args.warc}")
	print(f"Maximum records to process: {args.max_records}")

	predictions = build_predictions(args.warc, max_records=args.max_records)
	if not predictions:
		raise SystemExit("No extracted text was produced from scanned WARC responses.")

	print(f"\nTotal documents extracted: {len(predictions)}")

	# Calculate gopher pass rate
	passed = sum(1 for p in predictions if p.gopher_pass)
	passed_rate = passed / len(predictions) if predictions else 0.0
	print(f"Documents passing Gopher filter: {passed}/{len(predictions)} ({passed_rate:.1%})")

	# Sample for manual review
	samples = sample_predictions(predictions, args.sample_size, args.seed)
	print(f"\nSampling {len(samples)} random examples for manual review...")
	print(f"(Random seed: {args.seed})")

	# Get manual judgments
	judgments: dict[int, ManualQualityJudgment] = {}
	if args.interactive:
		judgments = interactive_labels(samples, args.preview_chars)
		print(f"Collected {len(judgments)} manual judgments")
	else:
		print("\nRun with --interactive to provide manual quality judgments")
		print("Example: uv run python cs336_data/gopher_quality_filters_example.py --interactive")
		return

	# Analyze results
	all_results = find_filter_errors(samples, judgments)

	# Separate by type
	correct = [r for r in all_results if r[2] in ("TP", "TN")]
	false_positives = [r for r in all_results if r[2] == "FP"]
	false_negatives = [r for r in all_results if r[2] == "FN"]
	true_positives = [r for r in all_results if r[2] == "TP"]
	true_negatives = [r for r in all_results if r[2] == "TN"]

	print("\n" + "=" * 100)
	print("GOPHER QUALITY FILTER ANALYSIS")
	print("=" * 100)

	accuracy = len(correct) / len(all_results) if all_results else 0.0
	print(f"\nOverall Accuracy: {accuracy:.1%} ({len(correct)}/{len(all_results)})")
	print(f"  True Positives (correctly flagged as low quality): {len(true_positives)}")
	print(f"  True Negatives (correctly flagged as high quality): {len(true_negatives)}")
	print(f"  False Positives (incorrectly flagged as high quality): {len(false_positives)}")
	print(f"  False Negatives (incorrectly flagged as low quality): {len(false_negatives)}")

	if false_positives:
		print("\n" + "-" * 100)
		print(f"FALSE POSITIVES ({len(false_positives)} cases):")
		print("Filter PASSED but should have FAILED - these are low quality texts that passed the filter")
		print("-" * 100)
		for p, manual, _ in false_positives:
			print(f"\ndoc_id={p.doc_id} | URI: {p.uri}")
			print(f"Text snippet: {p.text[:300]}")
			print("ANALYSIS: Why did this fail manual review but pass the filter?")
			print()

	if false_negatives:
		print("\n" + "-" * 100)
		print(f"FALSE NEGATIVES ({len(false_negatives)} cases):")
		print("Filter FAILED but should have PASSED - these are high quality texts that failed the filter")
		print("-" * 100)
		for p, manual, _ in false_negatives:
			print(f"\ndoc_id={p.doc_id} | URI: {p.uri}")
			print(f"Text snippet: {p.text[:300]}")
			print("ANALYSIS: Why did this pass manual review but fail the filter?")
			print()

	print("\n" + "=" * 100)
	print("COMMENTARY ON FILTER BEHAVIOR")
	print("=" * 100)
	print("""
The Gopher quality filters are rule-based and check for:
1. Word count: 50-100,000 words
2. Mean word length: 3-10 characters
3. Lines ending with ellipsis: < 30%
4. Words with alphabetic characters: > 80%

These rules capture obvious quality issues but may not align perfectly with human judgment because:
- They ignore semantic quality (a grammatically correct page could be advertising spam)
- They ignore domain-specific quality expectations
- They may be too strict or too lenient depending on use case
- They don't account for HTML remnants, formatting, or context

Their main value is in filtering out obviously malformed or degenerate text before more
sophisticated NLP processing.
""")


if __name__ == "__main__":
	main()
