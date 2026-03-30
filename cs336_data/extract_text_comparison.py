from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict, deque

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def read_wet_records_by_uri(wet_path: str) -> dict[str, deque[str]]:
	records_by_uri: dict[str, deque[str]] = defaultdict(deque)

	with open(wet_path, "rb") as wet_file:
		for record in ArchiveIterator(wet_file, parse_http=False):
			if record.record_type != WarcRecordType.conversion:
				continue

			uri = record.headers.get("WARC-Target-URI")
			if not uri:
				continue

			wet_text = record.reader.read().decode("utf-8", errors="replace").strip()
			records_by_uri[uri].append(wet_text)

	return records_by_uri


def compare_warc_against_wet(
	warc_path: str,
	wet_records_by_uri: dict[str, deque[str]],
	max_examples: int = 5,
) -> tuple[Counter[str], list[dict[str, str]]]:
	stats: Counter[str] = Counter()
	diff_examples: list[dict[str, str]] = []

	with open(warc_path, "rb") as warc_file:
		for record in ArchiveIterator(warc_file, parse_http=True):
			if record.record_type != WarcRecordType.response:
				continue

			stats["warc_response_records"] += 1
			uri = record.headers.get("WARC-Target-URI")
			if not uri:
				stats["missing_uri_in_warc"] += 1
				continue

			wet_queue = wet_records_by_uri.get(uri)
			if not wet_queue:
				stats["missing_in_wet"] += 1
				continue

			wet_text = wet_queue.popleft()
			if not wet_queue:
				del wet_records_by_uri[uri]

			html_bytes = record.reader.read()
			extracted = extract_text_from_html_bytes(html_bytes)
			if extracted is None:
				stats["extractor_returned_none"] += 1
				continue

			stats["compared_pairs"] += 1
			extracted = extracted.strip()
			wet_text = wet_text.strip()

			if extracted == wet_text:
				stats["exact_match"] += 1
				continue

			extracted_norm = normalize_text(extracted)
			wet_norm = normalize_text(wet_text)

			if extracted_norm == wet_norm:
				stats["normalized_match"] += 1
				continue

			stats["different"] += 1

			if len(diff_examples) < max_examples:
				diff_examples.append(
					{
						"uri": uri,
						"ours_preview": extracted_norm[:320],
						"wet_preview": wet_norm[:320],
						"ours_len": str(len(extracted_norm)),
						"wet_len": str(len(wet_norm)),
					}
				)

	return stats, diff_examples


def print_report(stats: Counter[str], diff_examples: list[dict[str, str]]) -> None:
	compared_pairs = stats.get("compared_pairs", 0)

	print("=== WARC vs WET Extraction Comparison ===")
	print(f"WARC response records: {stats.get('warc_response_records', 0)}")
	print(f"Compared pairs: {compared_pairs}")
	print(f"Exact matches: {stats.get('exact_match', 0)}")
	print(f"Matches after whitespace normalization: {stats.get('normalized_match', 0)}")
	print(f"Different content: {stats.get('different', 0)}")
	print(f"WARC records missing in WET: {stats.get('missing_in_wet', 0)}")
	print(f"Extractor returned None: {stats.get('extractor_returned_none', 0)}")

	if compared_pairs > 0:
		close_rate = (stats.get("exact_match", 0) + stats.get("normalized_match", 0)) / compared_pairs
		print(f"Similarity rate (exact + normalized): {close_rate:.2%}")

	if diff_examples:
		print("\n=== Sample Differences ===")
		for i, example in enumerate(diff_examples, start=1):
			print(f"\n[{i}] URI: {example['uri']}")
			print(f"Our extractor length: {example['ours_len']}")
			print(f"WET text length: {example['wet_len']}")
			print(f"Our extractor preview: {example['ours_preview']}")
			print(f"WET preview: {example['wet_preview']}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare local HTML extraction output against WET output.")
	parser.add_argument(
		"--warc",
		default="cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz",
		help="Path to the source WARC file.",
	)
	parser.add_argument(
		"--wet",
		default="cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz",
		help="Path to the corresponding WET file.",
	)
	parser.add_argument(
		"--max-examples",
		type=int,
		default=5,
		help="Number of differing examples to print.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	wet_records_by_uri = read_wet_records_by_uri(args.wet)
	stats, diff_examples = compare_warc_against_wet(args.warc, wet_records_by_uri, max_examples=args.max_examples)
	print_report(stats, diff_examples)


if __name__ == "__main__":
	main()
