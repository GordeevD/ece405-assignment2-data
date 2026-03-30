"""Analyze PII masking behavior on WARC-extracted text.

This script:
1) Reads WARC response records.
2) Extracts plain text using `extract_text_from_html_bytes`.
3) Applies email/phone/IP masking from `mask_pii.py`.
4) Samples 20 masked examples.
5) Prints heuristic false-positive and false-negative candidates.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType  # pyright: ignore[reportMissingImports]

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.mask_pii import mask_emails, mask_ips, mask_phone_numbers


EMAIL_TOKEN = "|||EMAIL_ADDRESS|||"
PHONE_TOKEN = "|||PHONE_NUMBER|||"
IP_TOKEN = "|||IP_ADDRESS|||"

# Mirrors patterns currently used in cs336_data/mask_pii.py so we can inspect raw matches.
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
IP_PATTERN = re.compile(
	r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
	r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)


@dataclass
class Example:
	url: str
	original: str
	masked: str
	email_count: int
	phone_count: int
	ip_count: int


def _get_snippet(text: str, start: int, end: int, radius: int = 60) -> str:
	left = max(0, start - radius)
	right = min(len(text), end + radius)
	snippet = text[left:right].replace("\n", " ")
	return snippet.strip()


def _extract_candidates(original: str, masked: str) -> tuple[list[str], list[str]]:
	"""Return heuristic (false_positive, false_negative) candidate snippets."""
	false_positives: list[str] = []
	false_negatives: list[str] = []

	# Heuristic FP candidates: matches that often represent non-PII technical text.
	for match in IP_PATTERN.finditer(original):
		token = match.group(0)
		if token.startswith(("0.", "1.", "2.", "3.")):
			false_positives.append(
				f"Possible FP (IP/version-like): '{token}' in '{_get_snippet(original, match.start(), match.end())}'"
			)

	for match in PHONE_PATTERN.finditer(original):
		token = match.group(0)
		if re.search(r"\b(?:id|order|sku|build|ticket)\b", _get_snippet(original, match.start(), match.end()), re.IGNORECASE):
			false_positives.append(
				f"Possible FP (numeric identifier): '{token}' in '{_get_snippet(original, match.start(), match.end())}'"
			)

	# Heuristic FN candidates: suspicious patterns left after masking.
	fn_patterns = [
		(
			"Obfuscated email",
			re.compile(
				r"\b[\w.+-]+\s*(?:\[at\]|\(at\)| at )\s*[\w.-]+\s*"
				r"(?:\[dot\]|\(dot\)| dot )\s*[A-Za-z]{2,}\b",
				re.IGNORECASE,
			),
		),
		(
			"IPv6 address",
			re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b"),
		),
		(
			"Phone with slash separator",
			re.compile(r"\b\d{3}/\d{3}-\d{4}\b"),
		),
	]

	for label, pattern in fn_patterns:
		for match in pattern.finditer(masked):
			false_negatives.append(
				f"Possible FN ({label}): '{match.group(0)}' in '{_get_snippet(masked, match.start(), match.end())}'"
			)

	return false_positives[:3], false_negatives[:3]


def _iter_warc_paths(input_dir: Path) -> list[Path]:
	paths = sorted(input_dir.glob("*.warc.gz"))
	return [p for p in paths if p.is_file()]


def collect_examples(
	warc_paths: list[Path],
	max_records: int,
) -> tuple[list[Example], int, int]:
	examples: list[Example] = []
	records_seen = 0
	extracted_seen = 0

	for warc_path in warc_paths:
		with open(warc_path, "rb") as stream:
			for record in ArchiveIterator(stream, parse_http=True):
				if records_seen >= max_records:
					return examples, records_seen, extracted_seen
				if record.record_type != WarcRecordType.response:
					continue

				records_seen += 1
				payload = record.reader.read()
				text = extract_text_from_html_bytes(payload)
				if not text:
					continue
				extracted_seen += 1

				masked_1, email_count = mask_emails(text)
				masked_2, phone_count = mask_phone_numbers(masked_1)
				masked_3, ip_count = mask_ips(masked_2)

				if email_count + phone_count + ip_count == 0:
					continue

				examples.append(
					Example(
						url=record.headers.get("WARC-Target-URI") or "",
						original=text,
						masked=masked_3,
						email_count=email_count,
						phone_count=phone_count,
						ip_count=ip_count,
					)
				)

	return examples, records_seen, extracted_seen


def main() -> None:
	parser = argparse.ArgumentParser(description="Run PII masking analysis on WARC files.")
	parser.add_argument(
		"--warc-dir",
		type=Path,
		default=Path("cs336_data"),
		help="Directory containing .warc.gz files (default: cs336_data).",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=20,
		help="Number of random masked examples to inspect (default: 20).",
	)
	parser.add_argument(
		"--max-records",
		type=int,
		default=5000,
		help="Maximum number of WARC response records to scan (default: 5000).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=405,
		help="Random seed for reproducible sampling (default: 405).",
	)
	args = parser.parse_args()

	warc_paths = _iter_warc_paths(args.warc_dir)
	if not warc_paths:
		raise SystemExit(f"No .warc.gz files found in: {args.warc_dir}")

	examples, records_seen, extracted_seen = collect_examples(
		warc_paths=warc_paths,
		max_records=args.max_records,
	)
	if not examples:
		raise SystemExit("No replacements were made in scanned records.")

	random.seed(args.seed)
	sample_size = min(args.sample_size, len(examples))
	sample = random.sample(examples, sample_size)

	print(f"WARC files: {len(warc_paths)}")
	print(f"Response records scanned: {records_seen}")
	print(f"Texts extracted: {extracted_seen}")
	print(f"Examples with >=1 replacement: {len(examples)}")
	print(f"Sample size: {sample_size}")

	print("\n=== 20 RANDOM MASKED EXAMPLES ===")
	for i, ex in enumerate(sample, 1):
		print(f"\n[{i}] URL: {ex.url}")
		print(
			"Counts: "
			f"email={ex.email_count}, phone={ex.phone_count}, ip={ex.ip_count}, "
			f"total={ex.email_count + ex.phone_count + ex.ip_count}"
		)

		# Show one masked-token snippet when possible.
		token_idx = -1
		token_len = 0
		for token in (EMAIL_TOKEN, PHONE_TOKEN, IP_TOKEN):
			idx = ex.masked.find(token)
			if idx != -1:
				token_idx = idx
				token_len = len(token)
				break
		if token_idx != -1:
			print("Snippet:", _get_snippet(ex.masked, token_idx, token_idx + token_len))
		else:
			print("Snippet: <token not found in sampled text>")

	print("\n=== POSSIBLE FALSE POSITIVES / FALSE NEGATIVES (HEURISTIC) ===")
	fp_examples: list[str] = []
	fn_examples: list[str] = []
	for ex in sample:
		fps, fns = _extract_candidates(ex.original, ex.masked)
		fp_examples.extend(fps)
		fn_examples.extend(fns)

	if fp_examples:
		print("\nPossible false positives:")
		for line in fp_examples[:8]:
			print("-", line)
	else:
		print("\nPossible false positives:")
		print("- None detected by heuristic in this sample.")

	if fn_examples:
		print("\nPossible false negatives:")
		for line in fn_examples[:8]:
			print("-", line)
	else:
		print("\nPossible false negatives:")
		print("- None detected by heuristic in this sample.")


if __name__ == "__main__":
	main()
