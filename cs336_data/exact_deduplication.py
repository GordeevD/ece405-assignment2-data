"""Utilities for exact line deduplication across a corpus of text files."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict


def _line_hash(line: str) -> bytes:
	"""Return a fixed-size hash for a line to reduce counter memory use."""
	return hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()


def exact_line_deduplication(
	input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
	"""Rewrite files keeping only lines that appear exactly once in the corpus."""
	line_counts: dict[bytes, int] = defaultdict(int)

	# First pass: count line occurrences across all input files.
	for input_path in input_files:
		with open(input_path, "r", encoding="utf-8") as infile:
			for line in infile:
				line_counts[_line_hash(line)] += 1

	os.makedirs(output_directory, exist_ok=True)

	# Second pass: keep only globally unique lines in each rewritten file.
	for input_path in input_files:
		output_path = os.path.join(output_directory, os.path.basename(input_path))
		with open(input_path, "r", encoding="utf-8") as infile, open(
			output_path, "w", encoding="utf-8"
		) as outfile:
			for line in infile:
				if line_counts[_line_hash(line)] == 1:
					outfile.write(line)
