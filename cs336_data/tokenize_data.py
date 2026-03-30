"""Tokenize filtered documents with GPT-2 tokenizer and serialize to uint16 binary.

The output format is a flat NumPy uint16 array written via `tofile`, which is
compatible with the provided training script in `cs336-basics`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def _extract_text(line: str, text_key: str) -> str:
	"""Extract a document string from either plain-text or JSONL input."""
	line = line.rstrip("\n")
	if not line:
		return ""

	if line.startswith("{"):
		try:
			payload = json.loads(line)
			value = payload.get(text_key, "")
			return value if isinstance(value, str) else ""
		except json.JSONDecodeError:
			return line

	return line


def tokenize_line_and_add_eos(item: tuple[str, str]) -> list[int]:
	"""Tokenize one document and append GPT-2 EOS token ID."""
	line, text_key = item
	text = _extract_text(line, text_key)
	if not text:
		return [TOKENIZER.eos_token_id]
	return TOKENIZER.encode(text) + [TOKENIZER.eos_token_id]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Tokenize filtered data with GPT-2 tokenizer and write uint16 .bin"
	)
	parser.add_argument("input_path", help="Path to filtered input data (JSONL or plain text)")
	parser.add_argument("output_path", help="Path to output binary token file")
	parser.add_argument(
		"--text-key",
		default="text",
		help="JSON key to read document text from when input is JSONL (default: text)",
	)
	parser.add_argument(
		"--chunksize",
		type=int,
		default=100,
		help="Chunk size for multiprocessing imap (default: 100)",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=multiprocessing.cpu_count(),
		help="Number of tokenizer workers (default: CPU count)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input_path)
	output_path = Path(args.output_path)

	with input_path.open("r", encoding="utf-8") as f:
		lines = f.readlines()

	work_items = [(line, args.text_key) for line in lines]

	with multiprocessing.Pool(processes=args.num_workers) as pool:
		results = list(
			tqdm(
				pool.imap(tokenize_line_and_add_eos, work_items, chunksize=args.chunksize),
				total=len(work_items),
				desc="Tokenizing lines",
			)
		)

	all_ids = [token_id for sublist in results for token_id in sublist]
	print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")

	ids_array = np.array(all_ids, dtype=np.uint16)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	ids_array.tofile(output_path)
	print(f"Wrote uint16 tokens to {output_path}")


if __name__ == "__main__":
	main()
