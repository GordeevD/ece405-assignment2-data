from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fasttext


MODEL_PATH = Path(__file__).with_name("lid.176.bin.nosync")


@lru_cache(maxsize=1)
def _load_model() -> fasttext.FastText._FastText:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"fastText language model not found at: {MODEL_PATH}")
	return fasttext.load_model(str(MODEL_PATH))


def identify_language(text: str) -> tuple[str, float]:
	"""Identify the primary language in a Unicode string using fastText lid.176."""
	# fastText predict expects exactly one input line.
	cleaned = " ".join(text.split())
	if not cleaned:
		return "unknown", 0.0

	labels, scores = _load_model().predict(cleaned, k=1)
	if not labels:
		return "unknown", 0.0

	label = labels[0]
	if label.startswith("__label__"):
		label = label[len("__label__") :]

	remap = {
		"eng": "en",
		"zh-cn": "zh",
		"zh-tw": "zh",
		"zho": "zh",
		"cmn": "zh",
	}
	label = remap.get(label, label)

	score = float(scores[0]) if scores else 0.0
	score = max(0.0, min(1.0, score))
	return label, score
