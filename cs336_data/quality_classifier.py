from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import fasttext


MODEL_PATH = Path(__file__).with_name("cc.en.300.bin.nosync")


@lru_cache(maxsize=1)
def _load_model_if_available() -> fasttext.FastText._FastText | None:
	"""Load a trained quality model when present; otherwise return None."""
	if not MODEL_PATH.exists():
		return None
	return fasttext.load_model(str(MODEL_PATH))


def train_quality_classifier(
	train_file: str | Path,
	model_output_path: str | Path | None = None,
	epoch: int = 25,
	lr: float = 0.5,
	word_ngrams: int = 2,
	dim: int = 100,
	bucket: int = 50000,
) -> Path:
	"""Train a fastText quality classifier from a labeled text file.

	The input file should contain one sample per line in fastText format:
	`__label__wiki some high-quality text`
	`__label__cc some low-quality text`
	"""
	train_path = Path(train_file)
	if not train_path.exists():
		raise FileNotFoundError(f"Training file not found at: {train_path}")

	output_path = Path(model_output_path) if model_output_path else MODEL_PATH

	model = fasttext.train_supervised(
		input=str(train_path),
		epoch=epoch,
		lr=lr,
		wordNgrams=word_ngrams,
		dim=dim,
		bucket=bucket,
	)
	model.save_model(str(output_path))

	# Invalidate cache so future predictions can use the newly saved model.
	_load_model_if_available.cache_clear()
	return output_path


def _normalize_label(label: str) -> str:
	if label.startswith("__label__"):
		label = label[len("__label__") :]

	label = label.lower()
	if label in {"wiki", "high", "high-quality", "high_quality", "positive"}:
		return "wiki"
	if label in {"cc", "low", "low-quality", "low_quality", "negative"}:
		return "cc"
	return "wiki" if "wiki" in label or "high" in label else "cc"


def _heuristic_quality(text: str) -> tuple[str, float]:
	"""Fallback quality classifier for environments without a trained model."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return "cc", 0.0

	low_quality_markers = [
		"all rights reserved",
		"powered by",
		"register",
		"memberlist",
		"usergroups",
		"log in",
		"forum",
		"faq",
	]

	marker_hits = sum(1 for marker in low_quality_markers if marker in cleaned.lower())
	word_count = len(cleaned.split())
	line_count = max(1, text.count("\n") + 1)
	avg_words_per_line = word_count / line_count

	# Combine simple readability and boilerplate signals into a pseudo-logit.
	logit = 0.0
	logit += 0.02 * min(word_count, 400)
	logit += 0.08 * min(avg_words_per_line, 25)
	logit -= 1.5 * marker_hits

	prob_wiki = 1.0 / (1.0 + math.exp(-logit + 3.0))
	label = "wiki" if prob_wiki >= 0.5 else "cc"
	confidence = prob_wiki if label == "wiki" else (1.0 - prob_wiki)
	return label, float(max(0.0, min(1.0, confidence)))


def classify_quality(text: str) -> tuple[str, float]:
	"""Classify text quality and return (`wiki` or `cc`, confidence)."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return "cc", 0.0

	model = _load_model_if_available()
	if model is None:
		return _heuristic_quality(text)

	try:
		labels, scores = model.predict(cleaned, k=1)
		if not labels:
			return _heuristic_quality(text)

		label = _normalize_label(labels[0])
		score = float(scores[0]) if scores else 0.0
		score = max(0.0, min(1.0, score))
		return label, score
	except (ValueError, RuntimeError) as e:
		# Model may not be supervised or have other issues; fall back to heuristic
		return _heuristic_quality(text)
