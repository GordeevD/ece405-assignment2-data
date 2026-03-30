from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fasttext


NSFW_MODEL_PATH = Path(__file__).with_name("jigsaw_fasttext_bigrams_nsfw_final.bin")
TOXIC_MODEL_PATH = Path(__file__).with_name("jigsaw_fasttext_bigrams_hatespeech_final.bin")


@lru_cache(maxsize=1)
def _load_nsfw_model() -> fasttext.FastText._FastText:
	if not NSFW_MODEL_PATH.exists():
		raise FileNotFoundError(f"NSFW fastText model not found at: {NSFW_MODEL_PATH}")
	return fasttext.load_model(str(NSFW_MODEL_PATH))


@lru_cache(maxsize=1)
def _load_toxic_model() -> fasttext.FastText._FastText:
	if not TOXIC_MODEL_PATH.exists():
		raise FileNotFoundError(f"Toxic-speech fastText model not found at: {TOXIC_MODEL_PATH}")
	return fasttext.load_model(str(TOXIC_MODEL_PATH))


def classify_nsfw(text: str) -> tuple[str, float]:
	"""Classify text as NSFW or non-NSFW using a trained fastText model."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return "non-nsfw", 0.0

	labels, scores = _load_nsfw_model().predict(cleaned, k=1)
	if not labels:
		return "non-nsfw", 0.0

	label = labels[0]
	if label.startswith("__label__"):
		label = label[len("__label__") :]

	if label not in {"nsfw", "non-nsfw"}:
		label = "nsfw" if "nsfw" in label and "non" not in label else "non-nsfw"

	score = float(scores[0]) if scores else 0.0
	score = max(0.0, score)
	return label, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
	"""Classify text as toxic or non-toxic using a trained fastText model."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return "non-toxic", 0.0

	labels, scores = _load_toxic_model().predict(cleaned, k=1)
	if not labels:
		return "non-toxic", 0.0

	label = labels[0]
	if label.startswith("__label__"):
		label = label[len("__label__") :]

	if label not in {"toxic", "non-toxic"}:
		label = "toxic" if "toxic" in label and "non" not in label else "non-toxic"

	score = float(scores[0]) if scores else 0.0
	score = max(0.0, score)
	return label, score
