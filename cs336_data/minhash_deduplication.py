"""Minhash + LSH fuzzy deduplication for text documents."""

from __future__ import annotations

import hashlib
import itertools
import os
import random
import re
import unicodedata
from collections import defaultdict

import mmh3


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MAX_U64 = (1 << 64) - 1


def _normalize_text(text: str) -> str:
	"""Normalize text to improve fuzzy duplicate recall."""
	# NFD + accent stripping follows common web-data normalization practice.
	text = unicodedata.normalize("NFD", text)
	text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
	text = text.lower()
	text = _PUNCT_RE.sub(" ", text)
	text = _WS_RE.sub(" ", text).strip()
	return text


def _word_ngrams(text: str, n: int) -> set[str]:
	words = text.split()
	if not words:
		return set()
	if len(words) < n:
		return {" ".join(words)}
	return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _compute_minhash_signature(ngrams: set[str], num_hashes: int) -> tuple[int, ...]:
	if not ngrams:
		return tuple(0 for _ in range(num_hashes))

	signature = [_MAX_U64] * num_hashes
	for gram in ngrams:
		gram_bytes = gram.encode("utf-8")
		for i in range(num_hashes):
			h = mmh3.hash64(gram_bytes, seed=i, signed=False)[0]
			if h < signature[i]:
				signature[i] = h
	return tuple(signature)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
	if not a and not b:
		return 1.0
	union = a | b
	if not union:
		return 0.0
	return len(a & b) / len(union)


def _connected_components(num_nodes: int, edges: set[tuple[int, int]]) -> list[list[int]]:
	adjacency: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
	for u, v in edges:
		adjacency[u].add(v)
		adjacency[v].add(u)

	seen: set[int] = set()
	components: list[list[int]] = []
	for node in range(num_nodes):
		if node in seen:
			continue
		stack = [node]
		seen.add(node)
		component: list[int] = []
		while stack:
			cur = stack.pop()
			component.append(cur)
			for nxt in adjacency[cur]:
				if nxt not in seen:
					seen.add(nxt)
					stack.append(nxt)
		components.append(component)
	return components


def minhash_deduplication(
	input_files: list[os.PathLike],
	num_hashes: int,
	num_bands: int,
	ngrams: int,
	jaccard_threshold: float,
	output_directory: os.PathLike,
) -> None:
	"""Perform fuzzy document deduplication with minhash + LSH."""
	if num_hashes % num_bands != 0:
		raise ValueError("num_hashes must be evenly divisible by num_bands")

	documents: list[str] = []
	for path in input_files:
		with open(path, "r", encoding="utf-8") as infile:
			documents.append(infile.read())

	normalized_docs = [_normalize_text(doc) for doc in documents]
	ngram_sets = [_word_ngrams(doc, ngrams) for doc in normalized_docs]
	signatures = [
		_compute_minhash_signature(ngram_set, num_hashes) for ngram_set in ngram_sets
	]

	rows_per_band = num_hashes // num_bands
	buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
	for doc_idx, signature in enumerate(signatures):
		for band_idx in range(num_bands):
			start = band_idx * rows_per_band
			end = start + rows_per_band
			band = signature[start:end]
			band_bytes = b"".join(v.to_bytes(8, "little", signed=False) for v in band)
			band_hash = hashlib.blake2b(band_bytes, digest_size=16).digest()
			buckets[(band_idx, band_hash)].append(doc_idx)

	candidate_pairs: set[tuple[int, int]] = set()
	for bucket_docs in buckets.values():
		if len(bucket_docs) < 2:
			continue
		for i, j in itertools.combinations(sorted(set(bucket_docs)), 2):
			candidate_pairs.add((i, j))

	duplicate_edges: set[tuple[int, int]] = set()
	for i, j in candidate_pairs:
		similarity = _jaccard_similarity(ngram_sets[i], ngram_sets[j])
		if similarity >= jaccard_threshold:
			duplicate_edges.add((i, j))

	components = _connected_components(len(documents), duplicate_edges)
	rng = random.Random(0)

	kept_docs: set[int] = set()
	for component in components:
		if len(component) == 1:
			kept_docs.add(component[0])
			continue
		kept_docs.add(rng.choice(component))

	os.makedirs(output_directory, exist_ok=True)
	for idx, input_path in enumerate(input_files):
		if idx not in kept_docs:
			continue
		output_path = os.path.join(output_directory, os.path.basename(input_path))
		with open(output_path, "w", encoding="utf-8") as outfile:
			outfile.write(documents[idx])
