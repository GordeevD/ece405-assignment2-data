"""Train a GPT-2-small-shaped language model on tokenized data.

This script is a local training runner for Assignment 2 deliverables. It tracks
periodic validation loss, records the best observed value, and writes learning
curve artifacts (JSON/CSV and optional PNG plot).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

# Make cs336-basics package importable when running from repository root.
ROOT = Path(__file__).resolve().parents[1]
CS336_BASICS_DIR = ROOT / "cs336-basics"
if str(CS336_BASICS_DIR) not in sys.path:
	sys.path.insert(0, str(CS336_BASICS_DIR))

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr


def pick_device(device_arg: str) -> str:
	if device_arg != "auto":
		return device_arg
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def estimate_validation_loss(
	model: BasicsTransformerLM,
	valid_data: np.ndarray,
	eval_batch_size: int,
	eval_iterations: int,
	device: str,
	context_length: int,
) -> float:
	model.eval()
	losses: list[float] = []
	with torch.no_grad():
		for _ in range(eval_iterations):
			x, y = get_batch(valid_data, eval_batch_size, context_length, device)
			logits = model(x)
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
			losses.append(float(loss.item()))
	model.train()
	return float(sum(losses) / len(losses)) if losses else math.nan


def maybe_plot_curve(curve: list[dict], output_png: Path) -> bool:
	try:
		import matplotlib.pyplot as plt
	except Exception:
		return False

	if not curve:
		return False

	steps = [row["step"] for row in curve]
	vals = [row["val_loss"] for row in curve]

	plt.figure(figsize=(8, 5))
	plt.plot(steps, vals, marker="o")
	plt.xlabel("Step")
	plt.ylabel("Validation Loss")
	plt.title("Validation Learning Curve")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_png, dpi=150)
	plt.close()
	return True


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train GPT-2-small-shaped LM and report best validation loss")
	parser.add_argument(
		"--train-bin",
		default="/tmp/ece405_filtered_data/CC-MAIN-20250417135010-20250417165010-00065_gpt2.bin",
		help="Path to uint16 GPT-2-tokenized training data (.bin)",
	)
	parser.add_argument(
		"--valid-bin",
		default="/data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
		help="Path to uint16 validation data (.bin), ideally Paloma C4 100 domains",
	)
	parser.add_argument(
		"--fallback-valid-fraction",
		type=float,
		default=0.01,
		help="If --valid-bin is missing, use this tail fraction of train data as fallback validation",
	)
	parser.add_argument("--output-dir", default="cs336_data/output/train_model", help="Directory for outputs")
	parser.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")

	# GPT-2 small shape defaults.
	parser.add_argument("--vocab-size", type=int, default=50257)
	parser.add_argument("--context-length", type=int, default=512)
	parser.add_argument("--d-model", type=int, default=768)
	parser.add_argument("--d-ff", type=int, default=2048)
	parser.add_argument("--num-layers", type=int, default=12)
	parser.add_argument("--num-heads", type=int, default=12)
	parser.add_argument("--rope-theta", type=float, default=10000.0)

	parser.add_argument("--train-steps", type=int, default=120)
	parser.add_argument("--train-batch-size", type=int, default=8)
	parser.add_argument("--eval-batch-size", type=int, default=8)
	parser.add_argument("--eval-interval", type=int, default=20)
	parser.add_argument("--eval-iterations", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--warmup-ratio", type=float, default=0.01)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--adam-beta1", type=float, default=0.9)
	parser.add_argument("--adam-beta2", type=float, default=0.98)
	parser.add_argument("--adam-eps", type=float, default=1e-9)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=0)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	train_bin = Path(args.train_bin)
	if not train_bin.exists():
		raise FileNotFoundError(f"Train bin not found: {train_bin}")

	valid_bin = Path(args.valid_bin)
	used_fallback_valid = False

	train_data_full = np.memmap(train_bin, dtype=np.uint16, mode="r")

	if valid_bin.exists():
		train_data = train_data_full
		valid_data = np.memmap(valid_bin, dtype=np.uint16, mode="r")
	else:
		used_fallback_valid = True
		split_idx = int(len(train_data_full) * (1.0 - args.fallback_valid_fraction))
		split_idx = max(split_idx, args.context_length + 2)
		split_idx = min(split_idx, len(train_data_full) - (args.context_length + 2))
		train_data = train_data_full[:split_idx]
		valid_data = train_data_full[split_idx:]

	if len(train_data) <= args.context_length + 2:
		raise ValueError("Training data is too small for the selected context length")
	if len(valid_data) <= args.context_length + 2:
		raise ValueError("Validation data is too small for the selected context length")

	device = pick_device(args.device)
	model = BasicsTransformerLM(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
	).to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.lr,
		betas=(args.adam_beta1, args.adam_beta2),
		eps=args.adam_eps,
		weight_decay=args.weight_decay,
	)

	curve: list[dict] = []
	best_val_loss = float("inf")
	best_step = -1
	warmup_iters = max(1, int(args.train_steps * args.warmup_ratio))
	start_time = time.time()

	x, y = get_batch(train_data, args.train_batch_size, args.context_length, device)

	for step in (pbar := trange(args.train_steps, desc="Training")):
		lr = get_cosine_lr(
			step,
			max_learning_rate=args.lr,
			min_learning_rate=args.lr * 0.1,
			warmup_iters=warmup_iters,
			cosine_cycle_iters=args.train_steps,
		)
		for group in optimizer.param_groups:
			group["lr"] = lr

		logits = model(x)
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		if args.max_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
		optimizer.step()

		x, y = get_batch(train_data, args.train_batch_size, args.context_length, device)
		pbar.set_description(f"Training step {step}, train_loss={loss.item():.4f}")

		if step > 0 and step % args.eval_interval == 0:
			val_loss = estimate_validation_loss(
				model,
				valid_data,
				args.eval_batch_size,
				args.eval_iterations,
				device,
				args.context_length,
			)
			record = {
				"step": int(step),
				"train_loss": float(loss.item()),
				"val_loss": float(val_loss),
				"lr": float(lr),
				"elapsed_sec": float(time.time() - start_time),
			}
			curve.append(record)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_step = step
				torch.save(model.state_dict(), output_dir / "best_model.pt")

	final_val_loss = estimate_validation_loss(
		model,
		valid_data,
		args.eval_batch_size,
		args.eval_iterations,
		device,
		args.context_length,
	)
	final_record = {
		"step": int(args.train_steps),
		"train_loss": float(loss.item()),
		"val_loss": float(final_val_loss),
		"lr": float(args.lr * 0.1),
		"elapsed_sec": float(time.time() - start_time),
	}
	curve.append(final_record)
	if final_val_loss < best_val_loss:
		best_val_loss = final_val_loss
		best_step = args.train_steps
		torch.save(model.state_dict(), output_dir / "best_model.pt")

	torch.save(model.state_dict(), output_dir / "final_model.pt")

	curve_json = output_dir / "learning_curve.json"
	curve_csv = output_dir / "learning_curve.csv"
	summary_json = output_dir / "train_summary.json"

	with curve_json.open("w", encoding="utf-8") as f:
		json.dump(curve, f, indent=2)

	with curve_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_sec"])
		writer.writeheader()
		writer.writerows(curve)

	plot_written = maybe_plot_curve(curve, output_dir / "learning_curve.png")

	summary = {
		"train_bin": str(train_bin),
		"valid_bin": str(valid_bin if valid_bin.exists() else ""),
		"used_fallback_validation_split": used_fallback_valid,
		"device": device,
		"train_steps": args.train_steps,
		"train_batch_size": args.train_batch_size,
		"eval_batch_size": args.eval_batch_size,
		"eval_interval": args.eval_interval,
		"eval_iterations": args.eval_iterations,
		"best_validation_loss": float(best_val_loss),
		"best_validation_step": int(best_step),
		"final_validation_loss": float(final_val_loss),
		"learning_curve_json": str(curve_json),
		"learning_curve_csv": str(curve_csv),
		"learning_curve_png": str(output_dir / "learning_curve.png") if plot_written else "",
	}
	with summary_json.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("=" * 72)
	print("Training complete")
	print(f"Best validation loss: {best_val_loss:.6f} at step {best_step}")
	if used_fallback_valid:
		print("Validation used fallback split from training bin (C4 validation file was missing).")
	else:
		print("Validation used provided validation bin.")
	print(f"Summary written to: {summary_json}")
	print("=" * 72)


if __name__ == "__main__":
	main()
