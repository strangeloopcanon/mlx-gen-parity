from __future__ import annotations

"""Quickstart: inference + a tiny training step.

Run:
  python examples/quickstart.py --model Qwen/Qwen3-0.6B --prompt "Hello MLX"
"""

import argparse

import mlx.core as mx
from mlx_gen_parity import GenerationConfig, generate, TrainingConfig, train_step, SoftPromptHook
from mlx_gen_parity.loader import auto_load
from mlx.optimizers import AdamW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--prompt", default="Hello MLX")
    args = ap.parse_args()

    model, tokenizer, local = auto_load(args.model)
    print(f"Loaded from {local}")

    # Inference
    cfg = GenerationConfig(max_tokens=64, temperature=0.7, top_p=0.95, seed=42)
    out = generate(model, tokenizer, args.prompt, cfg)
    print("--- Inference ---\n", out["text"])  # noqa: T201

    # Tiny training step (bf16 compute)
    pad_id = getattr(tokenizer, "pad_token_id", -100) or -100
    opt = AdamW(learning_rate=2e-4)
    # Make a tiny batch from the prompt itself (toy example)
    ids = tokenizer.encode(args.prompt)
    T = 64
    seq = ids[: T - 1] + [tokenizer.eos_token_id]
    if len(seq) < T:
        seq = seq + [pad_id] * (T - len(seq))
    batch = {"tokens": mx.array([seq, seq], dtype=mx.int32)}  # B=2

    tcfg = TrainingConfig(dtype="bf16", loss_scale=1024.0)
    hooks = [SoftPromptHook(n_virtual=8, param_key="_soft_prompt")]
    loss = train_step(model, batch, opt, tcfg, hooks=hooks, pad_id=pad_id)
    print("--- Train step loss ---", float(loss))  # noqa: T201


if __name__ == "__main__":
    main()

