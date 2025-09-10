from __future__ import annotations

"""
Minimal training skeleton for mlx-gen-parity.

This example trains persona vectors or LoRA adapters on top of an MLX model.
"""

import argparse
from typing import Dict

import mlx.core as mx
from mlx_lm import load

from mlx_gen_parity import (
    TrainingConfig,
    train_step,
    loss_forward,
    xent_loss,
    apply_lora,
    ResidualInjectionHook,
    LogitBiasHook,
    SoftPromptHook,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--persona", action="store_true")
    ap.add_argument("--soft-prompt", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load(args.model)

    if args.lora:
        apply_lora(model, rank=8, alpha=16)

    hooks = []
    if args.persona:
        # Attach a persona vector as a model parameter
        H = model.args.hidden_size
        model["_persona_v"] = mx.random.normal((H,)) * (1.0 / (H**0.5))
        hooks.append(LogitBiasHook(param_key="_persona_v", alpha=1.0))

    if args.soft_prompt:
        hooks.append(SoftPromptHook(n_virtual=10, init="rand", param_key="_soft_prompt"))

    cfg = TrainingConfig(learning_rate=args.lr, microbatch=0, grad_accum=1)

    # Optimizer over model trainable params
    from mlx.optimizers import AdamW

    opt = AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Dummy data: tokenize some synthetic sequences
    pad_id = getattr(tokenizer, "pad_token_id", -100) or -100

    def make_batch(B: int, T: int) -> Dict[str, mx.array]:
        import random

        prompts = ["Hello world " + str(i) for i in range(B)]
        ids = [tokenizer.encode(p)[: T - 1] for p in prompts]
        ids = [seq + [tokenizer.eos_token_id] for seq in ids]
        # pad
        padded = []
        for s in ids:
            s = s + [pad_id] * (T - len(s))
            padded.append(s)
        return {"tokens": mx.array(padded, dtype=mx.int32)}

    for step in range(args.steps):
        batch = make_batch(args.batch, args.seq)
        loss = train_step(model, batch, opt, cfg, hooks=hooks, pad_id=pad_id)
        print(f"step {step} loss {loss:.4f}")


if __name__ == "__main__":
    main()

