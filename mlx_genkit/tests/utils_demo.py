from __future__ import annotations

"""
Demo script to exercise mlx-genkit helpers:

 - sequence_logprob
 - token_kl
 - ema_update
 - build_action_mask
 - stable_softmax

Run:
  python -m mlx_genkit.tests.utils_demo --model ./mlx_qwen3_0_6b --seq 32
"""

import argparse
from typing import List

import mlx.core as mx
from mlx_lm import load as mlx_load

from mlx_genkit import (
    sequence_logprob,
    token_kl,
    ema_update,
    build_action_mask,
    stable_softmax,
    clone_reference,
)


def make_batch(tokenizer, prompts: List[str], seq_len: int):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_tok = eos_id if pad_id is None else pad_id
    ids = [tokenizer.encode(p) for p in prompts]
    prompt_lens = [min(len(x), seq_len - 1) for x in ids]
    batch = []
    for i, seq in enumerate(ids):
        seq = seq[: seq_len - 1] + [eos_id]
        if len(seq) < seq_len:
            seq = seq + [pad_tok] * (seq_len - len(seq))
        batch.append(seq)
    tokens = mx.array(batch, dtype=mx.int32)
    return tokens, prompt_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./mlx_qwen3_0_6b")
    ap.add_argument("--seq", type=int, default=32)
    args = ap.parse_args()

    model, tokenizer = mlx_load(args.model)

    prompts = [
        "Hello MLX parity",
        "Utilities demo for mlx-genkit",
    ]
    T = int(args.seq)
    tokens, prompt_lens = make_batch(tokenizer, prompts, T)

    # Labels: next-token targets (shift left), ignore last position
    ignore_index = -100
    labels = mx.roll(tokens, shift=-1, axis=1)
    labels = labels.astype(mx.int32)
    labels = mx.where(
        mx.arange(T).reshape((1, -1)) == (T - 1),
        mx.array(ignore_index, dtype=labels.dtype),
        labels,
    )

    # Mask supervised positions to only after the prompt
    action_mask = build_action_mask(prompt_lens, T)  # [B, T]
    labels = mx.where(action_mask, labels, mx.array(ignore_index, dtype=labels.dtype))

    # sequence_logprob
    seq_lp = sequence_logprob(model, tokens, labels, ignore_index=ignore_index)
    print("sequence_logprob per sample:", [float(x) for x in seq_lp.tolist()])  # noqa: T201

    # stable_softmax sanity: sum to ~1.0 at a prompt position
    from mlx_genkit import loss_forward

    logits = loss_forward(model, tokens)  # [B, T, V]
    pos0 = max(0, prompt_lens[0] - 1)
    probs = stable_softmax(logits[0, pos0, :], temperature=0.7)
    print("softmax sum:", float(probs.sum().item()))  # noqa: T201

    # Reference model for token_kl: reload same weights for independence
    try:
        ref_model = clone_reference(model)
    except Exception:
        try:
            # Prefer a second load to ensure independence in all environments
            ref_model, _ = mlx_load(args.model)
        except Exception:
            ref_model = model  # fallback; KL should be ~0 regardless

    # EMA update example (no-op if decay=1.0)
    ema_update(ref_model, model, decay=1.0)

    kl = token_kl(model, ref_model, tokens, labels, ignore_index=ignore_index)
    print("token_kl per sample:", [float(x) for x in kl.tolist()])  # noqa: T201


if __name__ == "__main__":
    main()
