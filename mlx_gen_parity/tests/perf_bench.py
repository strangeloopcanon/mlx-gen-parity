from __future__ import annotations

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlx_lm import load as mlx_load
from ..api import GenerationConfig, generate


def bench_torch(model_id: str, prompt: str, cfg: GenerationConfig) -> Tuple[float, int]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    attn_mask = input_ids.new_ones(input_ids.shape)
    gen_kwargs = dict(
        max_new_tokens=cfg.max_tokens,
        do_sample=cfg.temperature > 0,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn_mask, **gen_kwargs)
    dt = time.perf_counter() - t0
    new_tokens = out.shape[-1] - input_ids.shape[-1]
    tps = new_tokens / dt if dt > 0 else 0.0
    return tps, new_tokens


def bench_mlx(model_id: str, prompt: str, cfg: GenerationConfig) -> Tuple[float, int]:
    model, tokenizer = mlx_load(model_id)
    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt, cfg)
    dt = time.perf_counter() - t0
    new_tokens = len(out["tokens"]) - len(tokenizer.encode(prompt))
    tps = new_tokens / dt if dt > 0 else 0.0
    return tps, new_tokens


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("--prompt", default="Hello performance")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    cfg = GenerationConfig(max_tokens=args.max_tokens, temperature=0.7, top_p=0.95)
    tps_t, n_t = bench_torch(args.hf_model, args.prompt, cfg)
    tps_m, n_m = bench_mlx(args.mlx_model, args.prompt, cfg)
    print(f"Torch+MPS: {tps_t:.2f} tok/s ({n_t} new)\nMLX: {tps_m:.2f} tok/s ({n_m} new)")

