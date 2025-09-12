from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .parity_hf import run_torch, run_mlx
from ..api import GenerationConfig


PROMPTS = [
    "Hello parity world",
    "Write a short poem about MLX",
    "Explain beam search in one sentence",
    "List three fruits",
    "Why is the sky blue?",
    "What is 2+2?",
    "Define entropy",
    "Translate 'Bonjour' to English",
]


def compare_tokens(t: List[int], m: List[int]) -> Dict[str, float]:
    n = min(len(t), len(m))
    same = sum(1 for i in range(n) if t[i] == m[i])
    return {
        "length_t": len(t),
        "length_m": len(m),
        "prefix_equal_rate": same / n if n else 0.0,
    }


def run_suite(hf_model: str, mlx_model: str) -> Dict[str, any]:
    results = {"sampling": [], "beam": []}
    # Sampling config
    scfg = GenerationConfig(max_tokens=24, temperature=0.7, top_p=0.95, no_repeat_ngram_size=2)
    # Beam config
    bcfg = GenerationConfig(max_tokens=24, temperature=0.0, num_beams=4, early_stopping=True, length_penalty=0.2)

    for p in PROMPTS:
        t_tok, t_txt = run_torch(hf_model, p, scfg)
        m_tok, m_txt = run_mlx(mlx_model, p, scfg)
        results["sampling"].append(
            {
                "prompt": p,
                "torch_snip": t_txt[:120],
                "mlx_snip": m_txt[:120],
                **compare_tokens(t_tok, m_tok),
            }
        )
    for p in PROMPTS:
        t_tok, t_txt = run_torch(hf_model, p, bcfg)
        m_tok, m_txt = run_mlx(mlx_model, p, bcfg)
        results["beam"].append(
            {
                "prompt": p,
                "torch_snip": t_txt[:120],
                "mlx_snip": m_txt[:120],
                **compare_tokens(t_tok, m_tok),
            }
        )
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--mlx-model", required=True)
    args = ap.parse_args()

    res = run_suite(args.hf_model, args.mlx_model)
    print(json.dumps(res, indent=2))

