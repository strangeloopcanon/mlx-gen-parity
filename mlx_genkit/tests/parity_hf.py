from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlx_lm import load as mlx_load
from mlx_genkit import GenerationConfig, generate


@dataclass
class CompareResult:
    prompt: str
    torch_text: str
    mlx_text: str
    torch_tokens: List[int]
    mlx_tokens: List[int]


def run_torch(model_id: str, prompt: str, cfg: GenerationConfig):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    attn_mask = input_ids.new_ones(input_ids.shape)

    gen_kwargs = dict(
        max_new_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k if cfg.top_k else None,
        repetition_penalty=cfg.repetition_penalty if cfg.repetition_penalty else None,
    )
    if cfg.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = cfg.no_repeat_ngram_size
    if cfg.min_new_tokens:
        gen_kwargs["min_new_tokens"] = cfg.min_new_tokens
    if cfg.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = cfg.eos_token_id
    if cfg.eos_token_ids is not None:
        gen_kwargs["eos_token_id"] = cfg.eos_token_ids

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn_mask,
            do_sample=cfg.temperature > 0,
            **gen_kwargs,
        )
    tokens = out[0].tolist()
    text = tok.decode(tokens)
    return tokens, text


def run_mlx(model_id: str, prompt: str, cfg: GenerationConfig):
    model, tokenizer = mlx_load(model_id)
    res = generate(model, tokenizer, prompt, cfg)
    return res["tokens"], res["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", type=str, required=True)
    ap.add_argument("--mlx-model", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="hello world")
    args = ap.parse_args()

    cfg = GenerationConfig(max_tokens=32, temperature=0.7, top_p=0.95)
    t_tokens, t_text = run_torch(args.hf_model, args.prompt, cfg)
    m_tokens, m_text = run_mlx(args.mlx_model, args.prompt, cfg)

    print("Torch tokens:", t_tokens[-32:])
    print("MLX tokens:", m_tokens[-32:])
    print("Torch text:\n", t_text)
    print("MLX text:\n", m_text)


if __name__ == "__main__":
    main()
