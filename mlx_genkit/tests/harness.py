from __future__ import annotations

import argparse
from mlx_genkit import generate, GenerationConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    p.add_argument("--prompt", type=str, default="hello from harness")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.0)
    args = p.parse_args()

    from mlx_lm import load

    model, tokenizer = load(args.model)
    cfg = GenerationConfig(max_tokens=args.max_tokens, temperature=args.temp)
    out = generate(model, tokenizer, args.prompt, cfg)
    print(out["text"])  # noqa: T201


if __name__ == "__main__":
    main()
