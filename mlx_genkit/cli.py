from __future__ import annotations

import argparse
import json
from typing import List

from .api import GenerationConfig, generate
from .loader import auto_load


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def generate_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.generate", description="Generate text with MLX + generation-parity features")
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--typical-p", type=float, default=0.0)
    ap.add_argument("--epsilon-cutoff", type=float, default=0.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--stop", default=None, help="Comma-separated stop sequences")
    ap.add_argument("--suppress-tokens", default=None, help="Comma-separated token ids to suppress (ints)")
    ap.add_argument("--begin-suppress-tokens", default=None, help="Comma-separated token ids suppressed at first step")
    ap.add_argument("--force-words", default=None, help="Comma-separated phrases to force (joined by tokenizer)")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--draft-model", default=None, help="Draft model id/path for speculative decoding")
    args = ap.parse_args()

    # Auto-load (convert if needed)
    model, tokenizer, local_path = auto_load(args.model)

    force_words_ids = None
    if args.force_words:
        phrases = [p for p in args.force_words.split(",") if p.strip()]
        force_words_ids = [tokenizer.encode(p, add_special_tokens=False) for p in phrases]

    stop_sequences = None
    if args.stop:
        stop_sequences = [s for s in args.stop.split(",") if s]

    cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        typical_p=args.typical_p,
        epsilon_cutoff=args.epsilon_cutoff,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens if args.min_new_tokens > 0 else None,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        stop_sequences=stop_sequences,
        force_words_ids=force_words_ids,
        use_speculative=args.speculative,
        draft_model_id=args.draft_model,
    )

    res = generate(model, tokenizer, args.prompt, cfg)
    print(res["text"])  # noqa: T201
