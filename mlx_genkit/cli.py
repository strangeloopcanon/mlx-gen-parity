from __future__ import annotations

import argparse
import json
from typing import List, Any
import os
import sys

from .api import GenerationConfig, generate
from .loader import auto_load, _sanitize_repo_id


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _looks_like_chat_model(model_id_or_path: str) -> bool:
    """Heuristic to decide if a model is a chat/instruct variant.

    Used only for CLI defaulting; users can override with --no-auto-chat.
    """
    t = (model_id_or_path or "").lower()
    # Strong signals
    if "instruct" in t or "-chat" in t or "/chat" in t or "chat-" in t or "-it" in t:
        return True
    # Family-specific loose matches (still safe due to tokenizer-side guard)
    if "llama-3" in t or "llama3" in t:
        return ("instruct" in t) or ("chat" in t) or ("-it" in t)
    if "mistral" in t or "mixtral" in t:
        return ("instruct" in t) or ("-it" in t)
    if "gemma" in t:
        return ("instruct" in t) or ("-it" in t)
    if "phi-3" in t or "phi-4" in t:
        return ("instruct" in t) or ("-it" in t)
    if "qwen" in t:
        return True  # Many Qwen weights ship with chat templates; API guards if absent
    return False


def generate_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.generate", description="Generate text with MLX + generation-parity features")
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--prompt", required=False, help="Prompt text")
    ap.add_argument(
        "--messages-json",
        default=None,
        help="JSON list of chat messages [{role, content}, ...] to apply the model's chat template",
    )
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
    # Chat template controls
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--auto-chat", action="store_true", help="Auto-apply chat template for plain prompts if the tokenizer has one")
    grp.add_argument("--no-auto-chat", action="store_true", help="Disable auto chat templating")
    grp.add_argument("--assume-user-chat", action="store_true", help="Treat a plain prompt as a user message and apply chat template if available")
    ap.add_argument("--system", default=None, help="Optional system prompt when using chat templating")
    ap.add_argument("--stop", default=None, help="Comma-separated stop sequences")
    ap.add_argument(
        "--stop-strings",
        default=None,
        help="Alias for --stop; comma-separated stop strings (will be merged)",
    )
    ap.add_argument("--suppress-tokens", default=None, help="Comma-separated token ids to suppress (ints)")
    ap.add_argument("--begin-suppress-tokens", default=None, help="Comma-separated token ids suppressed at first step")
    ap.add_argument("--force-words", default=None, help="Comma-separated phrases to force (joined by tokenizer)")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--draft-model", default=None, help="Draft model id/path for speculative decoding")
    args = ap.parse_args()

    # Auto-load (convert if needed)
    model, tokenizer, local_path = auto_load(args.model)

    # Determine prompt input: messages (JSON) or plain string
    prompt_input: Any
    if args.messages_json:
        import json

        prompt_input = json.loads(args.messages_json)
    else:
        if args.prompt is None:
            raise SystemExit("Either --prompt or --messages-json must be provided")
        prompt_input = args.prompt

    force_words_ids = None
    if args.force_words:
        phrases = [p for p in args.force_words.split(",") if p.strip()]
        force_words_ids = [tokenizer.encode(p, add_special_tokens=False) for p in phrases]

    # Merge legacy --stop and alias --stop-strings
    stop_sequences = None
    stop_alias = []
    if args.stop:
        stop_alias.extend([s for s in args.stop.split(",") if s])
    if args.stop_strings:
        stop_alias.extend([s for s in args.stop_strings.split(",") if s])
    if stop_alias:
        # de-duplicate, preserve order
        seen = set()
        merged = []
        for s in stop_alias:
            if s not in seen:
                seen.add(s)
                merged.append(s)
        stop_sequences = merged

    # Decide default auto-chat if user didn't set a preference
    default_auto_chat = None
    # Prefer the tokenizer's actual capability (most reliable)
    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if not (args.auto_chat or args.no_auto_chat or args.assume_user_chat):
        default_auto_chat = True if has_template else None

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
        auto_chat_template=(
            True
            if (args.assume_user_chat or args.auto_chat)
            else (False if args.no_auto_chat else default_auto_chat)
        ),
        system_prompt=args.system,
        assume_user_chat=args.assume_user_chat,
        stop_strings=stop_sequences,
        force_words_ids=force_words_ids,
        use_speculative=args.speculative,
        draft_model_id=args.draft_model,
    )

    # Telemetry: report whether chat templating will be applied
    chat_input = "messages" if args.messages_json else "plain"
    # Mirror library logic for plain prompts: assume_user_chat OR (auto_chat_template if not None else has_template)
    final_auto = cfg.assume_user_chat or (cfg.auto_chat_template if cfg.auto_chat_template is not None else has_template)
    applied = has_template and (chat_input == "messages" or final_auto)
    print(
        f"[mlx-genkit] chat templating: applied={'yes' if applied else 'no'} "
        f"(input={chat_input}, tokenizer_template={'yes' if has_template else 'no'}, "
        f"auto={cfg.auto_chat_template}, assume_user={cfg.assume_user_chat})",
        file=sys.stderr,
    )

    res = generate(model, tokenizer, prompt_input, cfg)
    print(res["text"])  # noqa: T201


def download_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.download", description="Download and convert an HF model to MLX format, print local path")
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--cache-dir", default=None, help="Cache directory for converted models (default: ./mlx_cache)")
    ap.add_argument("--quantize", action="store_true", help="Quantize during conversion")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code from the repo")
    ap.add_argument("--force", action="store_true", help="Reconvert even if cached model exists (deletes existing cache dir)")
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join(os.getcwd(), "mlx_cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_name = _sanitize_repo_id(args.model)
    local_path = os.path.join(cache_dir, local_name)

    if args.force and os.path.exists(local_path):
        import shutil

        shutil.rmtree(local_path)

    # Convert without loading into memory
    _m, _t, out_path = auto_load(
        args.model,
        cache_dir=cache_dir,
        quantize=args.quantize,
        trust_remote_code=args.trust_remote_code,
        load_model=False,
    )
    print(out_path)  # noqa: T201
