from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

from .utils import try_import_mlx, VocabProjection


# ------------------ Tokenizer bridge ------------------


@dataclass
class TokenizerBridge:
    tokenizer: Any
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tk = self.tokenizer
        if hasattr(tk, "encode"):
            # HF or similar
            out = tk.encode(text, add_special_tokens=add_special_tokens)
        else:
            out = tk(text)
        if isinstance(out, dict) and "input_ids" in out:
            out = out["input_ids"]
        return list(out)

    def decode(self, ids: list[int]) -> str:
        tk = self.tokenizer
        if hasattr(tk, "decode"):
            return tk.decode(ids)
        # Fallback
        return "".join(map(str, ids))


def make_tokenizer_bridge(tokenizer: Any) -> TokenizerBridge:
    eos_id = None
    pad_id = None
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
    elif hasattr(tokenizer, "eos_id"):
        eos_id = getattr(tokenizer, "eos_id")
    if hasattr(tokenizer, "pad_token_id"):
        pad_id = getattr(tokenizer, "pad_token_id")
    elif hasattr(tokenizer, "pad_id"):
        pad_id = getattr(tokenizer, "pad_id")
    return TokenizerBridge(tokenizer=tokenizer, eos_token_id=eos_id, pad_token_id=pad_id)


# ------------------ Model adapters ------------------


@dataclass
class ModelComponents:
    model: Any
    inner_model: Any
    layers: list[Any]
    embed: Any
    norm: Optional[Any]
    lm_head: Optional[Any]
    tied_embed: Optional[Any]
    vocab_projection: VocabProjection
    hidden_size: int
    vocab_size: int


def _get_attr(obj, names: list[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def detect_components(model: Any) -> ModelComponents:
    """Resolve common model components across mlx-lm families.

    Finds: inner model, layers list, embedding module, final norm (if present),
    vocabulary projection (lm_head or tied embedding), and derives sizes.
    """
    mx, nn = try_import_mlx()
    # The common MLX-LM model wraps an inner model (.model) that returns hidden states
    inner = _get_attr(model, ["model"]) or model

    layers = _get_attr(inner, ["layers", "transformer", "blocks", "h"]) or []
    # Ensure layers is a list-like of modules
    if hasattr(layers, "__iter__") and not isinstance(layers, list):
        layers = list(layers)

    embed = _get_attr(inner, ["embed_tokens", "tok_embeddings", "embeddings", "wte"])  # type: ignore
    norm = _get_attr(inner, ["norm", "ln_f", "final_layernorm", "ln_out"])  # type: ignore
    lm_head = _get_attr(model, ["lm_head", "output", "output_linear"])  # type: ignore

    # Hidden/vocab sizes
    hidden_size = None
    vocab_size = None
    # Try to infer from weights
    if lm_head is not None and hasattr(lm_head, "weight"):
        w = lm_head.weight
        vocab_size, hidden_size = int(w.shape[0]), int(w.shape[1])
        vocab_proj = VocabProjection(weight=w, transpose=False)
    elif embed is not None and (hasattr(embed, "as_linear") or hasattr(embed, "weight")):
        # Tied embeddings path. Prefer as_linear for quantized embeddings.
        if hasattr(embed, "as_linear"):
            lin = embed.as_linear
            # Validate by creating a dummy to infer shapes lazily if needed
            vocab_proj = VocabProjection(linear=lin)
            # Attempt to infer sizes from module args if available
            try:
                # No reliable way without running; leave sizes as None and infer when used
                hidden_size = getattr(inner.args, "hidden_size", None) or getattr(model.args, "hidden_size", None)
                vocab_size = getattr(inner, "vocab_size", None) or getattr(model, "vocab_size", None)
            except Exception:
                hidden_size = hidden_size or 0
                vocab_size = vocab_size or 0
        else:
            w = embed.weight
            # logits = hidden @ embed.weight.T
            hidden_size, vocab_size = int(w.shape[1]), int(w.shape[0])
            vocab_proj = VocabProjection(weight=w, transpose=True)
    else:
        # Last resort: attempt to find a linear named lm_head in inner
        alt_head = _get_attr(inner, ["lm_head"])  # type: ignore
        if alt_head is None or not hasattr(alt_head, "weight"):
            raise ValueError("Could not resolve vocab projection (lm_head or tied embedding)")
        w = alt_head.weight
        vocab_size, hidden_size = int(w.shape[0]), int(w.shape[1])
        vocab_proj = VocabProjection(weight=w, transpose=False)

    # Fallbacks if sizes are still unknown
    if hidden_size is None:
        hidden_size = getattr(model, "args", None) and getattr(model.args, "hidden_size", None)
    if vocab_size is None:
        vocab_size = getattr(inner, "vocab_size", None) or getattr(model, "vocab_size", None)

    return ModelComponents(
        model=model,
        inner_model=inner,
        layers=layers,
        embed=embed,
        norm=norm,
        lm_head=lm_head,
        tied_embed=embed if lm_head is None else None,
        vocab_projection=vocab_proj,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )


def project_logits(components: ModelComponents, hidden) -> Any:
    return components.vocab_projection.project(hidden)
