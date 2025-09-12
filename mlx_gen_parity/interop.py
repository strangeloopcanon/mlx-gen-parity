from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def convert_hf_to_mlx(hf_repo: str, *, quantize: bool = False, upload_repo: Optional[str] = None, local_out: Optional[str] = None):
    """Thin wrapper over mlx_lm.convert to produce an MLX model locally.

    Args:
        hf_repo: Hugging Face model repo id, e.g. 'Qwen/Qwen3-0.6B'.
        quantize: Whether to quantize during conversion.
        upload_repo: Optional repo to upload the converted model.
        local_out: Optional local path to write the MLX weights (default: 'mlx_model').
    """
    from mlx_lm import convert

    mlx_path = local_out or 'mlx_model'
    return convert(hf_repo, mlx_path=mlx_path, quantize=quantize, upload_repo=upload_repo)


def apply_chat_template(tokenizer: Any, messages, add_generation_prompt: bool = True, **kwargs) -> str:
    """Apply HF-style chat template if available, else fall back to simple join.

    messages: list of {role, content}
    Returns a prompt string.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, **kwargs)
    # Fallback
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)
