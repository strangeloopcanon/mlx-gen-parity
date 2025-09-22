from __future__ import annotations

import os
import re
from typing import Optional, Tuple


def _sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id.strip())


def auto_load(
    repo_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    quantize: bool = False,
    trust_remote_code: bool = False,
    load_model: bool = True,
):
    """Load an MLX model/tokenizer from either an MLX path or an HF repo id.

    - If `repo_or_path` already points to an MLX model folder, loads it directly.
    - Otherwise, converts the HF repo to MLX into `cache_dir` (default: ./mlx_cache/<sanitized>)
      using mlx-lm's converter, and then loads it.

    Returns: (model, tokenizer, local_path)
    """
    from mlx_lm import load as mlx_load

    # First try to load directly (if asked to load)
    if load_model:
        try:
            model, tokenizer = mlx_load(repo_or_path, trust_remote_code=trust_remote_code)
            return model, tokenizer, repo_or_path
        except Exception:
            pass
    else:
        # If caller only wants conversion and the path appears already local, return it
        if os.path.isdir(repo_or_path):
            return None, None, repo_or_path

    # Convert path
    cache_dir = cache_dir or os.path.join(os.getcwd(), "mlx_cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_name = _sanitize_repo_id(repo_or_path)
    local_path = os.path.join(cache_dir, local_name)
    if not os.path.exists(local_path):
        # Convert
        from mlx_lm import convert

        convert(repo_or_path, mlx_path=local_path, quantize=quantize, trust_remote_code=trust_remote_code)
    # Load converted if requested
    if load_model:
        model, tokenizer = mlx_load(local_path, trust_remote_code=trust_remote_code)
        return model, tokenizer, local_path
    return None, None, local_path
