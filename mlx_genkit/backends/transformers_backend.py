from __future__ import annotations

from typing import Any, Optional

from ..config import GenerationConfig
from ..structure.result import GenerateResult


class TransformersGenerateBackend:
    """Placeholder backend for Hugging Face Transformers models."""

    name = "transformers"

    def __init__(self, *, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: Any,
        config: GenerationConfig,
        *,
        hooks: Optional[Any] = None,
        stream_observer: Optional[Any] = None,
    ) -> GenerateResult:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "Transformers backend not yet implemented. Install transformers and provide a suitable runner."
        )
