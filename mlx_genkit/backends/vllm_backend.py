from __future__ import annotations

from typing import Any, Optional

from ..config import GenerationConfig
from ..structure.result import GenerateResult


class VLLMGenerateBackend:
    """Placeholder backend for vLLM runners."""

    name = "vllm"

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
            "vLLM backend not yet implemented. Provide an adapter that satisfies GenerateBackend."
        )
