from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Grammar:
    """Declarative grammar constraint description.

    The backend decides whether a particular grammar kind is supported. Until we
    wire specialised decoders, grammars fall back to post-generation validation.
    """

    kind: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def json_schema(cls, schema: Dict[str, Any], *, dialect: Optional[str] = None) -> "Grammar":
        return cls(
            kind="json_schema",
            payload=schema,
            metadata={"dialect": dialect} if dialect else {},
        )

    @classmethod
    def gbnf(cls, grammar_text: str) -> "Grammar":
        return cls(kind="gbnf", payload=grammar_text)

    @classmethod
    def fst(cls, automaton: Any) -> "Grammar":  # pragma: no cover - backend specific
        return cls(kind="fst", payload=automaton)

    def supports_backend(self, backend_name: str) -> bool:
        """Return True if the grammar is natively supported by the backend."""

        backend_name = backend_name.lower()
        if self.kind == "json_schema":
            return True
        if backend_name == "mlx":
            # MLX backend currently lacks direct grammar support.
            return False
        # Future hooks: transformers/vllm backends may report support.
        return False
