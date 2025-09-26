from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class GenerateResult:
    """Primary return type for generation calls.

    Behaves like the previous dict-based return for backward compatibility while
    carrying placeholders for structured adherence metadata populated in later
    phases.
    """

    text: str
    tokens: Optional[Sequence[int]] = None
    eos_reached: Optional[bool] = None
    finish_reason: Optional[str] = None
    attempts: int = 1
    json: Any = None
    schema_ok: bool = False
    only_json: bool = False
    semantic_ok: Optional[bool] = None
    violations: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    reflection: Optional["GenerateResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "eos_reached": self.eos_reached,
            "finish_reason": self.finish_reason,
            "attempts": self.attempts,
            "json": self.json,
            "schema_ok": self.schema_ok,
            "only_json": self.only_json,
            "semantic_ok": self.semantic_ok,
            "violations": list(self.violations),
            "meta": dict(self.meta),
            "reflection": self.reflection.to_dict() if isinstance(self.reflection, GenerateResult) else self.reflection,
        }

    # Dict-like compatibility -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    # Convenience helpers -----------------------------------------------------
    @property
    def ok(self) -> bool:
        """Best-effort success flag combining schema and semantic checks."""
        schema_required = bool(self.meta.get("schema_required"))
        if schema_required:
            schema_ok = bool(self.schema_ok)
        else:
            schema_ok = self.schema_ok or self.json is None
        semantic_ok = True if self.semantic_ok is None else bool(self.semantic_ok)
        return schema_ok and semantic_ok

    # Reflection -------------------------------------------------------------
    def reflect(
        self,
        *,
        model: Any,
        tokenizer: Any,
        schema: Optional[Dict[str, Any]] = None,
        spec: Optional[Any] = None,
        task: str = "Reflect on the previous output and suggest improvements.",
        config: Optional[Any] = None,
        adherence: Optional[Any] = None,
        **kwargs: Any,
    ) -> "GenerateResult":
        if spec is None and schema is None:
            raise ValueError("Either `schema` or `spec` must be provided for reflection")
        if spec is None:
            from .dsl import StructuredSpec

            fields = list(schema.get("properties", {}).keys()) if schema and isinstance(schema, dict) else []
            spec = StructuredSpec(schema=schema or {}, fields=fields, name="reflection")
        from .dsl import generate_structured

        reflection = generate_structured(
            model,
            tokenizer,
            task=task,
            spec=spec,
            config=config,
            adherence=adherence,
            **kwargs,
        )
        self.reflection = reflection
        return reflection
