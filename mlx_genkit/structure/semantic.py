from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol


class SemanticCheck(Protocol):
    name: str

    def evaluate(self, data: Any) -> Optional[str]:
        """Return None when the check passes, otherwise an error message."""


@dataclass
class SemanticViolation:
    check: str
    message: str


def _get_field(data: Any, field: str) -> Any:
    parts = field.split(".")
    cur = data
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


@dataclass
class MustContain:
    field: str
    substrings: Iterable[str]
    name: str = "must_contain"

    def evaluate(self, data: Any) -> Optional[str]:
        value = _get_field(data, self.field)
        if value is None:
            return f"Field '{self.field}' missing"
        text = str(value)
        missing = [s for s in self.substrings if s not in text]
        if missing:
            return f"Field '{self.field}' missing substrings: {missing}"
        return None


@dataclass
class EnumIn:
    field: str
    allowed: Iterable[str]
    name: str = "enum_in"

    def evaluate(self, data: Any) -> Optional[str]:
        value = _get_field(data, self.field)
        if value is None:
            return f"Field '{self.field}' missing"
        allowed = list(self.allowed)
        if value not in allowed:
            return f"Field '{self.field}' expected one of {allowed}, got {value!r}"
        return None


@dataclass
class RegexOnField:
    field: str
    pattern: str
    flags: int = 0
    name: str = "regex_on_field"
    _compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, self.flags)

    def evaluate(self, data: Any) -> Optional[str]:
        value = _get_field(data, self.field)
        if value is None:
            return f"Field '{self.field}' missing"
        text = str(value)
        if not self._compiled.search(text):
            return f"Field '{self.field}' failed to match pattern {self.pattern!r}"
        return None


def run_semantic_checks(data: Any, checks: Optional[Iterable[SemanticCheck]]) -> List[SemanticViolation]:
    violations: List[SemanticViolation] = []
    if not checks:
        return violations
    for check in checks:
        message = check.evaluate(data)
        if message:
            violations.append(SemanticViolation(check=getattr(check, "name", type(check).__name__), message=message))
    return violations


_SEMANTIC_BUILDERS = {
    "must_contain": lambda spec: MustContain(field=spec["field"], substrings=spec.get("substrings", [])),
    "enum_in": lambda spec: EnumIn(field=spec["field"], allowed=spec.get("allowed", [])),
    "regex_on_field": lambda spec: RegexOnField(
        field=spec["field"], pattern=spec["pattern"], flags=spec.get("flags", 0)
    ),
}


def build_semantic_check(spec: Dict[str, Any]) -> SemanticCheck:
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError("Semantic check spec must be an object with a 'type'")
    builder = _SEMANTIC_BUILDERS.get(spec["type"])
    if not builder:
        raise ValueError(f"Unsupported semantic check type: {spec['type']}")
    return builder(spec)


def build_semantic_checks(specs: Iterable[Dict[str, Any]]) -> List[SemanticCheck]:
    return [build_semantic_check(spec) for spec in specs]
