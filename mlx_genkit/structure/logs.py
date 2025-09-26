from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .result import GenerateResult


@dataclass
class JsonlLogWriter:
    path: str
    append: bool = True
    include_raw_on_fail: bool = True
    _initialised: bool = field(default=False, init=False, repr=False)

    def __call__(self, result: GenerateResult, context: Optional[Dict[str, Any]] = None) -> None:
        ctx = context or {}
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_ok": result.schema_ok,
            "only_json_ok": result.only_json,
            "semantic_ok": True if result.semantic_ok is None else bool(result.semantic_ok),
            "attempts": result.attempts,
            "meta": result.meta,
            "violations": list(result.violations),
            "context": ctx,
        }
        if not result.schema_ok and self.include_raw_on_fail:
            record["raw_text"] = result.text
        mode = "a"
        if not self._initialised:
            mode = "a" if self.append else "w"
            self._initialised = True
        with open(self.path, mode, encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
