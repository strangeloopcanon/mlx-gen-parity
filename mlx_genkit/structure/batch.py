from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import GenerationConfig
from ..api import generate
from .result import GenerateResult
from .adherence import JsonAdherence


@dataclass
class BatchResult:
    results: List[GenerateResult]
    summary: Dict[str, Any]

    def ok(self) -> bool:
        return all(r.ok for r in self.results)


def _clone_config(cfg: GenerationConfig) -> GenerationConfig:
    return copy.deepcopy(cfg)


def _apply_overrides(cfg: GenerationConfig, overrides: Dict[str, Any]) -> GenerationConfig:
    updated = _clone_config(cfg)
    for key, value in overrides.items():
        if hasattr(updated, key):
            setattr(updated, key, value)
    return updated


def generate_many(
    model: Any,
    tokenizer: Any,
    items: Sequence[Any],
    *,
    base_config: Optional[GenerationConfig] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    adherence: Optional[JsonAdherence] = None,
    max_concurrency: Optional[int] = None,
    map_fn: Optional[Callable[[GenerateResult], Any]] = None,
    reduce_fn: Optional[Callable[[List[Any]], Any]] = None,
    **kwargs: Any,
) -> Tuple[BatchResult, Optional[Any]]:
    cfg = base_config or GenerationConfig()
    adherence_obj = adherence or JsonAdherence()
    total = len(items)
    results: List[Optional[GenerateResult]] = [None] * total
    mapped: List[Any] = [None] * total if map_fn else []

    def worker(idx: int, payload: Any):
        entry = payload
        prompt = entry
        local_cfg = cfg
        hooks = None
        extra_kwargs: Dict[str, Any] = {}
        if isinstance(entry, dict):
            prompt = entry.get("prompt")
            if prompt is None:
                raise ValueError("Batch item dict must include 'prompt'")
            overrides = entry.get("config") or {}
            if overrides:
                local_cfg = _apply_overrides(cfg, overrides)
            hooks = entry.get("hooks")
            extra_kwargs = {k: v for k, v in entry.items() if k not in {"prompt", "config", "hooks", "meta"}}
        result = generate(
            model,
            tokenizer,
            prompt,
            local_cfg,
            hooks,
            json_schema=json_schema,
            adherence=adherence_obj,
            **{**kwargs, **extra_kwargs},
        )
        if isinstance(entry, dict) and "meta" in entry:
            result.meta.setdefault("batch_item", entry["meta"])
        results[idx] = result
        if map_fn:
            mapped[idx] = map_fn(result)

    max_workers = max_concurrency or min(8, total) or 1
    if max_workers <= 1:
        for idx, item in enumerate(items):
            worker(idx, item)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, idx, item) for idx, item in enumerate(items)]
            for future in as_completed(futures):
                future.result()

    final_results = [r for r in results if r is not None]
    summary = {
        "total": len(final_results),
        "schema_ok": sum(1 for r in final_results if r.schema_ok),
        "semantic_ok": sum(1 for r in final_results if r.semantic_ok in (None, True)),
        "failures": [i for i, r in enumerate(final_results) if not r.ok],
    }
    batch_result = BatchResult(results=final_results, summary=summary)
    reduced = None
    if reduce_fn:
        mapped_values = mapped if map_fn else final_results
        reduced = reduce_fn(mapped_values)
    return batch_result, reduced
