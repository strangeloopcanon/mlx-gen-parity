from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, Optional, Sequence

from .config import GenerationConfig
from .backends.base import resolve_backend
from .structure.result import GenerateResult
from .adapters import detect_components
from .backends.mlx_backend import forward_with_hidden
from .structure.grammar import Grammar
from .structure.adherence import JsonAdherence, StructuredGenerationEngine
from .structure.stream import StreamCallbacks, build_stream_observer


def generate(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    config: GenerationConfig,
    hooks: Optional[Sequence[Any]] = None,
    *,
    json_schema: Optional[dict] = None,
    grammar: Optional[Grammar] = None,
    adherence: Optional[JsonAdherence] = None,
    validators: Optional[Sequence[Any]] = None,
    semantic_checks: Optional[Sequence[Any]] = None,
    on_parse_fail: Optional[Callable[[str, Exception], Optional[str]]] = None,
    on_semantic_fail: Optional[Callable[[dict, Sequence[dict]], Optional[dict]]] = None,
    log_writer: Optional[Callable[[GenerateResult, dict], None]] = None,
    stream_callbacks: Optional[StreamCallbacks] = None,
) -> GenerateResult:
    """Generate text using the configured backend (defaults to MLX).

    Returns a :class:`GenerateResult` carrying the legacy fields while enabling
    downstream structured adherence extensions.
    """

    cfg = copy.deepcopy(config)
    backend = resolve_backend(cfg.backend, model=model, tokenizer=tokenizer)
    cfg.backend = backend.name
    if grammar is None and json_schema is not None:
        grammar = Grammar.json_schema(json_schema)
    cfg.grammar = grammar
    cfg.validators = validators
    cfg.semantic_checks = semantic_checks

    structured = any(
        [
            json_schema,
            validators,
            semantic_checks,
            adherence and adherence.retries,
            adherence and adherence.strict_only_json,
            grammar and grammar.kind != "",
            on_parse_fail,
            on_semantic_fail,
            log_writer,
        ]
    )

    if structured or json_schema is not None:
        adherence_obj = adherence or JsonAdherence()
        stream_observer = (
            build_stream_observer(
                tokenizer=tokenizer,
                callbacks=stream_callbacks,
                adherence=adherence_obj,
                expect_json=bool(json_schema),
            )
            if stream_callbacks
            else None
        )
        engine = StructuredGenerationEngine(
            backend=backend,
            prompt=prompt,
            config=cfg,
            hooks=hooks,
            json_schema=json_schema,
            grammar=grammar,
            validators=validators,
            semantic_checks=semantic_checks,
            adherence=adherence_obj,
            on_parse_fail=on_parse_fail,
            on_semantic_fail=on_semantic_fail,
            log_writer=log_writer,
            stream_observer=stream_observer,
        )
        result = engine.run()
        _populate_meta(result, cfg, model, tokenizer)
        return result

    stream_observer = (
        build_stream_observer(
            tokenizer=tokenizer,
            callbacks=stream_callbacks,
            adherence=adherence,
            expect_json=False,
        )
        if stream_callbacks
        else None
    )
    result = backend.generate(prompt, cfg, hooks=hooks, stream_observer=stream_observer)
    if stream_observer is not None:
        result.meta.setdefault("stream_tokens_emitted", stream_observer.emitted)
        if stream_observer.invalid_triggered:
            result.meta.setdefault("stream_invalid_path", True)
    _populate_meta(result, cfg, model, tokenizer)
    return result


def _populate_meta(result: GenerateResult, cfg: GenerationConfig, model: Any, tokenizer: Any) -> None:
    meta = result.meta
    meta.setdefault("config_digest", _config_digest(cfg))
    if cfg.seed is not None:
        meta.setdefault("seed", cfg.seed)
    meta.setdefault("model_fingerprint", _model_fingerprint(model))
    meta.setdefault("tokenizer_class", getattr(tokenizer, "__class__", type(tokenizer)).__name__)


def _config_digest(cfg: GenerationConfig) -> str:
    data = {
        k: v
        for k, v in cfg.__dict__.items()
        if k not in {"grammar", "validators", "semantic_checks"}
    }
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_fingerprint(model: Any) -> Optional[str]:
    ident = getattr(model, "model_id", None) or getattr(model, "name", None)
    if ident is None:
        ident = f"{model.__class__.__module__}.{model.__class__.__name__}"
    try:
        digest = hashlib.sha1(str(ident).encode("utf-8")).hexdigest()
    except Exception:  # pragma: no cover - defensive
        return None
    return digest


__all__ = [
    "GenerationConfig",
    "generate",
    "forward_with_hidden",
    "detect_components",
]
