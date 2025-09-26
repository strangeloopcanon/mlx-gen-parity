from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Dict, Optional, Sequence

from ..config import GenerationConfig
from ..structure.adherence import JsonAdherence
from ..structure.grammar import Grammar
from ..structure.result import GenerateResult

TokenCallback = Callable[[Any, int], None]
InvalidPathCallback = Callable[[Dict[str, Any]], None]


@dataclass
class StreamCallbacks:
    """User callbacks invoked during streaming generation."""

    on_token: Optional[TokenCallback] = None
    on_invalid_path: Optional[InvalidPathCallback] = None
    stop_on_invalid: bool = True


class IncrementalJsonMonitor:
    """Best-effort incremental JSON validator for streaming output."""

    def __init__(
        self,
        *,
        expect_json: bool,
        strict_only_json: bool,
        stop_on_invalid: bool,
        callback: Optional[InvalidPathCallback],
    ) -> None:
        self.expect_json = expect_json
        self.strict_only_json = strict_only_json
        self.stop_on_invalid = stop_on_invalid
        self.callback = callback
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.triggered = False

    def reset(self) -> None:
        self.buffer = ""
        self.triggered = False

    def feed(self, fragment: str) -> bool:
        if not fragment:
            return not (self.stop_on_invalid and self.triggered)
        self.buffer += fragment
        stripped = self.buffer.strip()
        if not stripped:
            return not (self.stop_on_invalid and self.triggered)

        start_positions = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos != -1]
        start = min(start_positions) if start_positions else -1
        candidate = stripped if start <= 0 or self.strict_only_json else stripped[start:]

        try:
            _, end = self.decoder.raw_decode(candidate)
        except JSONDecodeError as err:
            if err.pos >= len(candidate) - 1 or err.msg.lower().startswith("unterminated string"):
                return not (self.stop_on_invalid and self.triggered)
            if not self.strict_only_json and start > 0 and err.pos < start and not self.expect_json:
                return not (self.stop_on_invalid and self.triggered)
            self._notify(err, stripped)
            return not self.stop_on_invalid

        suffix = candidate[end:].strip()
        if self.strict_only_json and suffix:
            self._notify(JSONDecodeError("Non-JSON content present", candidate, end), stripped)
            return not self.stop_on_invalid
        return not (self.stop_on_invalid and self.triggered)

    def _notify(self, err: JSONDecodeError, buffer: str) -> None:
        if self.triggered:
            return
        self.triggered = True
        if self.callback:
            self.callback({"message": err.msg, "position": err.pos, "buffer": buffer})


class StreamObserver:
    """Bridges backend tokens to callbacks while tracking incremental validation."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        callbacks: StreamCallbacks,
        monitor: Optional[IncrementalJsonMonitor],
    ) -> None:
        self.tokenizer = tokenizer
        self.callbacks = callbacks
        self.monitor = monitor
        self._token_index = 0
        self._emitted = 0

    def reset(self) -> None:
        self._token_index = 0
        self._emitted = 0
        if self.monitor:
            self.monitor.reset()

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def invalid_triggered(self) -> bool:
        return bool(self.monitor and self.monitor.triggered)

    def emit_token(self, token_id: Any) -> bool:
        idx = self._token_index
        self._token_index += 1
        if self.callbacks.on_token:
            self.callbacks.on_token(token_id, idx)
        self._emitted += 1

        fragment = ""
        if self.tokenizer is not None:
            try:
                fragment = self.tokenizer.decode([token_id])
            except Exception:
                fragment = ""
        if self.monitor and fragment:
            return self.monitor.feed(fragment)
        return True


def build_stream_observer(
    *,
    tokenizer: Any,
    callbacks: StreamCallbacks,
    adherence: Optional[JsonAdherence],
    expect_json: bool,
) -> StreamObserver:
    strict_only = bool(adherence and adherence.strict_only_json)
    monitor: Optional[IncrementalJsonMonitor] = None
    if callbacks.on_invalid_path or expect_json or strict_only:
        monitor = IncrementalJsonMonitor(
            expect_json=expect_json,
            strict_only_json=strict_only,
            stop_on_invalid=callbacks.stop_on_invalid,
            callback=callbacks.on_invalid_path,
        )
    observer = StreamObserver(tokenizer=tokenizer, callbacks=callbacks, monitor=monitor)
    observer.reset()
    return observer


def _replay_stream(text: str, callbacks: StreamCallbacks) -> None:
    buffer = ""
    notified = False
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        buffer += ch
        if callbacks.on_token:
            callbacks.on_token(ch, idx)
        if notified or callbacks.on_invalid_path is None:
            continue
        stripped = buffer.strip()
        if not stripped:
            continue
        try:
            decoder.raw_decode(stripped)
        except JSONDecodeError as err:
            if err.pos >= len(stripped) - 1:
                continue
            callbacks.on_invalid_path(
                {
                    "message": err.msg,
                    "position": err.pos,
                    "buffer": stripped,
                }
            )
            notified = True


def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    config: Optional[GenerationConfig] = None,
    *,
    hooks: Optional[Sequence[Any]] = None,
    json_schema: Optional[dict] = None,
    grammar: Optional[Grammar] = None,
    adherence: Optional[JsonAdherence] = None,
    validators: Optional[Sequence[Any]] = None,
    semantic_checks: Optional[Sequence[Any]] = None,
    on_parse_fail: Optional[Callable[[str, Exception], Optional[str]]] = None,
    on_semantic_fail: Optional[Callable[[dict, Sequence[dict]], Optional[dict]]] = None,
    log_writer: Optional[Callable[[GenerateResult, dict], None]] = None,
    on_token: Optional[TokenCallback] = None,
    on_invalid_path: Optional[InvalidPathCallback] = None,
    stop_on_invalid: bool = True,
) -> GenerateResult:
    from ..api import generate  # Local import to avoid circular dependency

    cfg = config or GenerationConfig()
    callbacks = StreamCallbacks(
        on_token=on_token,
        on_invalid_path=on_invalid_path,
        stop_on_invalid=stop_on_invalid,
    )
    result = generate(
        model,
        tokenizer,
        prompt,
        cfg,
        hooks,
        json_schema=json_schema,
        grammar=grammar,
        adherence=adherence,
        validators=validators,
        semantic_checks=semantic_checks,
        on_parse_fail=on_parse_fail,
        on_semantic_fail=on_semantic_fail,
        log_writer=log_writer,
        stream_callbacks=callbacks,
    )

    if callbacks.on_token and not result.meta.get("stream_tokens_emitted", 0):
        _replay_stream(result.text or "", callbacks)
    return result
