from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

from .adapters import make_tokenizer_bridge, detect_components, project_logits, ModelComponents
from .injection import ResidualInjectionHook, LogitBiasHook, ResidualInjector
from .sampling import (
    make_sampler,
    make_logits_processor_chain,
    make_force_words_processor,
)
from .utils import (
    try_import_mlx,
    try_import_mlx_lm_cache,
    set_seed,
    stable_log_softmax,
    as_mx_array,
)


@dataclass
class GenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    typical_p: float = 0.0
    epsilon_cutoff: float = 0.0
    repetition_penalty: float = 0.0
    repetition_context_size: int = 20
    no_repeat_ngram_size: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    eos_token_id: Optional[int] = None
    eos_token_ids: Optional[List[int]] = None
    pad_token_id: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    # Alias for convenience; merged with stop_sequences if provided
    stop_strings: Optional[List[str]] = None
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    min_new_tokens: Optional[int] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    forced_decoder_ids: Optional[List[Tuple[int, int]]] = None  # [(pos, token_id)] relative to new tokens
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    # Speculative decoding
    use_speculative: bool = False
    draft_model_id: Optional[str] = None
    num_draft_tokens: int = 3
    # KV cache window (sliding)
    max_kv_size: Optional[int] = None
    # Beam search
    num_beams: int = 1
    length_penalty: float = 0.0
    early_stopping: bool = False
    # Chat template
    auto_chat_template: Optional[bool] = None  # None: auto if tokenizer has chat_template; True/False to force
    system_prompt: Optional[str] = None  # Optional system message when auto-applying chat template
    assume_user_chat: bool = False  # Treat plain string as a user message for chat templating


def _prepare_prompt(tokenizer, prompt: Union[str, Sequence[int]]):
    tk = make_tokenizer_bridge(tokenizer)
    if isinstance(prompt, str):
        ids = tk.encode(prompt)
    else:
        ids = list(prompt)
    return ids, tk


from typing import Optional as _Optional  # local alias to avoid collision


def _maybe_render_chat_prompt(tokenizer: Any, prompt: Any, config: _Optional[GenerationConfig] = None) -> Any:
    """If prompt looks like HF-style chat `messages`, render with chat template.

    Accepts a list of {role, content} dicts (or tuples) and returns a string
    produced via `tokenizer.apply_chat_template` when available, otherwise a
    simple fallback formatting. If `prompt` is already a string or a list of
    ints, it is returned unchanged.
    """
    # Fast-path: strings and explicit token id sequences are left unchanged
    if isinstance(prompt, str):
        # Optional auto-application for plain strings
        auto = None
        if config is not None:
            # assume_user_chat explicitly forces chat templating for plain prompts
            if getattr(config, "assume_user_chat", False):
                auto = True
            else:
                auto = config.auto_chat_template
        # Heuristic default: if tokenizer exposes a non-empty chat_template, assume chat model
        if auto is None:
            auto = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
        if auto:
            try:
                messages = []
                if config and getattr(config, "system_prompt", None):
                    messages.append({"role": "system", "content": config.system_prompt})
                messages.append({"role": "user", "content": prompt})
                from .interop import apply_chat_template  # local import

                return apply_chat_template(tokenizer, messages, add_generation_prompt=True)
            except Exception:
                # Fallback: return unchanged
                return prompt
        return prompt
    if isinstance(prompt, (list, tuple)) and all(isinstance(x, int) for x in prompt):
        return prompt
    # Detect list of chat message dicts
    is_messages = False
    if isinstance(prompt, (list, tuple)) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and ("role" in first and "content" in first):
            is_messages = True
    if not is_messages:
        return prompt
    # Render using interop helper
    try:
        from .interop import apply_chat_template  # local import to avoid cycles

        return apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
    except Exception:
        # If anything goes wrong, just pass prompt through
        return prompt


def _maybe_make_prompt_cache(model, max_kv_size: Optional[int] = None):
    lm_cache = try_import_mlx_lm_cache()
    if lm_cache is None:
        return None
    try:
        return lm_cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    except Exception:
        return None


def _step_logits(model, components: ModelComponents, input_tokens, cache=None, input_embeddings=None):
    # Returns logits for last position and possibly updated cache
    # Some models do not support input_embeddings kwarg
    if input_embeddings is not None:
        try:
            out = model(input_tokens, cache=cache, input_embeddings=input_embeddings)  # type: ignore
        except TypeError:
            # Fallback: ignore input_embeddings
            out = model(input_tokens, cache=cache)  # type: ignore
    else:
        out = model(input_tokens, cache=cache)  # type: ignore
    # If model returns logits already, assume shape [B, L, V]
    logits = out
    # Some models may return hidden states prior to head; detect via shape heuristic
    if logits.ndim == 3 and logits.shape[-1] == components.hidden_size and components.vocab_size != components.hidden_size:
        hidden = logits
        logits = project_logits(components, hidden)
    return logits[:, -1, :], cache


def _batched_last_logits(
    model,
    components: ModelComponents,
    batch_token_lists: List[List[int]],
):
    """Compute last-position logits for a batch of token sequences.

    Assumes all sequences are the same length (true within beam steps).
    """
    mx, _ = try_import_mlx()
    arr = as_mx_array(batch_token_lists, dtype=mx.int32)
    logits, _ = _step_logits(model, components, arr, cache=None)
    return logits


def forward_with_hidden(
    model, tokenizer, tokens: Sequence[int], capture_layers: Optional[List[int]] = None, strict: bool = False
):
    """Forward tokens and optionally capture hidden states at specific layers.

    Note: With MLX compilation, Python-level patches may not trigger in some
    optimized paths. Captures can therefore be empty depending on model/compile
    settings. Logits are always returned for the final position.
    """
    mx, _ = try_import_mlx()
    components = detect_components(model)
    ids = as_mx_array(tokens, dtype=mx.int32)
    cache = None if strict else _maybe_make_prompt_cache(model)

    captured: Dict[int, Any] = {}
    if capture_layers and not strict:
        inj = ResidualInjector(components.layers, components.hidden_size)
        # Instrumentation: patch a no-op that captures outputs at specified layers
        to_capture = []
        for l in capture_layers:
            idx = l if l >= 0 else len(components.layers) + l
            if 0 <= idx < len(components.layers):
                to_capture.append(idx)

        class _Capture:
            def __init__(self, layer, idx):
                self.layer = layer
                self.idx = idx
                self._orig = layer.__call__

            def __call__(self, x, *args, **kwargs):
                out = self._orig(x, *args, **kwargs)
                captured[self.idx] = out
                return out

            def apply(self):
                self.layer.__call__ = self.__call__  # type: ignore

            def restore(self):
                self.layer.__call__ = self._orig  # type: ignore

        patches = []
        for idx in to_capture:
            p = _Capture(components.layers[idx], idx)
            p.apply()
            patches.append(p)
        logits, _ = _step_logits(model, components, ids[None], cache)
        for p in patches:
            p.restore()
    elif strict:
        # Manual forward with proper causal mask and no cache
        try:
            from mlx_lm.models.base import create_attention_mask  # type: ignore
        except Exception:
            create_attention_mask = None
        h = components.embed(ids[None])
        mask = create_attention_mask(h, None) if create_attention_mask is not None else "causal"
        to_capture = set()
        if capture_layers:
            for l in capture_layers:
                idx = l if l >= 0 else len(components.layers) + l
                if 0 <= idx < len(components.layers):
                    to_capture.add(idx)
        for i, layer in enumerate(components.layers):
            h = layer(h, mask, cache=None)
            if i in to_capture:
                captured[i] = h
        if components.norm is not None:
            h = components.norm(h)
        logits_full = project_logits(components, h)
        logits = logits_full[:, -1, :]
    else:
        logits, _ = _step_logits(model, components, ids[None], cache)

    return logits, captured


def _compute_bias_from_vector(components: ModelComponents, vec, alpha: float):
    # bias = (W @ v) scaled by alpha
    proj = components.vocab_projection
    bias = proj.project(vec.reshape((1, -1))).reshape((-1,))
    mx, _ = try_import_mlx()
    return (bias * alpha).astype(mx.float32)


def generate(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, Sequence[int]],
    config: GenerationConfig,
    hooks: Optional[List[Union[ResidualInjectionHook, LogitBiasHook]]] = None,
) -> Dict[str, Any]:
    """HF-compatible sampling interface over MLX models.

    - Applies processors (repetition penalty, no-repeat-ngrams, bad-words, etc.).
    - Supports forced BOS/EOS and forced decoder ids.
    - Works with explicit lm_head or tied/quantized embeddings.
    """
    mx, _ = try_import_mlx()
    set_seed(config.seed)

    # If prompt is chat messages, first render with chat template (if available)
    prompt = _maybe_render_chat_prompt(tokenizer, prompt, config)
    ids, tk = _prepare_prompt(tokenizer, prompt)
    components = detect_components(model)
    eos_ids: List[int] = []
    # Resolve EOS ids list
    if config.eos_token_ids is not None:
        eos_ids = [i for i in config.eos_token_ids if i is not None]
    else:
        if config.eos_token_id is not None:
            eos_ids = [config.eos_token_id]
        elif tk.eos_token_id is not None:
            eos_ids = [tk.eos_token_id]

    # Prepare logits processors
    bias_vec = None
    if hooks:
        for h in hooks:
            if isinstance(h, LogitBiasHook):
                vec = h.resolve_vector(model)
                b = _compute_bias_from_vector(components, vec, h.alpha)
                bias_vec = b if bias_vec is None else (bias_vec + b)

    forced_map = {pos: tid for (pos, tid) in (config.forced_decoder_ids or [])}
    processors = make_logits_processor_chain(
        repetition_penalty=config.repetition_penalty,
        repetition_context_size=config.repetition_context_size,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        bias_vector=bias_vec,
        bad_words_ids=config.bad_words_ids,
        min_new_tokens=config.min_new_tokens,
        eos_token_ids=eos_ids,
        prompt_len=len(ids),
        suppress_tokens=config.suppress_tokens,
        begin_suppress_tokens=config.begin_suppress_tokens,
        forced_decoder_map=forced_map,
        forced_bos_token_id=config.forced_bos_token_id,
    )
    if config.force_words_ids:
        processors.append(
            make_force_words_processor(
                config.force_words_ids, prompt_len=len(ids), strict_start=True
            )
        )
    sampler = make_sampler(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        min_tokens_to_keep=config.min_tokens_to_keep,
        typical_p=config.typical_p,
        epsilon_cutoff=config.epsilon_cutoff,
    )

    # Optional residual injection (sampling path only; not used in beam)
    injector = (
        ResidualInjector(components.layers, components.hidden_size)
        if (hooks and any(isinstance(h, ResidualInjectionHook) for h in hooks))
        else None
    )
    residual_hooks = [h for h in (hooks or []) if isinstance(h, ResidualInjectionHook)]

    # Try to use mlx-lm cache if available, otherwise fallback to no-cache
    prompt_cache = _maybe_make_prompt_cache(model, max_kv_size=config.max_kv_size)

    # Merge stop sequences from both fields (deduplicated, order-preserving)
    raw_stops: List[str] = []
    if config.stop_sequences:
        raw_stops.extend([s for s in config.stop_sequences if s])
    if config.stop_strings:
        for s in config.stop_strings:
            if s and s not in raw_stops:
                raw_stops.append(s)

    # Precompute token-level stop sequences if provided
    stop_token_seqs: List[List[int]] = []
    if raw_stops:
        for s in raw_stops:
            ss = s or ""
            if ss:
                stop_token_seqs.append(make_tokenizer_bridge(tokenizer).encode(ss, add_special_tokens=False))

    # Beam search path
    if config.num_beams and config.num_beams > 1:
        return _beam_search_generate(
            model=model,
            tokenizer=tokenizer,
            components=components,
            prompt_ids=ids,
            processors=processors,
            eos_ids=eos_ids,
            config=config,
            stop_token_seqs=stop_token_seqs,
        )

    tokens: List[int] = list(ids)
    y = as_mx_array(tokens, dtype=mx.int32)[None]
    n_generated = 0
    text_out = ""
    eos_reached = False
    finish_reason: Optional[str] = None

    # Handle forced BOS token at first generation step
    forced_bos = config.forced_bos_token_id
    # Map forced decoder ids
    forced_map = {pos: tid for (pos, tid) in (config.forced_decoder_ids or [])}

    # Fast path using mlx-lm generate_step when possible (no residual injection)
    if injector is None and not (config.num_beams and config.num_beams > 1) and not config.use_speculative:
        try:
            from mlx_lm.generate import generate_step as mlx_generate_step  # type: ignore
        except Exception:
            mlx_generate_step = None
        if mlx_generate_step is not None:
            sampler = make_sampler(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                min_tokens_to_keep=config.min_tokens_to_keep,
                typical_p=config.typical_p,
                epsilon_cutoff=config.epsilon_cutoff,
            )

            tokens: List[int] = list(ids)
            n_generated = 0
            eos_reached = False
            finish_reason = None

            # Use mlx-lm generate_step and our processors
            gen = mlx_generate_step(
                as_mx_array(tokens, dtype=mx.int32),
                model,
                max_tokens=config.max_tokens,
                sampler=sampler,
                logits_processors=processors,
                max_kv_size=config.max_kv_size,
                prompt_cache=prompt_cache,
            )
            for y, logprobs in gen:
                try:
                    t = int(y.item())
                except Exception:
                    t = int(y)
                tokens.append(t)
                n_generated += 1

                # EOS checks
                if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
                    eos_reached = True
                    finish_reason = "eos"
                    break
                if eos_ids and t in eos_ids:
                    eos_reached = True
                    finish_reason = "eos"
                    break
                # Token-level stop sequences
                if stop_token_seqs:
                    hit = False
                    for seq in stop_token_seqs:
                        n = len(seq)
                        if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                            tokens = tokens[:-n]
                            eos_reached = True
                            finish_reason = "stop_sequence"
                            hit = True
                            break
                    if hit:
                        break
                # String fallback stops on generated suffix
                if raw_stops:
                    gen_tokens = tokens[len(ids):]
                    if gen_tokens:
                        decoded_gen = tk.decode(gen_tokens)
                        for s in raw_stops:
                            if s and s in decoded_gen:
                                idx = decoded_gen.find(s)
                                text_trimmed = decoded_gen[:idx]
                                trimmed_gen_ids = make_tokenizer_bridge(tokenizer).encode(
                                    text_trimmed, add_special_tokens=False
                                )
                                tokens = tokens[: len(ids)] + trimmed_gen_ids
                                eos_reached = True
                                finish_reason = "stop_sequence"
                                break
                        if eos_reached:
                            break
                if n_generated >= config.max_tokens:
                    finish_reason = "length"
                    break

            text_out = tk.decode(tokens)
            return {
                "text": text_out,
                "tokens": tokens,
                "eos_reached": eos_reached,
                "finish_reason": finish_reason or ("length" if n_generated >= config.max_tokens else None),
            }

    # Speculative decoding path
    if config.use_speculative:
        try:
            from mlx_lm.generate import speculative_generate_step  # type: ignore
        except Exception:
            speculative_generate_step = None
        if speculative_generate_step is None:
            # Fallback to normal path if not available
            pass
        else:
            # Load draft model if provided via config
            draft_model = None
            if config.draft_model_id:
                try:
                    # Use auto_load to support HF repo ids with on-demand conversion
                    from .loader import auto_load  # type: ignore

                    draft_model, _tk, _local = auto_load(config.draft_model_id)
                except Exception:
                    draft_model = None
            if draft_model is None:
                # If no draft model, fallback to normal path
                pass
            else:
                # Build sampler on normalized logprobs
                sampler = make_sampler(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    min_p=config.min_p,
                    min_tokens_to_keep=config.min_tokens_to_keep,
                    typical_p=config.typical_p,
                    epsilon_cutoff=config.epsilon_cutoff,
                )
                tokens = list(ids)
                n_generated = 0
                eos_reached = False
                finish_reason = None
                # no prompt_cache rotation in this path beyond default
                gen = speculative_generate_step(
                    as_mx_array(tokens, dtype=mx.int32),
                    model,
                    draft_model,
                    num_draft_tokens=config.num_draft_tokens,
                    max_tokens=config.max_tokens,
                    sampler=sampler,
                    logits_processors=processors,
                    prompt_cache=None,
                )
                for y, logprobs, _from_draft in gen:
                    try:
                        t = int(y.item())
                    except Exception:
                        t = int(y)
                    tokens.append(t)
                    n_generated += 1
                    # EOS checks
                    if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
                        eos_reached = True
                        finish_reason = "eos"
                        break
                    if eos_ids and t in eos_ids:
                        eos_reached = True
                        finish_reason = "eos"
                        break
                    # Token-level stop sequences
                    if stop_token_seqs:
                        hit = False
                        for seq in stop_token_seqs:
                            n = len(seq)
                            if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                                tokens = tokens[:-n]
                                eos_reached = True
                                finish_reason = "stop_sequence"
                                hit = True
                                break
                        if hit:
                            break
                    # String fallback stops on generated suffix
                    if raw_stops:
                        gen_tokens = tokens[len(ids):]
                        if gen_tokens:
                            decoded_gen = tk.decode(gen_tokens)
                            for s in raw_stops:
                                if s and s in decoded_gen:
                                    idx = decoded_gen.find(s)
                                    text_trimmed = decoded_gen[:idx]
                                    trimmed_gen_ids = make_tokenizer_bridge(tokenizer).encode(
                                        text_trimmed, add_special_tokens=False
                                    )
                                    tokens = tokens[: len(ids)] + trimmed_gen_ids
                                    eos_reached = True
                                    finish_reason = "stop_sequence"
                                    break
                            if eos_reached:
                                break
                    if n_generated >= config.max_tokens:
                        finish_reason = "length"
                        break
                text_out = tk.decode(tokens)
                return {
                    "text": text_out,
                    "tokens": tokens,
                    "eos_reached": eos_reached,
                    "finish_reason": finish_reason or ("length" if n_generated >= config.max_tokens else None),
                }

    # generation loop
    while n_generated < config.max_tokens:
        # Patch residual injection for this step if present
        if injector and residual_hooks:
            try:
                injector.patch(residual_hooks, n_generated, batch=1, seq_len=y.shape[1], model=model)
            except Exception:
                injector.restore()
                injector = None  # fallback to no injection

        logits, prompt_cache = _step_logits(model, components, y, cache=prompt_cache)

        # Restore after step
        if injector and residual_hooks:
            injector.restore()

        logprobs = stable_log_softmax(logits)

        # Apply processors (HF-compatible order)
        for proc in processors:
            logits = proc(tokens, logits)
        logprobs = stable_log_softmax(logits)

        if forced_bos is not None and n_generated == 0:
            next_tok = as_mx_array([forced_bos], dtype=mx.int32)
        elif n_generated in forced_map:
            next_tok = as_mx_array([forced_map[n_generated]], dtype=mx.int32)
        else:
            next_tok = sampler(logprobs)
        mx.eval(next_tok)
        t = int(next_tok.item())
        tokens.append(t)
        n_generated += 1
        y = as_mx_array(tokens, dtype=mx.int32)[None]

        # Forced EOS token id
        if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
            eos_reached = True
            finish_reason = "eos"
            break

        # EOS by any eos id
        if eos_ids and t in eos_ids:
            eos_reached = True
            finish_reason = "eos"
            break

        # Token-level stop sequences
        if stop_token_seqs:
            for seq in stop_token_seqs:
                n = len(seq)
                if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                    # Trim the stop sequence from tokens
                    tokens = tokens[:-n]
                    eos_reached = True
                    finish_reason = "stop_sequence"
                    break
            if eos_reached:
                break
        # String-level fallback stop sequences
        if raw_stops:
            # Only consider generated segment for stop matching
            gen_tokens = tokens[len(ids):]
            hit = False
            if gen_tokens:
                decoded_gen = tk.decode(gen_tokens)
                for s in raw_stops:
                    if s and s in decoded_gen:
                        idx = decoded_gen.find(s)
                        text_trimmed = decoded_gen[:idx]
                        # Re-tokenize trimmed generated text
                        trimmed_gen_ids = make_tokenizer_bridge(tokenizer).encode(text_trimmed, add_special_tokens=False)
                        tokens = tokens[:len(ids)] + trimmed_gen_ids
                        eos_reached = True
                        finish_reason = "stop_sequence"
                        hit = True
                        break
                if hit:
                    break

    if not finish_reason and n_generated >= config.max_tokens:
        finish_reason = "length"
    text_out = tk.decode(tokens)
    return {
        "text": text_out,
        "tokens": tokens,
        "eos_reached": eos_reached,
        "finish_reason": finish_reason,
    }


def _apply_forced_token_mask(logits, forced_token_id: Optional[int]):
    if forced_token_id is None:
        return logits
    mx, _ = try_import_mlx()
    V = logits.shape[-1]
    mask = -mx.inf * mx.ones_like(logits)
    mask[..., forced_token_id] = 0.0
    return logits + mask


def _beam_search_generate(
    *,
    model,
    tokenizer,
    components: ModelComponents,
    prompt_ids: List[int],
    processors: List,
    eos_ids: List[int],
    config: GenerationConfig,
    stop_token_seqs: List[List[int]],
):
    mx, _ = try_import_mlx()
    num_beams = int(config.num_beams)
    max_new = int(config.max_tokens)
    length_penalty = float(config.length_penalty or 0.0)
    forced_bos = config.forced_bos_token_id
    forced_map = {pos: tid for (pos, tid) in (config.forced_decoder_ids or [])}

    # beams: list of (tokens, score, finished)
    beams: List[Tuple[List[int], float, bool]] = [(list(prompt_ids), 0.0, False)]
    finished: List[Tuple[List[int], float]] = []
    n_generated = 0

    # Optional residual injection for beam path
    beam_injector = None
    residual_hooks = []
    try:
        from .injection import ResidualInjectionHook, ResidualInjector  # type: ignore
        residual_hooks = [h for h in getattr(config, 'hooks', []) or [] if isinstance(h, ResidualInjectionHook)]
        if residual_hooks:
            beam_injector = ResidualInjector(components.layers, components.hidden_size)
    except Exception:
        residual_hooks = []
        beam_injector = None

    while n_generated < max_new:
        # Prepare alive beams
        alive = [(t, s) for (t, s, f) in beams if not f]
        if not alive:
            break
        seqs = [t for (t, s) in alive]
        # Compute logits for each alive beam as a batch
        if beam_injector and residual_hooks:
            try:
                # Patch for current step using batch and seq len
                beam_injector.patch(residual_hooks, n_generated, batch=len(seqs), seq_len=len(seqs[0]), model=model)
                logits_batch = _batched_last_logits(model, components, seqs)
            finally:
                beam_injector.restore()
        else:
            logits_batch = _batched_last_logits(model, components, seqs)
        # For each beam, apply processors and constraints
        candidates: List[Tuple[float, int, int]] = []  # (new_score, beam_index, token)
        for i, (tokens, score) in enumerate(alive):
            logits = logits_batch[i : i + 1, :]
            for proc in processors:
                logits = proc(tokens, logits)
            # Forced tokens
            pos = n_generated
            forced_tok = None
            if forced_bos is not None and n_generated == 0:
                forced_tok = forced_bos
            elif pos in forced_map:
                forced_tok = forced_map[pos]
            if forced_tok is not None:
                logits = _apply_forced_token_mask(logits, forced_tok)
            logprobs = stable_log_softmax(logits)
            # Select top-k for each beam to limit combinatorics
            k = num_beams
            # argsort ascending, take last k for top values
            idxs = mx.argsort(logprobs, axis=-1)[:, -k:]
            vals = mx.take_along_axis(logprobs, idxs, axis=-1)
            vals_list = [float(v.item()) for v in vals.reshape((-1,))]
            idxs_list = idxs.reshape((-1,)).tolist()
            for v, tok in zip(vals_list, idxs_list):
                candidates.append((score + v, i, int(tok)))

        # Select overall top beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        new_beams: List[Tuple[List[int], float, bool]] = []
        for new_score, i_beam, tok in candidates:
            if len(new_beams) >= num_beams:
                break
            base_tokens, base_score = alive[i_beam]
            new_tokens = base_tokens + [tok]
            # Finish checks: eos ids or stop sequences
            is_finish = False
            trimmed_tokens = new_tokens
            if (eos_ids and tok in eos_ids) or (
                config.forced_eos_token_id is not None and tok == config.forced_eos_token_id
            ):
                is_finish = True
            if not is_finish and stop_token_seqs:
                for seq in stop_token_seqs:
                    n = len(seq)
                    if n > 0 and len(new_tokens) >= n and new_tokens[-n:] == seq:
                        trimmed_tokens = new_tokens[:-n]
                        is_finish = True
                        break
            new_beams.append((trimmed_tokens if is_finish else new_tokens, new_score, is_finish))

        beams = new_beams
        # Move finished beams out, keep up to num_beams alive
        alive_next: List[Tuple[List[int], float, bool]] = []
        for t, s, f in beams:
            if f:
                length = max(1, len(t) - len(prompt_ids))
                norm = (length ** (length_penalty)) if length_penalty != 0.0 else 1.0
                finished.append((t, s / norm))
            else:
                alive_next.append((t, s, f))
        # Keep best alive beams
        alive_next.sort(key=lambda x: x[1], reverse=True)
        beams = alive_next[: num_beams]

        n_generated += 1
        # Early stop if enough finished
        if config.early_stopping and len(finished) >= num_beams:
            break

    # If no finished, take best alive
    if not finished:
        finished = [(t, s) for (t, s, f) in beams]
        # Normalize scores
        finished = [
            (t, (s / (max(1, len(t) - len(prompt_ids)) ** length_penalty)) if length_penalty != 0.0 else s)
            for (t, s) in finished
        ]
    finished.sort(key=lambda x: x[1], reverse=True)
    best_tokens = finished[0][0]
    text_out = make_tokenizer_bridge(tokenizer).decode(best_tokens)
    if best_tokens and eos_ids and (best_tokens[-1] in eos_ids):
        finish_reason = "eos"
    elif n_generated >= max_new:
        finish_reason = "length"
    else:
        finish_reason = "stop_sequence"
    return {
        "text": text_out,
        "tokens": best_tokens,
        "eos_reached": True,
        "finish_reason": finish_reason,
    }
