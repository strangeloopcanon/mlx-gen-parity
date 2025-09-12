from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .utils import try_import_mlx, stable_log_softmax, as_mx_array


# ------------------ Logits processors ------------------


def apply_repetition_penalty(tokens: List[int], logits, penalty: float, context_size: int = 20):
    if penalty is None or penalty == 0.0:
        return logits
    if not tokens:
        return logits
    mx, _ = try_import_mlx()
    toks = tokens[-context_size:]
    idx = as_mx_array(toks, dtype=mx.int32)
    selected = logits[:, idx]
    adjusted = mx.where(selected < 0, selected * penalty, selected / penalty)
    logits[:, idx] = adjusted
    return logits


def _ngram_bans(tokens: Sequence[int], n: int) -> Dict[Tuple[int, ...], List[int]]:
    bans: Dict[Tuple[int, ...], List[int]] = {}
    if n <= 0 or len(tokens) < n:
        return bans
    for i in range(len(tokens) - n + 1):
        ctx = tuple(tokens[i : i + n - 1])
        nxt = tokens[i + n - 1]
        bans.setdefault(ctx, []).append(nxt)
    return bans


def apply_no_repeat_ngram(tokens: List[int], logits, n: int):
    if n is None or n <= 0:
        return logits
    if len(tokens) < n - 1:
        return logits
    mx, _ = try_import_mlx()
    bans = _ngram_bans(tokens, n)
    ctx = tuple(tokens[-(n - 1) :]) if n > 1 else tuple()
    banned = bans.get(ctx, [])
    if banned:
        idx = as_mx_array(banned, dtype=mx.int32)
        logits[:, idx] = -mx.inf
    return logits


def apply_frequency_presence_penalties(tokens: List[int], logits, frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
    if (frequency_penalty is None or frequency_penalty == 0.0) and (
        presence_penalty is None or presence_penalty == 0.0
    ):
        return logits
    if not tokens:
        return logits
    mx, _ = try_import_mlx()
    # Count frequencies
    counts: Dict[int, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    idx = as_mx_array(list(counts.keys()), dtype=mx.int32)
    cnts = as_mx_array(list(counts.values()), dtype=logits.dtype)
    penalty = cnts * float(frequency_penalty) + (cnts > 0) * float(presence_penalty)
    logits[:, idx] = logits[:, idx] - penalty
    return logits


def apply_logit_bias(logits, bias_vector):
    # bias_vector is shape [V]
    logits[:, :] = logits[:, :] + bias_vector
    return logits


def make_logits_processor_chain(
    *,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    no_repeat_ngram_size: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    bias_vector=None,
    bad_words_ids: Optional[List[List[int]]] = None,
    min_new_tokens: Optional[int] = None,
    eos_token_ids: Optional[List[int]] = None,
    prompt_len: int = 0,
    suppress_tokens: Optional[List[int]] = None,
    begin_suppress_tokens: Optional[List[int]] = None,
    forced_decoder_map: Optional[Dict[int, int]] = None,
    forced_bos_token_id: Optional[int] = None,
) -> List[Callable[[List[int], any], any]]:
    processors = []
    if bias_vector is not None:
        processors.append(lambda tokens, logits: apply_logit_bias(logits, bias_vector))
    if repetition_penalty and repetition_penalty != 0.0:
        processors.append(
            lambda tokens, logits: apply_repetition_penalty(
                tokens, logits, repetition_penalty, repetition_context_size
            )
        )
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        processors.append(
            lambda tokens, logits: apply_no_repeat_ngram(tokens, logits, no_repeat_ngram_size)
        )
    if (frequency_penalty and frequency_penalty != 0.0) or (
        presence_penalty and presence_penalty != 0.0
    ):
        processors.append(
            lambda tokens, logits: apply_frequency_presence_penalties(
                tokens, logits, frequency_penalty, presence_penalty
            )
        )
    if bad_words_ids:
        processors.append(make_bad_words_processor(bad_words_ids))
    if (min_new_tokens is not None) and (min_new_tokens > 0):
        processors.append(
            make_min_length_processor(min_new_tokens, eos_token_ids or [], prompt_len)
        )
    if suppress_tokens:
        processors.append(make_suppress_tokens_processor(suppress_tokens))
    if begin_suppress_tokens:
        processors.append(
            make_begin_suppress_tokens_processor(begin_suppress_tokens, prompt_len)
        )
    if forced_decoder_map is not None or forced_bos_token_id is not None:
        processors.append(
            make_forced_decoder_processor(
                forced_decoder_map or {}, prompt_len, forced_bos_token_id
            )
        )
    return processors


def make_bad_words_processor(bad_words_ids: List[List[int]]):
    # Preprocess: separate single-token bans and multi-token patterns
    single_tokens = set()
    multi = []
    for seq in bad_words_ids:
        if not seq:
            continue
        if len(seq) == 1:
            single_tokens.add(seq[0])
        else:
            multi.append(seq)

    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        if single_tokens:
            idx = as_mx_array(list(single_tokens), dtype=mx.int32)
            logits[:, idx] = -mx.inf
        if multi and tokens:
            for seq in multi:
                n = len(seq)
                if n <= 1 or len(tokens) < n - 1:
                    continue
                if tokens[-(n - 1) :] == seq[:-1]:
                    tok = seq[-1]
                    logits[:, tok] = -mx.inf
        return logits

    return proc


def make_min_length_processor(min_new_tokens: int, eos_ids: List[int], prompt_len: int):
    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        new_tokens = max(len(tokens) - prompt_len, 0)
        if new_tokens < min_new_tokens and eos_ids:
            idx = as_mx_array(eos_ids, dtype=mx.int32)
            logits[:, idx] = -mx.inf
        return logits

    return proc


def make_suppress_tokens_processor(ids: List[int]):
    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        if not ids:
            return logits
        idx = as_mx_array(list(ids), dtype=mx.int32)
        logits[:, idx] = -mx.inf
        return logits

    return proc


def make_begin_suppress_tokens_processor(ids: List[int], prompt_len: int):
    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        new_tokens = max(len(tokens) - prompt_len, 0)
        if new_tokens == 0 and ids:
            idx = as_mx_array(list(ids), dtype=mx.int32)
            logits[:, idx] = -mx.inf
        return logits

    return proc


def make_forced_decoder_processor(
    forced_map: Dict[int, int], prompt_len: int, forced_bos_token_id: Optional[int]
):
    """Force next token ID at specific new-token positions.

    Positions are relative to new tokens (i.e., position 0 means first token after prompt).
    """

    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        pos = max(len(tokens) - prompt_len, 0)
        tok = None
        if pos == 0 and forced_bos_token_id is not None:
            tok = forced_bos_token_id
        elif pos in forced_map:
            tok = forced_map[pos]
        if tok is not None:
            mask = -mx.inf * mx.ones_like(logits)
            mask[..., tok] = 0.0
            logits = logits + mask
        return logits

    return proc


# ------------------ Samplers ------------------


def _categorical_sample(logprobs, temp):
    mx, _ = try_import_mlx()
    return mx.random.categorical(logprobs * (1.0 / temp))


def _apply_top_k(logprobs, top_k: int):
    mx, _ = try_import_mlx()
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        return logprobs
    kth = top_k - 1
    mask_idx = mx.argpartition(-logprobs, kth=kth, axis=-1)[..., top_k:]
    return mx.put_along_axis(logprobs, mask_idx, -mx.inf, axis=-1)


def _apply_top_p(logprobs, top_p: float):
    mx, _ = try_import_mlx()
    if not (0.0 < top_p < 1.0):
        return logprobs
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative = mx.take_along_axis(cumulative, inverse, axis=-1)
    return mx.where(cumulative > 1 - top_p, logprobs, -mx.inf)


def _apply_min_p(logprobs, min_p: float, min_tokens_to_keep: int = 1):
    mx, _ = try_import_mlx()
    if not (0.0 < min_p <= 1.0):
        return logprobs
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)
    top_logprobs = sorted_logprobs[:, 0:1]
    scaled = top_logprobs + math.log(min_p)
    to_remove = sorted_logprobs < scaled
    to_remove[..., : min_tokens_to_keep] = False
    selected = mx.where(to_remove, -float("inf"), sorted_logprobs)
    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    return mx.take_along_axis(selected, inverse, axis=-1)


def _apply_typical(logprobs, typical_p: float, min_tokens_to_keep: int = 1):
    """Typical decoding filter on normalized logprobs."""
    mx, _ = try_import_mlx()
    if not (0.0 < typical_p < 1.0):
        return logprobs
    # logprobs are normalized
    p = mx.exp(logprobs)
    ent = -(logprobs * p).sum(axis=-1, keepdims=True)
    shifted = mx.abs(-logprobs - ent)
    sorted_indices = mx.argsort(shifted, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
    last_ind = (cumulative_probs < typical_p).sum(axis=-1)
    # mask tokens where cumulative mass above threshold
    sorted_remove = cumulative_probs > typical_p
    # keep minimum tokens
    if min_tokens_to_keep > 0:
        sorted_remove = mx.concatenate(
            [mx.zeros_like(sorted_remove[..., :min_tokens_to_keep]), sorted_remove[..., min_tokens_to_keep:]], axis=-1
        )
    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    remove = mx.take_along_axis(sorted_remove, inverse, axis=-1)
    return mx.where(remove, -mx.inf, logprobs)


def _apply_epsilon(logprobs, epsilon: float, min_tokens_to_keep: int = 1):
    mx, _ = try_import_mlx()
    if not (0.0 < epsilon < 1.0):
        return logprobs
    probs = mx.exp(logprobs)
    indices_to_remove = probs < epsilon
    # ensure at least min_tokens_to_keep remain by keeping top-k by logprobs
    if min_tokens_to_keep > 0:
        top_k = min(min_tokens_to_keep, logprobs.shape[-1])
        kth = top_k - 1
        thresh = mx.topk(logprobs, k=top_k, axis=-1)[0][..., -1, None]
        # keep tokens with logprobs >= threshold
        indices_to_remove = indices_to_remove & (logprobs < thresh)
    return mx.where(indices_to_remove, -mx.inf, logprobs)


def make_sampler(
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    typical_p: float = 0.0,
    epsilon_cutoff: float = 0.0,
):
    mx, _ = try_import_mlx()
    if temperature == 0.0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    def sampler(logprobs):
        lp = logprobs
        if 0 < top_p < 1.0:
            lp = _apply_top_p(lp, top_p)
        if min_p and min_p > 0.0:
            lp = _apply_min_p(lp, min_p, min_tokens_to_keep)
        if typical_p and typical_p > 0.0 and typical_p < 1.0:
            lp = _apply_typical(lp, typical_p, min_tokens_to_keep)
        if epsilon_cutoff and epsilon_cutoff > 0.0 and epsilon_cutoff < 1.0:
            lp = _apply_epsilon(lp, epsilon_cutoff, min_tokens_to_keep)
        if top_k and top_k > 0:
            lp = _apply_top_k(lp, top_k)
        return _categorical_sample(lp, temperature)

    return sampler


# ------------------ Constraints ------------------


def make_force_words_processor(force_words_ids: List[List[int]], prompt_len: int, strict_start: bool = False):
    """Constrain next token to continue any partially matched forced phrase.

    This simplified version enforces the next token only when the end of the
    generated suffix partially matches one of the forced sequences. It does not
    enforce starting a forced phrase proactively.
    """
    fw = [list(seq) for seq in force_words_ids if seq]
    if not fw:
        return lambda tokens, logits: logits

    def proc(tokens: List[int], logits):
        mx, _ = try_import_mlx()
        gen = tokens[prompt_len:]
        if not gen:
            if strict_start:
                # at start, restrict to first tokens of all forced phrases
                firsts = list({seq[0] for seq in fw})
                if firsts:
                    mx, _ = try_import_mlx()
                    idx_all = as_mx_array(firsts, dtype=mx.int32)
                    mask_all = mx.ones_like(logits).astype(mx.bool_)
                    idx_b = idx_all.reshape((1, -1))
                    mask_all = mx.put_along_axis(mask_all, idx_b, mx.array(False, dtype=mx.bool_), axis=-1)
                    logits = mx.where(mask_all, -mx.inf, logits)
            return logits
        allow = set()
        for seq in fw:
            n = len(seq)
            kmax = min(n - 1, len(gen))
            for k in range(kmax, 0, -1):
                if gen[-k:] == seq[:k]:
                    allow.add(seq[k])
                    break
        if allow:
            idx_all = as_mx_array(list(allow), dtype=mx.int32)
            # Mask all except allowed
            mask_all = mx.ones_like(logits).astype(mx.bool_)
            idx_b = idx_all.reshape((1, -1))
            mask_all = mx.put_along_axis(mask_all, idx_b, mx.array(False, dtype=mx.bool_), axis=-1)
            logits = mx.where(mask_all, -mx.inf, logits)
        return logits

    return proc
