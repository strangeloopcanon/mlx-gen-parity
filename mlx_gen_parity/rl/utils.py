from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..utils import try_import_mlx


def gae_lambda(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """Compute Generalized Advantage Estimation (GAE-λ).

    rewards, values, dones: [T, B] arrays
    Returns advantages [T, B] and targets [T, B].
    """
    mx, _ = try_import_mlx()
    T, B = rewards.shape
    adv = mx.zeros_like(rewards)
    lastgaelam = mx.zeros((B,), dtype=values.dtype)
    for t in range(T - 1, -1, -1):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = values[t + 1] if t < T - 1 else mx.zeros_like(values[t])
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values
    return adv, ret


def kl_divergence(p_logprobs, q_logprobs):
    """Compute KL(p || q) from logprobs over same support along last axis."""
    mx, _ = try_import_mlx()
    p = mx.exp(p_logprobs)
    kl = (p * (p_logprobs - q_logprobs)).sum(axis=-1)
    return kl

