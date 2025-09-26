from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..utils import try_import_mlx


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    lr: float = 2e-4
    grad_clip: float = 1.0


def ppo_loss(
    *,
    logp_new,  # [N]
    logp_old,  # [N]
    advantage,  # [N]
    value_pred,  # [N]
    value_target,  # [N]
    cfg: PPOConfig,
):
    mx, _ = try_import_mlx()
    ratio = mx.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = mx.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
    policy_loss = -mx.minimum(unclipped, clipped).mean()
    value_loss = ((value_pred - value_target) ** 2).mean()
    entropy = - (mx.exp(logp_new) * logp_new).mean()
    total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    return total, {"policy": policy_loss, "value": value_loss, "entropy": entropy}

