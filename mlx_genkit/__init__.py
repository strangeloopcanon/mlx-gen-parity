__all__ = [
    "GenerationConfig",
    "generate",
    "ResidualInjectionHook",
    "LogitBiasHook",
    "SoftPromptHook",
    "forward_with_hidden",
    "detect_components",
    # Training
    "TrainingConfig",
    "loss_forward",
    "xent_loss",
    "apply_lora",
    "merge_lora",
    "unmerge_lora",
    "train_step",
    # Training utilities
    "sequence_logprob",
    "token_kl",
    # Model helpers
    "ema_update",
    "build_action_mask",
    "stable_softmax",
    "clone_reference",
]

from .api import (
    GenerationConfig,
    generate,
    forward_with_hidden,
    detect_components,
)
from .injection import ResidualInjectionHook, LogitBiasHook, SoftPromptHook

from .training import (
    TrainingConfig,
    loss_forward,
    xent_loss,
    apply_lora,
    merge_lora,
    unmerge_lora,
    train_step,
    sequence_logprob,
    token_kl,
)

from .utils import ema_update, build_action_mask, stable_softmax, clone_reference

__version__ = "0.3.3"
