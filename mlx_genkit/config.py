from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union


@dataclass
class GenerationConfig:
    """Configuration options for text generation.

    Mirrors Hugging Face's GenerationConfig subset while tracking MLX-specific
    toggles (speculative decoding, chat templating, etc.).
    """

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
    stop_strings: Optional[List[str]] = None
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    min_new_tokens: Optional[int] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    forced_decoder_ids: Optional[List[Tuple[int, int]]] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    use_speculative: bool = False
    draft_model_id: Optional[str] = None
    num_draft_tokens: int = 3
    max_kv_size: Optional[int] = None
    num_beams: int = 1
    length_penalty: float = 0.0
    early_stopping: bool = False
    auto_chat_template: Optional[bool] = None
    system_prompt: Optional[str] = None
    assume_user_chat: bool = False
    # Backend dispatch (future-proof); None defaults to MLX backend
    backend: Optional[str] = None
    # Placeholder fields for structured adherence (populated later phases)
    grammar: Any = None
    validators: Optional[Sequence[Any]] = None
    semantic_checks: Optional[Sequence[Any]] = None
