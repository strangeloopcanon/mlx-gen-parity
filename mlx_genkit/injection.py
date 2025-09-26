from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .utils import try_import_mlx


@dataclass
class ResidualInjectionHook:
    layer_idx: int  # can be negative
    vector: Any = None  # mx.array with hidden_size or None if using param_key
    schedule: Optional[Callable[[int], float]] = None  # alpha(step) -> float
    gradient_target: str = "vector"  # 'vector' | 'both'
    param_key: Optional[str] = None  # fetch vector from model[param_key] if set

    def resolve_vector(self, model) -> Any:
        if self.param_key is not None:
            return model[self.param_key]
        return self.vector

    def alpha(self, step: int) -> float:
        if self.schedule is None:
            return 1.0
        return float(self.schedule(step))


@dataclass
class LogitBiasHook:
    # Implements W @ v biasing path
    vector: Any = None  # mx.array with hidden_size or None if using param_key
    alpha: float = 1.0
    param_key: Optional[str] = None
    freeze_projection: bool = True  # If True, no grads to W

    def resolve_vector(self, model) -> Any:
        if self.param_key is not None:
            return model[self.param_key]
        return self.vector


@dataclass
class SoftPromptHook:
    n_virtual: int
    init: str = "rand"  # 'embed' | 'rand'
    param_key: str = "_soft_prompt"


class _PatchedLayer:
    def __init__(self, layer, add_vec, gradient_target: str = "vector"):
        self.layer = layer
        self.add_vec = add_vec
        self.gradient_target = gradient_target
        self._orig = layer.__call__

    def __call__(self, x, *args, **kwargs):
        mx, _ = try_import_mlx()
        out = self._orig(x, *args, **kwargs)
        if self.gradient_target == "vector":
            out = mx.stop_gradient(out) + self.add_vec
        else:
            out = out + self.add_vec
        return out

    def apply(self):
        self.layer.__call__ = self.__call__  # type: ignore[attr-defined]

    def restore(self):
        self.layer.__call__ = self._orig  # type: ignore[attr-defined]


class ResidualInjector:
    def __init__(self, layers: List[Any], hidden_size: int):
        self.layers = layers
        self.hidden_size = hidden_size
        self._patched: List[_PatchedLayer] = []

    def _broadcast(self, v, batch: int, seq_len: int):
        mx, _ = try_import_mlx()
        v = v.reshape((1, 1, -1)).astype(mx.float32)
        v = mx.broadcast_to(v, (batch, seq_len, v.shape[-1]))
        return v

    def patch(self, hooks: List[ResidualInjectionHook], step: int, batch: int, seq_len: int, model=None):
        # Clear any previous patches
        self.restore()
        mx, _ = try_import_mlx()
        for h in hooks:
            idx = h.layer_idx
            if idx < 0:
                idx = len(self.layers) + idx
            if not (0 <= idx < len(self.layers)):
                continue
            a = h.alpha(step)
            if a == 0.0:
                continue
            vec = h.resolve_vector(model)
            add_vec = self._broadcast(vec * a, batch, seq_len)
            pl = _PatchedLayer(self.layers[idx], add_vec, gradient_target=h.gradient_target)
            pl.apply()
            self._patched.append(pl)

    def restore(self):
        for p in self._patched:
            p.restore()
        self._patched = []
