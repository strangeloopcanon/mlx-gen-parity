from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import try_import_mlx


@dataclass
class StepStats:
    step: int
    dt_ms: float
    mem_gb: float


class Profiler:
    """Simple generation profiler: times, peak memory, tokens/sec."""

    def __init__(self):
        self._t0 = None
        self._steps: List[StepStats] = []
        self._peak_mem = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def step(self, i: int):
        mx, _ = try_import_mlx()
        dt = time.perf_counter() - (self._t0 or time.perf_counter())
        mem = 0.0
        try:
            info = mx.metal.device_info()
            # Report recommended working set as proxy; MLX doesn't expose exact allocs
            mem = info.get("max_recommended_working_set_size", 0) / (1024**3)
        except Exception:
            mem = 0.0
        self._peak_mem = max(self._peak_mem, mem)
        self._steps.append(StepStats(i, dt * 1000.0, mem))

    def report(self) -> Dict[str, Any]:
        if not self._steps:
            return {"steps": [], "peak_memory_gb": self._peak_mem, "tps": 0.0}
        total_s = self._steps[-1].dt_ms / 1000.0
        tps = len(self._steps) / total_s if total_s > 0 else 0.0
        return {
            "steps": [s.__dict__ for s in self._steps],
            "peak_memory_gb": self._peak_mem,
            "tps": tps,
        }


def profile_generate(generate_fn, *args, **kwargs):
    """Wrap `generate` to time per step (sampling uses internal loop)."""
    prof = Profiler()
    prof.start()
    out = generate_fn(*args, **kwargs)
    # We can’t intercept inner steps without a callback; for now, report total.
    prof.step(i=out and len(out.get("tokens", [])) or 0)
    return out, prof.report()

