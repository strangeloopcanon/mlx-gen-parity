Proposal

- Goal: A small, reusable MLX decoding library that matches Hugging Face/Torch “generate()” behavior closely and supports persona
steering cleanly.
- Name (placeholder): mlx-genkit
- Value: Use it across projects to avoid re‑implementing safe sampling, logits processors, and injection hooks each time.

Scope

- Targets: Qwen, Llama-family, and other mlx_lm-compatible decoders.
- Modes:
    - Base generation parity (no steering).
    - Persona steering via two paths: residual layer injection and logit-bias (W @ v).
- Tokenizers: HF and mlx_lm; ensure consistent EOS/pad handling and id mapping.

Public API

- generate(model, tokenizer, prompt, config, hooks=None): main entry (single/batch).
- GenerationConfig: mirrors HF fields (temperature, top_p, top_k, max_tokens, repetition_penalty, no_repeat_ngram, frequency/presence
penalties, seed).
- hooks: list of steering hooks:
    - ResidualInjectionHook(layer_idx, vector, schedule) with alpha warmup/ramp.
    - LogitBiasHook(vector, alpha) as a safe fallback.
- forward_with_hidden(tokens, capture_layers): logits + hidden capture for scoring/eval.
- detect_components(model): robust embedding/layers/norm/lm_head resolver.

Key Components

- Tokenizer bridge:
    - encode/decode/eos_id that work with either HF or mlx tokenizer.
    - Option to force HF tokenizer for variants while keeping MLX model execution.
- Model adapters:
    - Prefer explicit lm_head; fallback to tied embedding with correct dtype/shape checks.
    - Uniform layer forward: (x, mask=None, cache=None) -> (x, cache).
- Sampling parity:
    - Stable softmax in fp32.
    - Top‑p, top‑k, temperature; deterministic seeding.
    - Repetition penalty (HF style), no‑repeat‑n‑gram, frequency/presence penalties.
    - Optional length penalty and bad-words list (later).
- Cache & numerics:
    - KV cache per layer; minimal copies; mask dtype aligned to model params.
    - Always convert logits to fp32 for processing; return to model dtype as needed.
- Steering:
    - Residual injection at the same residual location as HF/Torch (post‑block add), with schedule.
    - Logit-bias path (W @ v) for models lacking clean injection structure.
- Guardrails:
    - Seed separation between base and variants.
    - Optional “avoid-identical” one-resample guard.
    - Optional suspect-pattern resample (digits/LaTeX) for bulk runs.

Parity Criteria

- Cleanliness: ≤1–2% “suspect” lines on CC‑News smoke vs 0% Torch on same settings.
- Determinism: Fixed seed returns identical tokens for base on repeated runs.
- Divergence: Variants differ from base in >95% rows on small slices with sensible α.
- Distribution: KL between MLX-base and Torch-base logits small on a small validation set (sanity check).

Milestones

- MVP (1–2 days):
    - Implement GenerationConfig, logits processors (rep penalty, no‑repeat‑ngram, freq/presence), top‑k/p, seed, fp32 sampling.
    - Robust head detection and dtype hygiene.
    - Logit-bias hook parity (done in our repo; extract and harden).
    - Adapters for Qwen/Llama; tokenizer bridge.
- Layer Injection v1 (1–2 days):
    - Exact residual injection point; cache‑aware loop; alpha scheduling.
    - Stress test on Qwen‑4B; fix tokenization alignment and head parity issues.
- Validation Suite (0.5–1 day):
    - 10–20 row harness: suspect rate, variant≠base, seed determinism, optional KL probes.
    - Side‑by‑side Torch vs MLX reports.
- Packaging (0.5 day):
    - pyproject.toml, versioning, examples, docs, typed API.

Risks & Mitigations

- Model structure variance: Adapt via robust component discovery; maintain per‑model quirks in adapters.
- Numerical drift: Force fp32 for sampling; minimize dtype conversions; prefer true lm_head.
- Tokenizer mismatch: Default to HF tokenizer for variants and ensure EOS/pad parity; add tests that compare HF vs mlx tokenization on
sample prompts.

Integration Plan

- Drop-in: Replace current MLX helper calls with mlx-genkit.generate(...).
- Scripts: Add flags mapping to GenerationConfig fields; use hooks for persona vectors.
- Fallback: If a model fails layer injection, auto‑fallback to logit-bias with a warning (configurable).

Project Layout (proposed)

- mlx_genkit/
    - api.py (generate, config, hooks)
    - sampling.py (processors, samplers)
    - adapters.py (model/heads, tokenizer bridge)
    - injection.py (residual hooks)
    - utils.py (seed, masks, dtype helpers)
    - tests/ (parity harness)
    - examples/ (CC‑News mini rewrite)

Usage Sketch

- Base: generate(model, tok, prompt, GenerationConfig(...))
- Persona:
    - hooks=[ResidualInjectionHook(layer=-3, vector=v, schedule=...)] or
    - hooks=[LogitBiasHook(vector=v, alpha=1.6)]

Next Steps

- I can extract and refactor the working pieces from our repo into a mlx-genkit module, stub the adapters for Qwen/Llama, and wire the
parity harness. When you’re ready, we’ll iterate on injection parity and extend coverage.
