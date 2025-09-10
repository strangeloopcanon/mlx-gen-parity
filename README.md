# mlx-gen-parity

Small, reusable MLX decoding and training library that brings HF/Torch `generate()` feature parity and clean persona steering to Apple Silicon. It reuses mlx-lm primitives (caches, projections, speculative) and fills the missing parity pieces.

Features
- HF-style `GenerationConfig` with processors/warpers: repetition penalty, no-repeat-ngrams, frequency/presence, bad-words, min_new_tokens, `typical_p`, `epsilon_cutoff`.
- Constraints: `force_words_ids` (strict start + continuation), `suppress_tokens`, `begin_suppress_tokens`, multiple `eos_token_ids`, forced BOS/EOS and per-position `forced_decoder_ids`.
- Modes: sampling (fast path via mlx-lm), beam (`num_beams`, `length_penalty`, `early_stopping`), speculative (mlx-lm), sliding KV (`max_kv_size`).
- Hooks: ResidualInjectionHook (sampling) and LogitBiasHook (sampling/beam); SoftPromptHook for training.
- Training (MLX): `loss_forward`, `xent_loss` (label smoothing), mixed-precision compute (bf16) with fp32 master weights.

Install
- From PyPI (recommended):
```
pip install mlx-gen-parity
```
- Dependencies (if not already installed):
```
pip install mlx mlx-lm transformers
```
- From source (editable):
```
pip install -e .
```

Models from Hugging Face
- If the repo provides MLX weights (e.g., in `mlx-community`), you can load directly: `load('mlx-community/<model>')`.
- For standard HF (PyTorch) repos, convert once using mlx-lm:
  - Python: `from mlx_gen_parity.interop import convert_hf_to_mlx; convert_hf_to_mlx('Qwen/Qwen3-0.6B', quantize=False, local_out='mlx_qwen3_0_6b')`
  - CLI: `mlx_lm.convert --hf-path Qwen/Qwen3-0.6B --mlx-path mlx_qwen3_0_6b`
  - Then load with `load('mlx_qwen3_0_6b')`.

Basic usage
```
from mlx_gen_parity import GenerationConfig, generate
from mlx_lm import load

model, tokenizer = load('mlx_qwen3_0_6b')
cfg = GenerationConfig(max_tokens=64, temperature=0.7, top_p=0.95, seed=17)
out = generate(model, tokenizer, 'Hello MLX parity', cfg)
print(out['text'])
```

Beam and constraints
```
cfg = GenerationConfig(max_tokens=64, temperature=0.0, num_beams=4, early_stopping=True, length_penalty=0.2,
                       force_words_ids=[tokenizer.encode(' cat')], min_new_tokens=8,
                       bad_words_ids=[[tokenizer.eos_token_id]], suppress_tokens=[tokenizer.eos_token_id])
out = generate(model, tokenizer, 'The', cfg)
```

Speculative decoding
```
cfg = GenerationConfig(max_tokens=64, temperature=0.7, top_p=0.95,
                       use_speculative=True, draft_model_id='mlx_qwen3_0_6b', num_draft_tokens=3)
out = generate(model, tokenizer, 'Speculative test', cfg)
```

Persona steering
```
import mlx.core as mx
from mlx_gen_parity import LogitBiasHook
H = model.args.hidden_size
model['_persona_v'] = mx.random.normal((H,)) * (1.0/(H**0.5))
cfg = GenerationConfig(max_tokens=64, temperature=0.7)
out = generate(model, tokenizer, 'Summarize MLX', cfg, hooks=[LogitBiasHook(param_key='_persona_v', alpha=1.2)])
```

Training (MLX)
```
from mlx_gen_parity import TrainingConfig, train_step, SoftPromptHook
from mlx.optimizers import AdamW
pad_id = getattr(tokenizer, 'pad_token_id', -100) or -100
opt = AdamW(learning_rate=2e-4)
batch = {'tokens': ...}  # mx.array [B, T]
cfg = TrainingConfig(dtype='bf16', loss_scale=1024.0)
loss = train_step(model, batch, opt, cfg, hooks=[SoftPromptHook(n_virtual=10, param_key='_soft_prompt')], pad_id=pad_id)
```

Parity testing
- Torch vs MLX: `python -m mlx_gen_parity.tests.parity_hf --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b --prompt 'hello'`
- Suite (8 prompts): `python -m mlx_gen_parity.tests.parity_suite --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b`

Performance bench
```
python -m mlx_gen_parity.tests.perf_bench --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b --prompt "Hello performance" --max-tokens 64
```

Releases
- Bump version across files:
  - `make bump-version PART=patch` (or `minor`/`major`)
- Create and push a git tag (vX.Y.Z):
  - `make git-release`
  - This tags and pushes the repo; PyPI packaging can be added later.

Notes
- Parity targets control‑surface equivalence: constraints, stops, finish reasons, determinism; token streams may differ across frameworks/devices.
- Sampling fast path reuses mlx-lm’s decoding loop and caches for best performance on Apple Silicon.
