from __future__ import annotations

import json

from mlx_genkit import (
    GenerationConfig,
    JsonAdherence,
    StreamCallbacks,
    StructuredSpec,
    generate,
    generate_stream,
    generate_structured,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "confidence"],
}


def structured_generate(model, tokenizer) -> None:
    cfg = GenerationConfig(max_tokens=220, temperature=0.0)
    adherence = JsonAdherence(retries=2, strict_only_json=True)
    result = generate(
        model,
        tokenizer,
        "Summarise the latest change log entry as JSON",
        cfg,
        json_schema=SCHEMA,
        adherence=adherence,
    )
    print(json.dumps(result.json, indent=2))


def structured_spec(model, tokenizer) -> None:
    spec = StructuredSpec(
        schema=SCHEMA,
        fields=["summary", "confidence"],
        examples=[
            {
                "input": "Fix bug in sorting",
                "output": {"summary": "Bug fix", "confidence": 0.8},
            }
        ],
    )
    result = generate_structured(model, tokenizer, task="Summarise the diff", spec=spec)
    print(json.dumps(result.json, indent=2))


def structured_stream(model, tokenizer) -> None:
    def on_token(tok, idx):
        piece = tokenizer.decode([tok]) if isinstance(tok, int) else str(tok)
        if piece:
            print(piece, end="", flush=True)

    def on_invalid(info):
        print("\n[invalid]", info["message"])

    callbacks = StreamCallbacks(on_token=on_token, on_invalid_path=on_invalid)
    result = generate_stream(
        model,
        tokenizer,
        "Stream a JSON object with fields summary and confidence",
        GenerationConfig(max_tokens=128, temperature=0.0),
        json_schema=SCHEMA,
        adherence=JsonAdherence(strict_only_json=True, retries=1),
        on_token=callbacks.on_token,
        on_invalid_path=callbacks.on_invalid_path,
        stop_on_invalid=False,
    )
    print("\nAttempts:", result.attempts, "schema_ok=", result.schema_ok)
    if result.json:
        print(json.dumps(result.json, indent=2))


if __name__ == "__main__":
    from mlx_lm import load

    model, tokenizer = load("mlx_qwen3_0_6b")
    structured_generate(model, tokenizer)
    structured_spec(model, tokenizer)
    structured_stream(model, tokenizer)
