from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mlx_genkit.eval import EvalSuite
from mlx_genkit.structure.result import GenerateResult


class EvalSuiteTests(unittest.TestCase):
    def test_eval_suite_runs_with_stub_generate(self):
        suite_payload = {
            "name": "unit_suite",
            "model": "stub/model",
            "config": {"max_tokens": 8, "temperature": 0.0},
            "cases": [
                {
                    "name": "case_one",
                    "prompt": "Return a JSON object with field foo",
                    "json_schema": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    },
                    "retries": 1,
                    "strict_only_json": True,
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(suite_payload, handle)
            suite_path = handle.name

        try:
            fake_model = object()
            fake_tokenizer = object()

            def _fake_generate(*args, **kwargs):
                return GenerateResult(
                    text='{"foo": "bar"}',
                    json={"foo": "bar"},
                    schema_ok=True,
                    attempts=1,
                    only_json=True,
                )

            with patch(
                "mlx_genkit.eval.auto_load",
                return_value=(fake_model, fake_tokenizer, "./local"),
            ) as load_mock, patch(
                "mlx_genkit.eval.generate",
                side_effect=_fake_generate,
            ) as gen_mock:
                suite = EvalSuite(suite_path)
                outcomes = suite.run()

            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(gen_mock.call_count, 1)
            self.assertEqual(len(outcomes), 1)
            outcome = outcomes[0]
            self.assertTrue(outcome.result.schema_ok)
            self.assertEqual(outcome.result.json, {"foo": "bar"})
            markdown = EvalSuite.render_markdown(outcomes, suite.name)
            self.assertIn("case_one", markdown)
            summary = EvalSuite.to_dict(outcomes, suite.name)
            self.assertEqual(summary["passed"], 1)
        finally:
            os.unlink(suite_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
