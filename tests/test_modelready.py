"""llm-try/bench/grind/agent refuse a model that can't answer instead of scoring every function 0."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tools.dk.platforms.gc_mwcc import llmtry


class MissingModelTest(unittest.TestCase):
    def ollama(self, names=None, error=None):
        c = llmtry.Ollama("http://127.0.0.1:1")
        def call(path, body=None):
            if error:
                raise error
            self.assertEqual(path, "/api/tags")
            return {"models": [{"name": n} for n in names]}
        return mock.patch.object(c, "_call", call), c

    def test_installed_model_passes(self):
        p, c = self.ollama(["game-coder:lora1-pruned", "qwen3:latest"])
        with p:
            self.assertIsNone(c.missing("game-coder:lora1-pruned"))
            self.assertIsNone(c.missing("qwen3"))          # ":latest" is implied

    def test_missing_model_and_dead_server_are_named(self):
        p, c = self.ollama(["qwen3:latest"])
        with p:
            self.assertIn("not installed", c.missing("game-coder:lora9"))
        p, c = self.ollama(error=OSError("refused"))
        with p:
            self.assertIn("no Ollama server", c.missing("qwen3"))

    def test_model_ready_prints_the_reason_and_a_hint(self):
        p, c = self.ollama(["qwen3:latest"])
        out = io.StringIO()
        with p, redirect_stdout(out):
            self.assertFalse(llmtry.model_ready(c, "nope"))
        self.assertIn("error: model nope is not installed", out.getvalue())
        self.assertIn("next: ", out.getvalue())

    def test_test_doubles_without_a_check_pass(self):
        self.assertTrue(llmtry.model_ready(object(), "anything"))


if __name__ == "__main__":
    unittest.main()
