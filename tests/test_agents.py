from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src import agents


class AgentConfigurationTests(unittest.TestCase):
    @patch.object(agents, "ChatGroq")
    def test_llm_configuration_respects_groq_rate_budget(self, chat_groq):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            configured = agents._build_llm(max_tokens=777)

        self.assertIs(configured, chat_groq.return_value)
        options = chat_groq.call_args.kwargs
        self.assertEqual(options["model"], "openai/gpt-oss-120b")
        self.assertEqual(options["reasoning_effort"], "low")
        self.assertEqual(options["max_tokens"], 777)
        self.assertEqual(options["max_retries"], 0)
        self.assertIs(options["rate_limiter"], agents._MODEL_RATE_LIMITER)


if __name__ == "__main__":
    unittest.main()
