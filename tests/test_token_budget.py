from __future__ import annotations

import unittest

from src.token_budget import count_tokens, truncate_tokens


class TokenBudgetTests(unittest.TestCase):
    def test_truncation_never_exceeds_requested_budget(self):
        text = "Evidence and analysis. " * 500

        truncated = truncate_tokens(text, 120)

        self.assertLessEqual(count_tokens(truncated), 120)
        self.assertTrue(text.startswith(truncated))

    def test_short_text_is_returned_unchanged(self):
        text = "Short source excerpt."

        self.assertEqual(truncate_tokens(text, 100), text)

    def test_negative_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            truncate_tokens("text", -1)


if __name__ == "__main__":
    unittest.main()
