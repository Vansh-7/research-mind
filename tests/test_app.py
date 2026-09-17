from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import app


class AppHelperTests(unittest.TestCase):
    def test_score_parser_accepts_valid_integer_and_decimal_scores(self):
        self.assertEqual(app.parse_critic_score("Score: 8/10"), 8.0)
        self.assertEqual(app.parse_critic_score("score: 7.5 / 10"), 7.5)

    def test_score_parser_rejects_missing_or_out_of_range_scores(self):
        self.assertIsNone(app.parse_critic_score("Strong report."))
        self.assertIsNone(app.parse_critic_score("Score: 12/10"))

    def test_export_contains_topic_report_and_feedback(self):
        export = app.build_markdown_export(
            "Edge AI",
            {
                "search_results": "source",
                "scraped_content": "evidence",
                "report": "## Findings\nUseful finding.",
                "feedback": "Score: 9/10",
            },
        )
        self.assertIn("# Research Mind: Edge AI", export)
        self.assertIn("Useful finding.", export)
        self.assertIn("## Critic assessment", export)

    def test_missing_configuration_reports_names_only(self):
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "", "TAVILY_API_KEY": ""},
            clear=False,
        ):
            self.assertEqual(
                app.missing_configuration(), ["GROQ_API_KEY", "TAVILY_API_KEY"]
            )


class StreamlitSmokeTests(unittest.TestCase):
    @staticmethod
    def _app_path() -> Path:
        return Path(__file__).resolve().parents[1] / "ui" / "app.py"

    def test_app_renders_without_credentials(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "", "TAVILY_API_KEY": ""},
            clear=False,
        ):
            rendered = AppTest.from_file(
                self._app_path(), default_timeout=10
            ).run()

        self.assertEqual(len(rendered.exception), 0)
        self.assertEqual(rendered.text_area[0].label, "What should the agents investigate?")
        self.assertEqual(rendered.button[0].label, "Run research")
        self.assertTrue(rendered.button[0].disabled)
        self.assertTrue(
            any("Add" in item.value and ".env" in item.value for item in rendered.markdown)
        )

    def test_entrypoint_imports_src_outside_repository(self):
        command = (
            "import runpy; "
            f"runpy.run_path({str(self._app_path())!r}, "
            "run_name='streamlit_cloud_import')"
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-c", command],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_completed_result_survives_a_rerun_and_exposes_download(self):
        from streamlit.testing.v1 import AppTest

        rendered = AppTest.from_file(self._app_path(), default_timeout=10).run()
        rendered.session_state["submitted_topic"] = "Edge AI"
        rendered.session_state["run_status"] = "complete"
        rendered.session_state["research_result"] = {
            "search_results": "Title: Source\nURL: https://example.com",
            "scraped_content": "Evidence",
            "report": "# Edge AI report\n\nA useful finding.",
            "feedback": "Score: 9/10\n\nStrengths:\n- Clear evidence",
        }
        rendered.run()

        self.assertEqual(len(rendered.exception), 0)
        self.assertEqual(rendered.download_button[0].label, "Download report")
        self.assertTrue(
            any("A useful finding." in item.value for item in rendered.markdown)
        )


if __name__ == "__main__":
    unittest.main()
