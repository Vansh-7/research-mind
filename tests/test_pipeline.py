from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import pipeline


class Invokable:
    def __init__(self, response):
        self.response = response

    def invoke(self, _payload):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def agent_response(content: str) -> dict:
    return {"messages": [SimpleNamespace(content=content)]}


class PipelineTests(unittest.TestCase):
    @patch.object(pipeline, "build_critic_chain")
    @patch.object(pipeline, "build_writer_chain")
    @patch.object(pipeline, "build_reader_agent")
    @patch.object(pipeline, "build_search_agent")
    def test_pipeline_returns_existing_contract_and_ordered_events(
        self, search, reader, writer, critic
    ):
        search.return_value = Invokable(agent_response("search output"))
        reader.return_value = Invokable(agent_response("reader output"))
        writer.return_value = Invokable("# report")
        critic.return_value = Invokable("Score: 8/10\n\nStrong report.")
        events = []

        result = pipeline.run_research_pipeline("  edge AI  ", on_event=events.append)

        self.assertEqual(
            result,
            {
                "search_results": "search output",
                "scraped_content": "reader output",
                "report": "# report",
                "feedback": "Score: 8/10\n\nStrong report.",
            },
        )
        self.assertEqual(
            [(event["stage"], event["state"]) for event in events],
            [
                ("search", "running"),
                ("search", "complete"),
                ("reader", "running"),
                ("reader", "complete"),
                ("writer", "running"),
                ("writer", "complete"),
                ("critic", "running"),
                ("critic", "complete"),
            ],
        )

    def test_pipeline_emits_the_correct_error_for_every_stage(self):
        for failing_stage in ("search", "reader", "writer", "critic"):
            with self.subTest(stage=failing_stage):
                with (
                    patch.object(pipeline, "build_search_agent") as search,
                    patch.object(pipeline, "build_reader_agent") as reader,
                    patch.object(pipeline, "build_writer_chain") as writer,
                    patch.object(pipeline, "build_critic_chain") as critic,
                ):
                    search.return_value = Invokable(agent_response("search output"))
                    reader.return_value = Invokable(agent_response("reader output"))
                    writer.return_value = Invokable("# report")
                    critic.return_value = Invokable("Score: 8/10")
                    factories = {
                        "search": search,
                        "reader": reader,
                        "writer": writer,
                        "critic": critic,
                    }
                    factories[failing_stage].return_value = Invokable(
                        RuntimeError("provider unavailable")
                    )
                    events = []

                    with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                        pipeline.run_research_pipeline(
                            "edge AI", on_event=events.append
                        )

                    self.assertEqual(events[-1]["stage"], failing_stage)
                    self.assertEqual(events[-1]["state"], "error")
                    self.assertEqual(events[-1]["message"], "provider unavailable")

    def test_pipeline_rejects_blank_topics_before_building_agents(self):
        with self.assertRaisesRegex(ValueError, "cannot be blank"):
            pipeline.run_research_pipeline("   ")


if __name__ == "__main__":
    unittest.main()
