"""Sequential orchestration for the Research Mind agent pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict

from .agents import (
    GROQ_MODEL,
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
)

StageName = Literal["search", "reader", "writer", "critic"]
StageState = Literal["running", "complete", "error"]


class ResearchState(TypedDict):
    search_results: str
    scraped_content: str
    report: str
    feedback: str


class PipelineEvent(TypedDict):
    stage: StageName
    state: StageState
    message: str
    output: NotRequired[str]


EventCallback = Callable[[PipelineEvent], None]


def _emit(on_event: EventCallback | None, event: PipelineEvent) -> None:
    if on_event is not None:
        on_event(event)


def _message_content(response: dict) -> str:
    """Extract the final agent message as text with a stable failure mode."""
    messages = response.get("messages", [])
    if not messages:
        raise RuntimeError("The agent returned no messages.")
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return str(content)


def _is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return (
        getattr(exc, "status_code", None) == 429
        or getattr(response, "status_code", None) == 429
    )


def _retry_after_seconds(exc: Exception) -> int:
    """Return Groq's retry delay, with a safe fallback for exhausted quotas."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    try:
        return max(1, math.ceil(float(headers.get("retry-after", 60))))
    except (TypeError, ValueError):
        return 60


def _provider_error_message(exc: Exception) -> str:
    if not _is_rate_limit_error(exc):
        return str(exc)
    wait_seconds = _retry_after_seconds(exc)
    return (
        f"Groq's {GROQ_MODEL} quota is temporarily exhausted. "
        f"Try again in about {wait_seconds} seconds. Completed stages are preserved."
    )


def run_research_pipeline(
    topic: str, *, on_event: EventCallback | None = None
) -> ResearchState:
    """Run search, reading, writing, and critique stages in order.

    ``on_event`` receives operational progress only. It never receives model
    reasoning or hidden chain-of-thought content.
    """
    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("Research topic cannot be blank.")

    state: dict[str, str] = {}
    current_stage: StageName = "search"

    try:
        _emit(on_event, {
            "stage": "search",
            "state": "running",
            "message": "Querying the web for five relevant, recent sources.",
        })
        search_result = build_search_agent().invoke({
            "messages": [("user", "Find recent, reliable and detailed "
                          f"information about: {normalized_topic}")]
        })
        state["search_results"] = _message_content(search_result)
        _emit(on_event, {
            "stage": "search",
            "state": "complete",
            "message": "Source discovery complete.",
            "output": state["search_results"],
        })

        current_stage = "reader"
        _emit(on_event, {
            "stage": "reader",
            "state": "running",
            "message": "Selecting and extracting the strongest source.",
        })
        reader_result = build_reader_agent().invoke({
            "messages": [("user", f"Based on the following search results about "
                          f"'{normalized_topic}', pick the most relevant URL and "
                          "scrape it for deeper content.\n\n"
                          f"Search Results:\n{state['search_results'][:800]}")]
        })
        state["scraped_content"] = _message_content(reader_result)
        _emit(on_event, {
            "stage": "reader",
            "state": "complete",
            "message": "Source reading complete.",
            "output": state["scraped_content"],
        })

        current_stage = "writer"
        _emit(on_event, {
            "stage": "writer",
            "state": "running",
            "message": "Synthesizing evidence into a structured report.",
        })
        research = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        state["report"] = build_writer_chain().invoke(
            {"topic": normalized_topic, "research": research}
        )
        _emit(on_event, {
            "stage": "writer",
            "state": "complete",
            "message": "Research report drafted.",
            "output": state["report"],
        })

        current_stage = "critic"
        _emit(on_event, {
            "stage": "critic",
            "state": "running",
            "message": "Reviewing evidence, structure, and clarity.",
        })
        state["feedback"] = build_critic_chain().invoke(
            {"report": state["report"]}
        )
        _emit(on_event, {
            "stage": "critic",
            "state": "complete",
            "message": "Independent critique complete.",
            "output": state["feedback"],
        })
    except Exception as exc:
        message = _provider_error_message(exc)
        _emit(on_event, {
            "stage": current_stage,
            "state": "error",
            "message": message,
        })
        if _is_rate_limit_error(exc):
            raise RuntimeError(message) from exc
        raise

    return ResearchState(**state)


if __name__ == "__main__":
    research_topic = input("Enter a research topic: ")
    completed = run_research_pipeline(research_topic)
    print(completed["report"])
