"""Sequential orchestration for the Research Mind agent pipeline."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict

from .agents import (
    GROQ_MODEL,
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
)
from .token_budget import truncate_tokens

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

MAX_TOPIC_TOKENS = 160
MAX_READER_SEARCH_TOKENS = 300
MAX_WRITER_SEARCH_TOKENS = 450
MAX_WRITER_EVIDENCE_TOKENS = 550
MAX_CRITIC_REPORT_TOKENS = 800
MAX_RATE_LIMIT_RETRIES = 2
_RETRY_SECONDS_PATTERN = re.compile(
    r"(?:try again|retry)\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


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


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(
        exc,
        "status_code",
        getattr(response, "status_code", None),
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    status_code = _status_code(exc)
    return (
        status_code == 429
        or (status_code == 413 and error.get("code") == "rate_limit_exceeded")
    )


def _retry_after_seconds(exc: Exception) -> int:
    """Return Groq's retry delay, with a safe fallback for exhausted quotas."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    header_value = headers.get("retry-after")
    try:
        if header_value is not None:
            return max(1, math.ceil(float(header_value)))
    except (TypeError, ValueError):
        pass

    body = getattr(exc, "body", None)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    message = error.get("message", str(exc))
    match = _RETRY_SECONDS_PATTERN.search(message)
    if match:
        return max(1, math.ceil(float(match.group(1))))
    return 60


def _invoke_with_rate_limit_retry(
    action: Callable[[], object],
    *,
    stage: StageName,
    on_event: EventCallback | None,
) -> object:
    """Run one stage and transparently honor Groq's rolling TPM cooldown."""
    for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return action()
        except Exception as exc:
            if (
                _status_code(exc) != 429
                or retry_number >= MAX_RATE_LIMIT_RETRIES
            ):
                raise

            wait_seconds = _retry_after_seconds(exc)
            _emit(on_event, {
                "stage": stage,
                "state": "running",
                "message": (
                    "Groq's token window is full. "
                    f"Resuming automatically in {wait_seconds} seconds "
                    f"(retry {retry_number + 1}/{MAX_RATE_LIMIT_RETRIES})."
                ),
            })
            time.sleep(wait_seconds)

    raise RuntimeError("Rate-limit retry loop ended unexpectedly.")


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
    normalized_topic = truncate_tokens(normalized_topic, MAX_TOPIC_TOKENS)

    state: dict[str, str] = {}
    current_stage: StageName = "search"

    try:
        _emit(on_event, {
            "stage": "search",
            "state": "running",
            "message": "Querying the web for five relevant, recent sources.",
        })
        search_agent = build_search_agent()
        search_result = _invoke_with_rate_limit_retry(
            lambda: search_agent.invoke({
                "messages": [("user", "Find recent, reliable and detailed "
                              f"information about: {normalized_topic}")]
            }),
            stage="search",
            on_event=on_event,
        )
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
        reader_agent = build_reader_agent()
        reader_result = _invoke_with_rate_limit_retry(
            lambda: reader_agent.invoke({
                "messages": [("user", f"Based on the following search results about "
                              f"'{normalized_topic}', pick the most relevant URL and "
                              "scrape it for deeper content.\n\n"
                              "Search Results:\n"
                              f"{truncate_tokens(state['search_results'], MAX_READER_SEARCH_TOKENS)}")]
            }),
            stage="reader",
            on_event=on_event,
        )
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
            "SEARCH RESULTS:\n"
            f"{truncate_tokens(state['search_results'], MAX_WRITER_SEARCH_TOKENS)}"
            "\n\nDETAILED SCRAPED CONTENT:\n"
            f"{truncate_tokens(state['scraped_content'], MAX_WRITER_EVIDENCE_TOKENS)}"
        )
        writer_chain = build_writer_chain()
        state["report"] = _invoke_with_rate_limit_retry(
            lambda: writer_chain.invoke(
                {"topic": normalized_topic, "research": research}
            ),
            stage="writer",
            on_event=on_event,
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
        critic_chain = build_critic_chain()
        state["feedback"] = _invoke_with_rate_limit_retry(
            lambda: critic_chain.invoke(
                {
                    "report": truncate_tokens(
                        state["report"], MAX_CRITIC_REPORT_TOKENS
                    )
                }
            ),
            stage="critic",
            on_event=on_event,
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
