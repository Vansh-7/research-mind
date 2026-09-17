"""LangChain agent and LCEL chain factories."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from .tools.search_api import web_search
from .tools.web_scraper import scrape_url

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_REQUESTS_PER_MINUTE = 30
GROQ_REQUESTS_PER_DAY = 1_000
GROQ_TOKENS_PER_MINUTE = 8_000
GROQ_TOKENS_PER_DAY = 200_000

# TPM is the tighter limit for this multi-step pipeline. Five requests per
# minute, bounded tool context, and stage-specific completion budgets leave
# headroom for prompts and low-effort reasoning under the 8K TPM ceiling. The
# limiter is shared by every model instance, coordinating Streamlit sessions
# inside one application process.
SAFE_REQUESTS_PER_MINUTE = 5
SEARCH_MAX_TOKENS = 400
READER_MAX_TOKENS = 450
REPORT_MAX_TOKENS = 800
CRITIC_MAX_TOKENS = 300
_MODEL_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=SAFE_REQUESTS_PER_MINUTE / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
# Allow the first request immediately; subsequent calls consume the shared rate.
_MODEL_RATE_LIMITER.available_tokens = 1.0


def _build_llm(*, max_tokens: int) -> ChatGroq:
    """Return the shared model configuration after validating credentials."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env before running research."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        reasoning_effort="low",
        max_tokens=max_tokens,
        # Pipeline-level retries can surface the cooldown in Streamlit and
        # resume the affected stage after Groq's exact retry interval.
        max_retries=0,
        timeout=60,
        rate_limiter=_MODEL_RATE_LIMITER,
    )


def build_search_agent():
    """Build the web-search agent."""
    return create_agent(
        model=_build_llm(max_tokens=SEARCH_MAX_TOKENS),
        tools=[web_search],
    )


def build_reader_agent():
    """Build the agent that selects and reads a promising source."""
    return create_agent(
        model=_build_llm(max_tokens=READER_MAX_TOKENS),
        tools=[scrape_url],
    )


def build_writer_chain():
    """Build the LCEL report-writing chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert research writer. Write clear, structured and "
                "insightful reports.",
            ),
            (
                "human",
                """Write a detailed research report on the topic below.
                Topic: {topic}

                Research gathered:
                {research}

                Structure the report as:
                - Introduction
                - Key Findings (minimum 3 well-explained points)
                - Conclusion
                - Sources (list all URLs found in the research)

                Be detailed, factual and professional.""",
            ),
        ]
    )
    return (
        prompt
        | _build_llm(max_tokens=REPORT_MAX_TOKENS)
        | StrOutputParser()
    )


def build_critic_chain():
    """Build the LCEL quality-review chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a sharp and constructive research critic. Be honest and "
                "specific.",
            ),
            (
                "human",
                """Review the research report below and evaluate it strictly.
                Report:
                {report}

                Respond in this exact format:

                Score: X/10

                Strengths:
                - ...
                - ...

                Areas to Improve:
                - ...
                - ...

                One line verdict:
                ...""",
            )
        ]
    )
    return (
        prompt
        | _build_llm(max_tokens=CRITIC_MAX_TOKENS)
        | StrOutputParser()
    )
