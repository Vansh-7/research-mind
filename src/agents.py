"""LangChain agent and LCEL chain factories."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .tools.search_api import web_search
from .tools.web_scraper import scrape_url

load_dotenv()


def _build_llm() -> ChatGroq:
    """Return the shared model configuration after validating credentials."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env before running research."
        )
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def build_search_agent():
    """Build the web-search agent."""
    return create_agent(model=_build_llm(), tools=[web_search])


def build_reader_agent():
    """Build the agent that selects and reads a promising source."""
    return create_agent(model=_build_llm(), tools=[scrape_url])


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
    return prompt | _build_llm() | StrOutputParser()


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
    return prompt | _build_llm() | StrOutputParser()
