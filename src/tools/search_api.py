"""Tavily search tool used by the search agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web for recent sources and return titles, URLs, and snippets."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to .env before running research."
        )

    results = TavilyClient(api_key=api_key).search(query=query, max_results=5)
    entries = []
    for result in results.get("results", []):
        entries.append(
            "\n".join(
                [
                    f"Title: {result.get('title', 'Untitled source')}",
                    f"URL: {result.get('url', '')}",
                    f"Snippet: {result.get('content', '')}",
                ]
            )
        )
    return "\n-----\n".join(entries)
