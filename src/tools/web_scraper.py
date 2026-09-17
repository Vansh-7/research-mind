"""Web-page extraction tool used by the reader agent."""

from __future__ import annotations

from bs4 import BeautifulSoup
from langchain.tools import tool
import requests

MAX_CONTENT_CHARS = 12_000


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "ResearchMind/0.1 (+https://github.com/)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:MAX_CONTENT_CHARS]
    except requests.RequestException as exc:
        return f"Could not scrape URL: {exc}"
