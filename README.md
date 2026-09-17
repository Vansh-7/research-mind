# Research Mind

Research Mind is a multi-stage AI research assistant that searches the web,
reads a relevant source, writes a structured report, and runs an independent
quality review. The Streamlit interface exposes each stage as it happens and
keeps the final report ready for download.

## Pipeline

1. **Discover** — a LangChain agent searches Tavily for recent sources.
2. **Read** — a second agent selects a useful result and extracts its content.
3. **Synthesize** — an LCEL chain turns the evidence into a structured report.
4. **Review** — an LCEL critic scores the report and identifies improvements.

The callback interface in `src.pipeline.run_research_pipeline` emits operational
stage events for UI integrations. It does not expose model reasoning.

## Setup

This project targets Python 3.14 and uses `uv` for reproducible environments.

```powershell
uv sync
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

The app loads without credentials and tells you which variables are missing.
Keys are only validated when a research run starts.

## Run the interface

```powershell
uv run streamlit run app.py
```

Then open the local address printed by Streamlit, usually
`http://localhost:8501`.

## Run from the command line

```powershell
uv run python -m src.pipeline
```

## Test

The test suite uses stubbed agents and never calls paid APIs.

```powershell
uv run python -m unittest discover -s tests -v
```

## Event callback

Pass a callable through `on_event` to receive dictionaries shaped like:

```python
{
    "stage": "search",  # search | reader | writer | critic
    "state": "running",  # running | complete | error
    "message": "Querying the web for five relevant, recent sources.",
    "output": "...",  # included for completed stages
}
```

Existing callers can continue using `run_research_pipeline(topic)` without a
callback.
