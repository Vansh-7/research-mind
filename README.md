# Research Mind

Research Mind is a multi-agent research workspace that turns a focused question
into a sourced report and an independent quality review. A Streamlit interface
shows each handoff in real time while a LangChain pipeline coordinates search,
source reading, synthesis, and critique.

![Research Mind agent pipeline](docs/assets/research-mind-pipeline.png)

## Why this project

Research tasks often hide the path between a prompt and the final answer.
Research Mind makes that path inspectable through four explicit stages, typed
pipeline state, operational progress events, recoverable partial output, and a
separate critic pass.

### Product highlights

- **Live execution trace** for discovery, reading, synthesis, and review.
- **Sourced report generation** using Tavily search and web extraction.
- **Independent critique** with a structured score, strengths, and gaps.
- **Failure-aware UI** that preserves completed work when a later stage fails.
- **Credential-safe startup** with disabled execution and clear setup guidance.
- **Persistent results** across Streamlit reruns, plus Markdown export.
- **Responsive custom interface** inspired by Mistral's restrained visual system.

## Architecture

```mermaid
flowchart LR
    U[Research question] --> UI[Streamlit workspace]
    UI --> P[Sequential pipeline]
    P --> D[Discover agent]
    D -->|Tavily| W[Web sources]
    W --> R[Reader agent]
    R -->|Requests + BeautifulSoup| E[Extracted evidence]
    E --> S[LCEL writer chain]
    S --> C[LCEL critic chain]
    C --> O[Report + review]
    P -. operational events .-> UI
```

| Stage | Responsibility | Output |
| --- | --- | --- |
| Discover | Find up to five current, relevant sources | Titles, URLs, and snippets |
| Read | Select and extract the strongest source | Clean source text |
| Synthesize | Combine search results and evidence | Structured Markdown report |
| Review | Challenge clarity, evidence, and structure | Score and actionable critique |

The search and reader stages use LangChain agents. The writer and critic use
LCEL chains backed by Groq. `run_research_pipeline` remains usable without a UI
and accepts an optional event callback for other clients.

## Technology

- Python 3.14
- Streamlit
- LangChain and LCEL
- Groq (`openai/gpt-oss-120b`)
- Tavily Search
- Requests and Beautiful Soup
- `uv` for dependency and lockfile management
- `unittest` and Streamlit's application test harness

## Project structure

```text
research-mind/
├── ui/
│   └── app.py              # Streamlit product interface
├── src/
│   ├── agents.py           # Agent and LCEL chain factories
│   ├── pipeline.py         # Typed orchestration and progress events
│   └── tools/
│       ├── search_api.py   # Tavily search tool
│       └── web_scraper.py  # Source extraction tool
├── tests/
│   ├── test_app.py         # UI state and rendering smoke tests
│   └── test_pipeline.py    # Pipeline contract and failure tests
├── docs/assets/            # README media
├── pyproject.toml
└── uv.lock
```

## Quick start

### 1. Install dependencies

Install [uv](https://docs.astral.sh/uv/) and use the checked-in lockfile:

```powershell
uv sync
```

### 2. Configure providers

Create `.env` in the repository root:

```dotenv
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

| Variable | Purpose | Required |
| --- | --- | --- |
| `GROQ_API_KEY` | Runs the research agents, writer, and critic | Yes |
| `TAVILY_API_KEY` | Finds current web sources | Yes |

The application can render without credentials. Research execution remains
disabled until every required key is available. Secrets are loaded from `.env`,
which is excluded from version control.

### 3. Run the application

```powershell
uv run streamlit run ui/app.py
```

Open the local address printed by Streamlit, typically
`http://localhost:8501`.

### Command-line execution

The orchestration layer also runs independently of Streamlit:

```powershell
uv run python -m src.pipeline
```

## Testing

Tests use stubbed agents and do not call Groq, Tavily, or external websites.

```powershell
uv run python -m unittest discover -s tests -v
```

The suite verifies:

- pipeline order and the returned state contract;
- running, completed, and failed event emission;
- blank-topic validation and per-stage error attribution;
- credential-aware UI behavior and persistent completed results;
- critic score parsing and Markdown export generation.

## Pipeline API

```python
from src.pipeline import PipelineEvent, run_research_pipeline


def handle_event(event: PipelineEvent) -> None:
    print(event["stage"], event["state"], event["message"])


result = run_research_pipeline(
    "How are small language models changing on-device AI?",
    on_event=handle_event,
)

print(result["report"])
print(result["feedback"])
```

Events contain operational status and completed output. They do not expose
private model reasoning or chain-of-thought.

```python
{
    "stage": "search",      # search | reader | writer | critic
    "state": "complete",   # running | complete | error
    "message": "Source discovery complete.",
    "output": "...",       # present when a stage completes
}
```

## Extending the system

- Change the model or provider in `src/agents.py::_build_llm`.
- Add tools to the agent factories in `src/agents.py`.
- Add a stage by extending the typed stage names and orchestration in
  `src/pipeline.py`, then mirror it in the UI `STAGES` configuration.
- Reuse `on_event` to stream pipeline state to another frontend, job queue, or
  observability service.

## Operational notes

- The pipeline is intentionally sequential because each stage consumes the
  preceding stage's output.
- The scraper applies an HTTP timeout, checks response status, removes common
  layout elements, and caps extracted content.
- A downstream failure emits an error event and leaves earlier stage output
  available for inspection.
- Generated research should be reviewed before consequential use; source
  availability and model output quality can vary.
