"""Streamlit frontend for the Research Mind autonomous research pipeline."""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Streamlit Cloud executes subdirectory entrypoints with ``ui/`` on sys.path.
# Add the repository root so the sibling ``src`` package resolves consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineEvent, ResearchState, run_research_pipeline

load_dotenv()

STAGES: tuple[dict[str, str], ...] = (
    {
        "key": "search",
        "number": "01",
        "name": "Discover",
        "description": "Find current, credible sources",
    },
    {
        "key": "reader",
        "number": "02",
        "name": "Read",
        "description": "Extract the strongest evidence",
    },
    {
        "key": "writer",
        "number": "03",
        "name": "Synthesize",
        "description": "Build a structured report",
    },
    {
        "key": "critic",
        "number": "04",
        "name": "Review",
        "description": "Challenge quality and clarity",
    },
)

EXAMPLE_TOPICS: tuple[tuple[str, str], ...] = (
    (
        "On-device AI",
        "How are small language models changing on-device AI in 2026?",
    ),
    (
        "Fusion energy",
        "What technical milestones are shaping commercial fusion energy?",
    ),
    (
        "Agent protocols",
        "Compare the leading interoperability protocols for AI agents.",
    ),
)


CSS = r"""
<style>
/* ---------- Foundation ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --rm-bg: #0d0d0f;
    --rm-surface: #151517;
    --rm-surface-raised: #1b1b1e;
    --rm-line: #2a2a2e;
    --rm-line-strong: #3a3a3f;
    --rm-text: #f4f3ef;
    --rm-muted: #aaa8a4;
    --rm-orange: #ff5a1f;
    --rm-red: #e91808;
    --rm-amber: #ff9700;
    --rm-success: #a9e34b;
    --rm-error: #ff7669;
    --rm-radius: 3px;
}

html, body, [class*="css"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
}

html { color-scheme: dark; }
body { background: var(--rm-bg); }

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--rm-bg);
    color: var(--rm-text);
}

[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"] {
    margin-inline: auto;
    max-width: 1240px;
    padding: 0 32px 72px;
    width: 100%;
}

p, li { line-height: 1.68; }
a { color: var(--rm-text); text-underline-offset: 3px; }

/* ---------- Product frame ---------- */
.rm-topbar {
    align-items: center;
    border-bottom: 1px solid var(--rm-line);
    display: flex;
    height: 58px;
    justify-content: space-between;
}

.rm-brand {
    align-items: center;
    display: flex;
    font-size: 14px;
    font-weight: 600;
    gap: 12px;
    letter-spacing: -0.01em;
}

.rm-mark {
    display: grid;
    grid-template-columns: repeat(5, 4px);
    grid-template-rows: repeat(5, 4px);
    height: 20px;
    width: 20px;
}

.rm-mark span { background: transparent; }
.rm-mark .o { background: var(--rm-orange); }
.rm-mark .r { background: var(--rm-red); }
.rm-mark .a { background: var(--rm-amber); }

.rm-meta {
    color: var(--rm-muted);
    display: flex;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    gap: 22px;
}

.rm-hero {
    border-bottom: 1px solid var(--rm-line);
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(320px, .65fr);
    min-height: 310px;
}

.rm-hero-copy {
    align-items: flex-end;
    border-right: 1px solid var(--rm-line);
    display: flex;
    padding: 56px 48px 48px 0;
}

.rm-hero h1 {
    color: var(--rm-text);
    font-size: clamp(54px, 7.2vw, 112px);
    font-weight: 400;
    letter-spacing: -0.075em;
    line-height: .87;
    margin: 0;
}

.rm-hero-side {
    background-color: var(--rm-surface);
    background-image:
        linear-gradient(var(--rm-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--rm-line) 1px, transparent 1px);
    background-size: 56px 56px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 40px;
}

.rm-hero-side p {
    font-size: clamp(19px, 1.8vw, 26px);
    letter-spacing: -0.035em;
    line-height: 1.18;
    margin: 0;
    max-width: 360px;
}

.rm-route {
    color: var(--rm-muted);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    line-height: 1.8;
}

.rm-route b { color: var(--rm-orange); font-weight: 500; }

/* ---------- Section rhythm ---------- */
.rm-section-head {
    margin-top: 48px;
    padding: 0 0 18px;
}

.rm-section-title {
    min-width: 0;
}

.rm-section-index {
    color: var(--rm-orange);
    display: block;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    letter-spacing: .08em;
    line-height: 1.2;
    margin-bottom: 10px;
}

.rm-section-head h2 {
    color: var(--rm-text);
    font-size: clamp(26px, 2.6vw, 38px);
    font-weight: 500;
    letter-spacing: -0.045em;
    line-height: 1.05;
    margin: 0;
}

.rm-section-description {
    color: var(--rm-muted);
    font-size: 14px;
    line-height: 1.5;
    margin: 7px 0 0;
    max-width: 650px;
}

.rm-section-meta {
    color: var(--rm-muted);
    font-family: "JetBrains Mono", Consolas, monospace;
    display: none;
}

/* ---------- Research composer ---------- */
.st-key-research_input {
    background: var(--rm-surface);
    border: 1px solid var(--rm-line);
    border-top: 0;
    padding: 30px 32px 26px;
}

[data-testid="stTextArea"] label p {
    color: var(--rm-text) !important;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -.015em;
}

[data-testid="stTextArea"] textarea {
    background: var(--rm-bg) !important;
    border: 1px solid var(--rm-line-strong) !important;
    border-radius: var(--rm-radius) !important;
    box-shadow: none !important;
    color: var(--rm-text) !important;
    font-size: 17px !important;
    line-height: 1.55 !important;
    min-height: 126px !important;
    padding: 18px 20px !important;
}

[data-testid="stTextArea"] textarea::placeholder { color: #747278; }
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--rm-orange) !important;
    outline: 2px solid rgba(255, 90, 31, .22) !important;
    outline-offset: 1px;
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
    background: var(--rm-orange) !important;
    border: 1px solid var(--rm-orange) !important;
    border-radius: var(--rm-radius) !important;
    box-shadow: none !important;
    color: var(--rm-bg) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    min-height: 48px;
    padding: 0 22px !important;
    transition: background-color 140ms ease, border-color 140ms ease;
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: #ff713d !important;
    border-color: #ff713d !important;
}

[data-testid="stButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {
    outline: 3px solid rgba(255, 90, 31, .4) !important;
    outline-offset: 3px;
}

[data-testid="stButton"] button:disabled {
    background: #727176 !important;
    border-color: #727176 !important;
    color: #1b1b1e !important;
    cursor: not-allowed;
}

.rm-form-note {
    color: var(--rm-muted);
    font-size: 12px;
    line-height: 1.5;
    margin-top: 4px;
}

.rm-composer-divider {
    border-top: 1px solid var(--rm-line);
    margin: 24px 0 20px;
}

.rm-config-message {
    border-left: 2px solid var(--rm-orange);
    color: var(--rm-muted);
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
    padding: 3px 0 3px 12px;
}

.rm-config-message strong { color: var(--rm-text); font-weight: 500; }

.rm-config-message code {
    background: transparent;
    color: var(--rm-text);
    font-family: "JetBrains Mono", Consolas, monospace;
    padding: 0;
}

.rm-examples-label {
    color: var(--rm-muted);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .04em;
    margin: 24px 0 8px;
    text-transform: uppercase;
}

.st-key-example_prompts [data-testid="stButton"] button {
    background: transparent !important;
    border-color: var(--rm-line) !important;
    color: var(--rm-muted) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    justify-content: flex-start !important;
    min-height: 40px;
    padding: 0 12px !important;
}

.st-key-example_prompts [data-testid="stButton"] button:hover {
    background: var(--rm-surface-raised) !important;
    border-color: var(--rm-line-strong) !important;
    color: var(--rm-text) !important;
}

.st-key-result_actions [data-testid="stButton"] button {
    background: var(--rm-bg) !important;
    border-color: var(--rm-line-strong) !important;
    color: var(--rm-text) !important;
}

.st-key-result_actions [data-testid="stButton"] button:hover {
    background: var(--rm-surface-raised) !important;
    border-color: var(--rm-orange) !important;
}

/* ---------- Pipeline ---------- */
.rm-pipeline-shell {
    padding: 14px 0 4px;
}

.rm-pipeline-summary {
    align-items: center;
    background: var(--rm-surface);
    display: grid;
    gap: 16px;
    grid-template-columns: minmax(160px, .4fr) minmax(220px, 1fr) auto;
    margin-bottom: 10px;
    padding: 14px 18px;
}

.rm-pipeline-count {
    color: var(--rm-text);
    font-size: 14px;
    font-weight: 500;
}

.rm-progress-track {
    background: var(--rm-line);
    height: 3px;
    overflow: hidden;
}

.rm-progress-fill {
    background: var(--rm-orange);
    height: 100%;
    transition: transform 240ms ease;
    transform-origin: left;
}

.rm-pipeline-state {
    color: var(--rm-muted);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    text-align: right;
}

.rm-pipeline-grid {
    display: grid;
    gap: 8px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}

.rm-pipeline-stage {
    background: var(--rm-surface);
    display: flex;
    flex-direction: column;
    min-height: 142px;
    padding: 18px;
    position: relative;
}

.rm-pipeline-stage.is-next {
    background: var(--rm-surface-raised);
    box-shadow: inset 0 3px 0 var(--rm-orange);
}

.rm-pipeline-stage.is-running {
    background-color: var(--rm-orange);
    background-image:
        linear-gradient(rgba(0, 0, 0, .10) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 0, 0, .10) 1px, transparent 1px);
    background-size: 30px 30px;
    color: #160d09;
}

.rm-pipeline-stage.is-complete { box-shadow: inset 0 3px 0 var(--rm-success); }
.rm-pipeline-stage.is-error { box-shadow: inset 0 3px 0 var(--rm-error); }

.rm-stage-number, .rm-stage-state, .rm-stage-log {
    font-family: "JetBrains Mono", Consolas, monospace;
}

.rm-stage-top { align-items: center; display: flex; justify-content: space-between; }
.rm-stage-number { color: #77757a; font-size: 11px; }
.rm-stage-name {
    font-size: 17px;
    font-weight: 500;
    letter-spacing: -.03em;
    margin-top: 18px;
}
.rm-stage-log {
    color: var(--rm-muted);
    font-size: 11px;
    line-height: 1.6;
    margin-top: auto;
    overflow-wrap: anywhere;
    padding-top: 14px;
}
.rm-stage-state { font-size: 9px; text-align: right; }
.rm-stage-state.waiting { color: #77757a; }
.rm-pipeline-stage.is-next .rm-stage-state.waiting { color: var(--rm-orange); }
.rm-stage-state.running { color: var(--rm-orange); }
.rm-stage-state.complete { color: var(--rm-success); }
.rm-stage-state.error { color: var(--rm-error); }

.rm-pipeline-stage.is-running .rm-stage-number,
.rm-pipeline-stage.is-running .rm-stage-state,
.rm-pipeline-stage.is-running .rm-stage-log { color: #25130b; }

/* ---------- Results ---------- */
.rm-result-context {
    color: var(--rm-muted);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    margin: 16px 0 0;
}

.st-key-report_surface,
.st-key-score_surface,
.st-key-critique_surface {
    background: var(--rm-surface);
    border: 1px solid var(--rm-line);
    padding: clamp(24px, 4vw, 52px);
}

.st-key-report_surface { border-top: 0; }
.st-key-report_surface [data-testid="stMarkdownContainer"] {
    margin: 0 auto;
    max-width: 780px;
}

.st-key-report_surface h1,
.st-key-report_surface h2,
.st-key-report_surface h3 {
    color: var(--rm-text);
    font-weight: 500;
    letter-spacing: -.035em;
    line-height: 1.15;
}

.st-key-report_surface h1 { font-size: 40px; }
.st-key-report_surface h2 {
    border-top: 1px solid var(--rm-line);
    font-size: 28px;
    margin-top: 42px;
    padding-top: 32px;
}

.st-key-report_surface p,
.st-key-report_surface li { color: #d6d4cf; font-size: 16px; }

.rm-score-label {
    color: var(--rm-muted);
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
}

.rm-score-value {
    color: var(--rm-text);
    font-size: clamp(72px, 9vw, 118px);
    font-weight: 400;
    letter-spacing: -.08em;
    line-height: .95;
    margin-top: 28px;
}

.rm-score-value small { color: var(--rm-muted); font-size: 22px; letter-spacing: -.03em; }

[data-testid="stExpander"] {
    background: var(--rm-surface) !important;
    border: 1px solid var(--rm-line) !important;
    border-radius: var(--rm-radius) !important;
}

[data-testid="stExpander"] summary { min-height: 52px; }
[data-testid="stExpander"] summary:focus-visible {
    outline: 2px solid var(--rm-orange);
    outline-offset: 2px;
}

[data-testid="stAlert"] {
    background: var(--rm-surface-raised) !important;
    border: 1px solid var(--rm-line-strong) !important;
    border-radius: var(--rm-radius) !important;
    color: var(--rm-text) !important;
}

hr { border-color: var(--rm-line) !important; }

/* ---------- Responsive and motion ---------- */
@media (max-width: 760px) {
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"] { padding: 0 18px 48px; }
    .rm-meta span:first-child { display: none; }
    .rm-hero { grid-template-columns: 1fr; }
    .rm-hero-copy {
        border-bottom: 1px solid var(--rm-line);
        border-right: 0;
        min-height: 240px;
        padding: 48px 4px 34px;
    }
    .rm-hero h1 { font-size: clamp(52px, 16vw, 82px); }
    .rm-hero-side { min-height: 230px; padding: 30px 24px; }
    .rm-section-head { margin-top: 40px; }
    .st-key-research_input { padding: 24px 20px; }
    .rm-pipeline-shell { padding: 18px 0 14px; }
    .rm-pipeline-summary { grid-template-columns: 1fr auto; }
    .rm-progress-track { grid-column: 1 / 3; grid-row: 2; }
    .rm-pipeline-grid { grid-template-columns: 1fr 1fr; }
    .rm-pipeline-stage { min-height: 162px; padding: 16px; }
    .rm-stage-name { font-size: 18px; margin-top: 22px; }
}

@media (max-width: 460px) {
    .rm-pipeline-grid { grid-template-columns: 1fr; }
    .rm-pipeline-stage { min-height: 136px; }
    .rm-stage-name { margin-top: 18px; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
</style>
"""


def initialize_state() -> None:
    """Create all session-state keys used by the application."""
    defaults: dict[str, Any] = {
        "topic_input": "",
        "submitted_topic": "",
        "run_status": "idle",
        "pipeline_events": [],
        "partial_outputs": {},
        "research_result": None,
        "run_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def missing_configuration() -> list[str]:
    """Return missing credential names without exposing credential values."""
    return [
        name
        for name in ("GROQ_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(name)
    ]


def parse_critic_score(feedback: str) -> float | None:
    """Parse a validated score from the critic's expected ``X/10`` format."""
    match = re.search(r"\bScore:\s*(\d+(?:\.\d+)?)\s*/\s*10\b", feedback, re.I)
    if match is None:
        return None
    score = float(match.group(1))
    return score if 0 <= score <= 10 else None


def build_markdown_export(topic: str, result: ResearchState) -> str:
    """Build a portable Markdown record of a completed run."""
    return (
        f"# Research Mind: {topic}\n\n"
        f"{result['report'].strip()}\n\n"
        "---\n\n"
        "## Critic assessment\n\n"
        f"{result['feedback'].strip()}\n"
    )


def apply_example_topic(topic: str) -> None:
    """Populate the composer from a quick-start topic button."""
    st.session_state.topic_input = topic


def reset_research() -> None:
    """Clear run-specific state while preserving app configuration."""
    st.session_state.topic_input = ""
    st.session_state.submitted_topic = ""
    st.session_state.run_status = "idle"
    st.session_state.pipeline_events = []
    st.session_state.partial_outputs = {}
    st.session_state.research_result = None
    st.session_state.run_error = None


def _pixel_mark() -> str:
    cells = (
        "o", "", "", "", "o",
        "o", "a", "", "a", "o",
        "r", "r", "o", "r", "r",
        "r", "", "r", "", "r",
        "r", "r", "", "r", "r",
    )
    return '<span class="rm-mark" aria-hidden="true">' + "".join(
        f'<span class="{cell}"></span>' for cell in cells
    ) + "</span>"


def render_header() -> None:
    st.markdown(
        f"""
        <header class="rm-topbar">
            <div class="rm-brand">{_pixel_mark()}<span>Research Mind</span></div>
            <div class="rm-meta"><span>Autonomous research system</span><span>LCEL / 04 stages</span></div>
        </header>
        <section class="rm-hero">
            <div class="rm-hero-copy"><h1>Research,<br>synthesized.</h1></div>
            <div class="rm-hero-side">
                <p>Turn an open question into a sourced report with an independent quality review.</p>
                <div class="rm-route"><b>01</b> discover &nbsp;→&nbsp; <b>02</b> read<br>
                <b>03</b> synthesize &nbsp;→&nbsp; <b>04</b> review</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(
    title: str, meta: str, index: str, description: str
) -> None:
    st.markdown(
        '<div class="rm-section-head">'
        '<div class="rm-section-title">'
        '<div>'
        f'<span class="rm-section-index">SECTION {html.escape(index)}</span>'
        f'<h2>{html.escape(title)}</h2>'
        f'<p class="rm-section-description">{html.escape(description)}</p>'
        '</div></div>'
        f'<span class="rm-section-meta">{html.escape(meta)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _latest_event(stage_key: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in reversed(events) if event["stage"] == stage_key), None)


def _stage_html(
    stage: dict[str, str], event: dict[str, Any] | None, *, is_next: bool = False
) -> str:
    state = event["state"] if event else "waiting"
    message = event["message"] if event else stage["description"]
    state_label = {
        "waiting": "queued",
        "running": "running",
        "complete": "complete",
        "error": "failed",
    }[state]
    visible_state = "next" if state == "waiting" and is_next else (
        "" if state == "waiting" else state_label
    )
    next_class = " is-next" if is_next else ""
    return f"""
    <div class="rm-pipeline-stage is-{state}{next_class}" role="status" aria-live="polite"
         aria-label="{html.escape(stage['name'])}: {state_label}">
        <div class="rm-stage-top">
            <div class="rm-stage-number">{stage['number']}</div>
            <div class="rm-stage-state {state}">{visible_state}</div>
        </div>
        <div class="rm-stage-name">{html.escape(stage['name'])}</div>
        <div class="rm-stage-log">{html.escape(str(message))}</div>
    </div>
    """


def _pipeline_html(events: list[dict[str, Any]]) -> str:
    completed = len({event["stage"] for event in events if event["state"] == "complete"})
    has_error = any(event["state"] == "error" for event in events)
    is_running = any(event["state"] == "running" for event in events) and completed < 4
    if has_error:
        label = "attention needed"
    elif is_running:
        label = "pipeline active"
    elif completed == len(STAGES):
        label = "complete"
    else:
        label = "ready"
    stage_events = {
        stage["key"]: _latest_event(stage["key"], events) for stage in STAGES
    }
    next_stage = None
    if not is_running and not has_error:
        next_stage = next(
            (
                stage["key"]
                for stage in STAGES
                if stage_events[stage["key"]] is None
            ),
            None,
        )
    cards = "".join(
        _stage_html(
            stage,
            stage_events[stage["key"]],
            is_next=stage["key"] == next_stage and not has_error,
        )
        for stage in STAGES
    )
    scale = completed / len(STAGES)
    return f"""
    <div class="rm-pipeline-shell">
        <div class="rm-pipeline-summary">
            <div class="rm-pipeline-count">{completed} of {len(STAGES)} stages complete</div>
            <div class="rm-progress-track" aria-label="Pipeline progress: {completed} of {len(STAGES)} stages complete">
                <div class="rm-progress-fill" style="transform:scaleX({scale})"></div>
            </div>
            <div class="rm-pipeline-state">{label}</div>
        </div>
        <div class="rm-pipeline-grid">{cards}</div>
    </div>
    """


def render_pipeline(events: list[dict[str, Any]]) -> Any:
    """Render the stage grid and return one placeholder for live updates."""
    placeholder = st.empty()
    placeholder.markdown(_pipeline_html(events), unsafe_allow_html=True)
    return placeholder


def render_results(result: ResearchState, topic: str) -> None:
    render_section_heading(
        "Research report",
        "Sourced output",
        "03",
        "Read the final synthesis first, then inspect its evidence and critique.",
    )
    st.markdown(
        f'<p class="rm-result-context">Prepared for: {html.escape(topic)}</p>',
        unsafe_allow_html=True,
    )
    with st.container(key="report_surface"):
        st.markdown(result["report"])

    render_section_heading(
        "Independent review",
        "Quality pass",
        "04",
        "A separate critic scores the report and identifies concrete gaps.",
    )
    score_column, critique_column = st.columns([0.34, 0.66], gap="small")
    score = parse_critic_score(result["feedback"])
    with score_column, st.container(key="score_surface"):
        value = "—" if score is None else f"{score:g}"
        suffix = "unavailable" if score is None else "/ 10"
        st.markdown(
            '<div class="rm-score-label">CRITIC ASSESSMENT</div>'
            f'<div class="rm-score-value">{value}<small>{suffix}</small></div>',
            unsafe_allow_html=True,
        )
    with critique_column, st.container(key="critique_surface"):
        st.markdown(result["feedback"])

    with st.expander("Source discovery output"):
        st.code(result["search_results"], language=None, wrap_lines=True)
    with st.expander("Extracted source material"):
        st.code(result["scraped_content"], language=None, wrap_lines=True)

    export = build_markdown_export(topic, result)
    with st.container(key="result_actions"):
        download_column, reset_column = st.columns([0.7, 0.3], gap="small")
        with download_column:
            st.download_button(
                "Download report",
                data=export,
                file_name="research-mind-report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with reset_column:
            st.button(
                "Start new research",
                on_click=reset_research,
                use_container_width=True,
            )


def render_partial_outputs(outputs: dict[str, str]) -> None:
    if not outputs:
        return
    with st.expander("Completed work from this run"):
        for stage in STAGES:
            content = outputs.get(stage["key"])
            if content:
                st.markdown(f"**{stage['name']}**")
                st.code(content, language=None, wrap_lines=True)


def execute_research(pipeline_placeholder: Any) -> None:
    """Run the pipeline and stream operational events into the stage rail."""
    topic = st.session_state.topic_input.strip()
    st.session_state.submitted_topic = topic
    st.session_state.pipeline_events = []
    st.session_state.partial_outputs = {}
    st.session_state.research_result = None
    st.session_state.run_error = None
    st.session_state.run_status = "running"

    def on_event(event: PipelineEvent) -> None:
        serializable_event = dict(event)
        st.session_state.pipeline_events.append(serializable_event)
        if event["state"] == "complete" and event.get("output"):
            st.session_state.partial_outputs[event["stage"]] = event["output"]
        pipeline_placeholder.markdown(
            _pipeline_html(st.session_state.pipeline_events),
            unsafe_allow_html=True,
        )

    try:
        st.session_state.research_result = run_research_pipeline(
            topic, on_event=on_event
        )
        st.session_state.run_status = "complete"
        st.toast("Research report ready")
    except Exception as exc:  # The interface must recover from provider failures.
        st.session_state.run_status = "error"
        st.session_state.run_error = str(exc)


def main() -> None:
    st.set_page_config(
        page_title="Research Mind",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    initialize_state()
    render_header()

    missing_keys = missing_configuration()
    render_section_heading(
        "Start a research run",
        "One focused question",
        "01",
        "Give the agents a precise topic, timeframe, or comparison to investigate.",
    )
    with st.container(key="research_input"):
        st.text_area(
            "What should the agents investigate?",
            key="topic_input",
            placeholder="Example: How are small language models changing on-device AI in 2026?",
            disabled=st.session_state.run_status == "running",
        )
        st.markdown(
            '<p class="rm-form-note">Include the timeframe, comparison, and evidence standard when they matter.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rm-composer-divider"></div>', unsafe_allow_html=True)

        config_column, action_column = st.columns(
            [0.68, 0.32], gap="large", vertical_alignment="center"
        )
        with config_column:
            if missing_keys:
                keys = " and ".join(
                    f"<code>{html.escape(key)}</code>" for key in missing_keys
                )
                st.markdown(
                    '<p class="rm-config-message"><strong>Configuration required.</strong> '
                    f"Add {keys} to your <code>.env</code> file, then restart the app.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="rm-config-message"><strong>Ready to research.</strong> '
                    "Four agents will find, read, synthesize, and review the evidence.</p>",
                    unsafe_allow_html=True,
                )

        with action_column:
            run_clicked = st.button(
                "Run research",
                type="primary",
                use_container_width=True,
                disabled=bool(missing_keys)
                or st.session_state.run_status == "running",
            )

        st.markdown(
            '<p class="rm-examples-label">Example questions</p>',
            unsafe_allow_html=True,
        )
        with st.container(key="example_prompts"):
            example_columns = st.columns(len(EXAMPLE_TOPICS), gap="small")
            for column, (label, topic) in zip(example_columns, EXAMPLE_TOPICS):
                with column:
                    st.button(
                        label,
                        key=f"example_{label}",
                        on_click=apply_example_topic,
                        args=(topic,),
                        use_container_width=True,
                        disabled=st.session_state.run_status == "running",
                    )

    if run_clicked and not st.session_state.topic_input.strip():
        st.error("Enter a research topic before starting the pipeline.")

    render_section_heading(
        "Agent pipeline",
        "Live execution trace",
        "02",
        "Follow each handoff from source discovery through independent review.",
    )
    pipeline_placeholder = render_pipeline(st.session_state.pipeline_events)

    if run_clicked and st.session_state.topic_input.strip():
        execute_research(pipeline_placeholder)

    if st.session_state.run_error:
        st.error(st.session_state.run_error)
        render_partial_outputs(st.session_state.partial_outputs)

    result = st.session_state.research_result
    if result:
        render_results(result, st.session_state.submitted_topic)


if __name__ == "__main__":
    main()
