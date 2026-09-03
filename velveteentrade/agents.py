"""The three agents. Each analyst sees ONLY its own data domain; the executive
sees only the two reports. All outputs are schema-validated proposals.

Each agent carries a "constitution" — knowledge distilled from the canon
(see velveteentrade/constitutions/) — as a stable, cacheable system prompt.
When the RAG corpus is built, relevant canon excerpts are additionally
injected per-symbol, and agents cite them by tag ([doc_id#chunk]).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .llm import StructuredLLM
from .schemas import AnalystReport, ExecutiveDecision, TechnicalSnapshot

_CONSTITUTIONS_DIR = Path(__file__).resolve().parent / "constitutions"


@lru_cache(maxsize=None)
def _constitution(name: str) -> str:
    path = _CONSTITUTIONS_DIR / f"{name}.md"
    return path.read_text() if path.exists() else ""


_COMMON_RULES = (
    "Rules you must follow strictly:\n"
    "- Ground every claim ONLY in the data provided. Never use outside knowledge "
    "about this company, recent events, or price levels — your training data is stale "
    "and using it contaminates the system.\n"
    "- Never invent or compute numbers. Quote the provided figures as-is.\n"
    "- If the data is thin or missing, say so and set data_quality accordingly.\n"
    "- If canon_excerpts are provided, they are your evidence base from the research "
    "literature: cite the relevant ones by their [doc_id#chunk] tag inside your summary, "
    "and list the tags you actually relied on in canon_citations. Never cite a tag "
    "that was not provided.\n"
    "- Be direct. No hedging boilerplate.\n"
)


def _system(role_intro: str, constitution_name: str) -> str:
    parts = [role_intro, _COMMON_RULES]
    constitution = _constitution(constitution_name)
    if constitution:
        parts.append("Your constitution — operate strictly within it:\n\n" + constitution)
    return "\n".join(parts)


def fundamental_system() -> str:
    return _system(
        "You are the fundamental analyst of a systematic swing-trading desk. "
        "You assess corporate health and valuation from the fundamental snapshot and "
        "recent headlines provided.\n",
        "fundamental",
    )


def technical_system() -> str:
    return _system(
        "You are the technical analyst of a systematic swing-trading desk. "
        "You interpret precomputed indicators (trend, momentum, volatility, tail risk) "
        "for a holding period of days to weeks. The indicators were computed "
        "deterministically; your job is interpretation, not calculation.\n",
        "technical",
    )


def executive_system() -> str:
    return _system(
        "You are the executive decision-maker of a systematic swing-trading desk. "
        "You receive one fundamental report and one technical report and decide "
        "BUY, SELL, or HOLD with a conviction from 1 to 5.\n"
        "Additional rules:\n"
        "- Your decision is a PROPOSAL. A deterministic risk engine sizes and can veto it. "
        "Do not reason about position size or portfolio weights.\n"
        "- Weigh reports by their data_quality. Two POOR reports cannot justify conviction above 2.\n"
        "- The invalidation condition must be concrete and checkable from price or data.\n"
        "- SELL applies to symbols currently held; for symbols not held it means 'do not want'.\n",
        "executive",
    )


def run_fundamental(llm: StructuredLLM, model: str, symbol: str, fundamentals: dict,
                    news: list[dict], canon: list[str] | None = None) -> AnalystReport:
    payload = {
        "symbol": symbol,
        "fundamentals": fundamentals or "UNAVAILABLE",
        "recent_headlines": news or "UNAVAILABLE",
        "canon_excerpts": canon or [],
    }
    return llm.complete(model, fundamental_system(), json.dumps(payload, default=str), AnalystReport)


def run_technical(llm: StructuredLLM, model: str, snap: TechnicalSnapshot,
                  canon: list[str] | None = None) -> AnalystReport:
    payload = {"snapshot": snap.model_dump(), "canon_excerpts": canon or []}
    return llm.complete(model, technical_system(), json.dumps(payload), AnalystReport)


def run_executive(
    llm: StructuredLLM,
    model: str,
    symbol: str,
    fundamental: AnalystReport,
    technical: AnalystReport,
    held_qty: float,
    past_lessons: list[str] | None = None,
    canon: list[str] | None = None,
) -> ExecutiveDecision:
    payload = {
        "symbol": symbol,
        "currently_held_qty": held_qty,
        "fundamental_report": fundamental.model_dump(),
        "technical_report": technical.model_dump(),
        "lessons_from_past_trades": past_lessons or [],
        "canon_excerpts": canon or [],
    }
    return llm.complete(model, executive_system(), json.dumps(payload), ExecutiveDecision)
