"""Pydantic schemas — the contracts between every layer of the system.

Design principle (non-negotiable): LLM agents PROPOSE via these schemas;
deterministic code computes, sizes, and executes. An LLM output never
becomes an order without passing through the risk gate in `risk.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Action = Literal["BUY", "SELL", "HOLD"]


class TechnicalSnapshot(BaseModel):
    """Deterministic indicator values computed in pandas — never by an LLM."""

    symbol: str
    close: float
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    momentum_63d: Optional[float] = Field(None, description="63-day (3-month) return")
    momentum_21d: Optional[float] = Field(None, description="21-day (1-month) return")
    realized_vol_21d: Optional[float] = Field(None, description="Annualized 21-day realized volatility")
    hist_cvar_95_20d: Optional[float] = Field(
        None, description="Historical CVaR 95% of 20-day returns (empirical expected shortfall)"
    )
    dist_from_52w_high: Optional[float] = None


class AnalystReport(BaseModel):
    """Output of the fundamental and technical analyst agents."""

    symbol: str
    stance: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    summary: str = Field(..., description="3-6 sentence thesis grounded ONLY in the provided data")
    key_risks: list[str] = Field(default_factory=list, max_length=5)
    data_quality: Literal["FULL", "PARTIAL", "POOR"] = Field(
        ..., description="How complete the input data was; POOR means the report should carry little weight"
    )
    canon_citations: list[str] = Field(
        default_factory=list, max_length=6,
        description="Tags of the provided canon excerpts actually relied on, e.g. 'tsmom#12'",
    )


class ExecutiveDecision(BaseModel):
    """The executive agent's proposal. A proposal, never an order."""

    symbol: str
    action: Action
    conviction: int = Field(..., ge=1, le=5, description="1 = weak, 5 = table-pounding")
    thesis: str = Field(..., description="2-4 sentences citing both analyst reports")
    invalidation: str = Field(
        ..., description="Concrete, checkable condition under which this thesis is wrong"
    )
    horizon_days: int = Field(..., ge=2, le=60, description="Expected holding period")
    canon_citations: list[str] = Field(
        default_factory=list, max_length=6,
        description="Tags of the provided canon excerpts actually relied on, e.g. 'kelly#3'",
    )


class RiskVerdict(BaseModel):
    """Deterministic risk-gate ruling on an ExecutiveDecision."""

    symbol: str
    approved: bool
    action: Action
    qty: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    stop_price: Optional[float] = None


class JournalEntry(BaseModel):
    """One decision cycle record — the audit trail and the raw material for reflection."""

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    fundamental_report: Optional[AnalystReport] = None
    technical_report: Optional[AnalystReport] = None
    decision: Optional[ExecutiveDecision] = None
    verdict: Optional[RiskVerdict] = None
    executed: bool = False
    note: str = ""
