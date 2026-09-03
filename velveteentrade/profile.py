"""Risk profile: the bridge between the Advisor conversation and the risk gate.

The LLM interviews and extracts; THIS module — deterministic code — converts
the profile into hard limits. No prompt decides how much risk a user runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from .config import RiskLimits, Settings


class RiskProfile(BaseModel):
    """Structured output of the Advisor interview."""

    name: str = Field(..., description="How the user wants to be called")
    tolerance: int = Field(..., ge=1, le=5, description="1=very conservative, 5=aggressive")
    horizon_years: float = Field(..., ge=0.25, le=50)
    experience: Literal["NONE", "BASIC", "INTERMEDIATE", "ADVANCED"]
    max_drawdown_comfort: float = Field(
        ..., ge=0.02, le=0.50,
        description="Largest peak-to-trough loss the user says they could hold through, e.g. 0.15",
    )
    income_stability: Literal["STABLE", "VARIABLE", "PRECARIOUS"]
    has_emergency_fund: bool
    has_high_interest_debt: bool = Field(
        False, description="Credit cards, quick loans or similar expensive debt outstanding",
    )
    capital: Optional[float] = Field(
        None, description="Amount they plan to invest, in account currency (USD), if shared. "
                          "The engine sizes positions against this, not the full broker equity.",
    )
    goals: str = Field("", description="Their goals in their own words, one or two sentences")
    notes: str = Field("", description="Anything else material the advisor learned")


# ---------------------------------------------------------------------------
# Deterministic mapping: profile -> risk limits + portfolio shape.
# Rows: tolerance 1..5. Chosen so that lower tolerance means smaller positions,
# fewer of them, tighter drawdown halt and a higher conviction bar.
# ---------------------------------------------------------------------------
_TOLERANCE_TABLE: dict[int, dict] = {
    1: dict(max_position_pct=0.05, max_positions=6,  per_trade_risk_pct=0.004,
            max_drawdown_halt_pct=0.05, min_conviction=4, max_sector_exposure_pct=0.20,
            max_daily_turnover_pct=0.10, etf_core_weight=0.90),
    2: dict(max_position_pct=0.06, max_positions=7,  per_trade_risk_pct=0.006,
            max_drawdown_halt_pct=0.07, min_conviction=4, max_sector_exposure_pct=0.25,
            max_daily_turnover_pct=0.15, etf_core_weight=0.75),
    3: dict(max_position_pct=0.08, max_positions=8,  per_trade_risk_pct=0.008,
            max_drawdown_halt_pct=0.10, min_conviction=3, max_sector_exposure_pct=0.30,
            max_daily_turnover_pct=0.20, etf_core_weight=0.60),
    4: dict(max_position_pct=0.10, max_positions=8,  per_trade_risk_pct=0.010,
            max_drawdown_halt_pct=0.12, min_conviction=3, max_sector_exposure_pct=0.30,
            max_daily_turnover_pct=0.25, etf_core_weight=0.40),
    5: dict(max_position_pct=0.12, max_positions=10, per_trade_risk_pct=0.012,
            max_drawdown_halt_pct=0.15, min_conviction=3, max_sector_exposure_pct=0.35,
            max_daily_turnover_pct=0.30, etf_core_weight=0.25),
}


def effective_tolerance(profile: RiskProfile) -> int:
    """Stated tolerance, capped by circumstances. Words say 5; life may say 3."""
    t = profile.tolerance
    # The drawdown they can actually stomach binds harder than the label.
    if profile.max_drawdown_comfort < 0.08:
        t = min(t, 2)
    elif profile.max_drawdown_comfort < 0.15:
        t = min(t, 3)
    if profile.horizon_years < 2:
        t = min(t, 2)
    if profile.income_stability == "PRECARIOUS":
        t = min(t, 2)
    if not profile.has_emergency_fund:
        t = min(t, 2)
    if profile.has_high_interest_debt:
        # Investing while paying 25%+ APR is negative arbitrage; if they insist
        # on investing anyway, the system runs at its most conservative.
        t = min(t, 1)
    if profile.experience == "NONE":
        t = min(t, 3)
    return max(1, t)


def limits_for(profile: RiskProfile) -> RiskLimits:
    row = _TOLERANCE_TABLE[effective_tolerance(profile)]
    kwargs = {k: v for k, v in row.items() if k != "etf_core_weight"}
    return RiskLimits(**kwargs)


def etf_core_weight(profile: RiskProfile) -> float:
    """Fraction of the portfolio held as a diversified ETF core (not traded by
    the swing engine) vs the satellite the autopilot manages."""
    return _TOLERANCE_TABLE[effective_tolerance(profile)]["etf_core_weight"]


# ------------------------------------------------------------------- storage
def save(profile: RiskProfile, path: Path) -> None:
    path.write_text(yaml.safe_dump(profile.model_dump(), allow_unicode=True, sort_keys=False))


def load(path: Path) -> RiskProfile | None:
    if not path.exists():
        return None
    return RiskProfile.model_validate(yaml.safe_load(path.read_text()))


def apply_profile(settings: Settings) -> Settings:
    """If a profile exists, its deterministic limits replace config.yaml's."""
    profile = load(settings.profile_path)
    if profile is not None:
        settings.risk = limits_for(profile)
    return settings
