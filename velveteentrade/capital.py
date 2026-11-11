"""Capital base of the satellite the engine manages (B6, B7, B8 — 24 Oct 2026).

One number used to do three jobs — `min(account equity, declared capital)` —
and each job needed a different one:

  S  ALLOCATED satellite: (1 - etf_core_weight) x declared capital, or
     x account equity when no amount was declared. The ETF core is the
     user's; the swing engine never trades it (B8: the plan promised a 60/40
     split while the gate sized against 100% of the capital).
  V  VALUE of the satellite today: S + (account equity - equity when the
     base was set). Every gain and every LOSS of the account is the
     satellite's (the engine already treats every position as its own).
     The public track record and the drawdown halt read V (B6b: they used
     to read the declared capital, so losses were invisible).
  E  SIZING base: max(0, min(V, S, account equity)). Never more than the
     user allocated, less after losses, never more than the account has.
     A declared capital of 0 means "nothing to invest yet", not "the whole
     account" (B6a).

The base (S and the account equity it started from) is re-set whenever the
profile's capital or core weight changes, and the drawdown peak restarts in
the new base (B6c: a peak from the full account compared against a $200
capital halted buys forever).

Known limit: deposits and withdrawals in the broker account are counted as
P&L of the satellite. On a paper account there are none; for real money the
broker's cash-transfer activity should be subtracted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from .memory import Journal
from .profile import RiskProfile, etf_core_weight

_KEY = "capital_base"


@dataclass(frozen=True)
class CapitalBase:
    allocated: float      # S
    value: float          # V
    sizing: float         # E
    core_weight: float
    rebased: bool         # True if this call set a new base (peak restarts)


def _fingerprint(profile: RiskProfile | None) -> dict:
    if profile is None:
        return {"capital": None, "core": 0.0}
    cap = None if profile.capital is None else float(profile.capital)
    return {"capital": cap, "core": float(etf_core_weight(profile))}


def current_base(journal: Journal) -> dict | None:
    """The base in force: {'allocated', 'start_equity', 'since', 'fp'}, or None."""
    stored = journal.get_str(_KEY)
    return json.loads(stored) if stored else None


def pnl_since_base(journal: Journal) -> tuple[float, float, str | None] | None:
    """(start value, value now, since) of the equity series IN THE CURRENT BASE.

    The series records the satellite's value; when the profile changes the base
    the series jumps, and a P&L measured across that jump would mix two bases."""
    hist = journal.equity_history()
    if not hist:
        return None
    base = current_base(journal)
    since = base.get("since") if base else None
    seg = [e for ts, e in hist if since is None or ts >= since] or [hist[-1][1]]
    return seg[0], seg[-1], since


def resolve(journal: Journal, profile: RiskProfile | None, account_equity: float) -> CapitalBase:
    fp = _fingerprint(profile)
    stored = journal.get_str(_KEY)
    base = json.loads(stored) if stored else None
    rebased = base is None or base.get("fp") != fp
    if rebased:
        start = float(account_equity)
        gross = start if fp["capital"] is None else min(max(fp["capital"], 0.0), start)
        allocated = (1.0 - fp["core"]) * gross
        base = {"fp": fp, "start_equity": start, "allocated": allocated,
                "since": datetime.now(timezone.utc).isoformat()}
        journal.set_str(_KEY, json.dumps(base))
        legacy = stored is None and fp == {"capital": None, "core": 0.0}
        if not legacy:
            # New base, new peak. (The one exception: first run of this code
            # with no profile, where the old peak is in the same base.)
            journal.set_float("peak_equity", allocated)
    allocated = float(base["allocated"])
    value = allocated + (float(account_equity) - float(base["start_equity"]))
    sizing = max(0.0, min(value, allocated, float(account_equity)))
    return CapitalBase(allocated=allocated, value=value, sizing=sizing,
                       core_weight=fp["core"], rebased=rebased)
