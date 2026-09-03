"""The deterministic risk gate — the most important file in this repository.

Every ExecutiveDecision passes through here. This code, not the LLM, decides
whether an order exists and how large it is. No prompt can override it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .broker import AccountState
from .config import RiskLimits
from .schemas import ExecutiveDecision, RiskVerdict, TechnicalSnapshot

log = logging.getLogger(__name__)

MIN_ORDER_NOTIONAL = 1.0  # Alpaca's minimum for fractional orders


def _round_qty(x: float) -> float:
    """Fractional share quantity, floored to 3 decimals (never round up past a cap)."""
    import math

    return max(0.0, math.floor(x * 1000) / 1000)


@dataclass
class CycleContext:
    """Mutable state tracked across one decision cycle (turnover budget, etc.)."""

    turnover_used: float = 0.0
    peak_equity: float | None = None
    sectors: dict[str, str] = field(default_factory=dict)          # symbol -> sector
    sector_notional: dict[str, float] = field(default_factory=dict)  # sector -> $ exposure
    notes: list[str] = field(default_factory=list)

    def seed_sector_exposure(self, account: AccountState) -> None:
        """Initialize sector exposure from currently held positions."""
        for sym, pos in account.positions.items():
            sector = self.sectors.get(sym, f"UNKNOWN:{sym}")
            self.sector_notional[sector] = self.sector_notional.get(sector, 0.0) + abs(pos.market_value)


def evaluate(
    decision: ExecutiveDecision,
    snap: TechnicalSnapshot,
    account: AccountState,
    limits: RiskLimits,
    ctx: CycleContext,
) -> RiskVerdict:
    v = RiskVerdict(symbol=decision.symbol, approved=False, action=decision.action)
    equity = account.equity
    held = account.positions.get(decision.symbol)

    # ---------------------------------------------------------------- HOLD
    if decision.action == "HOLD":
        v.reasons.append("HOLD — no order.")
        return v

    # ---------------------------------------------------------------- SELL
    if decision.action == "SELL":
        if held is None or held.qty <= 0:
            v.reasons.append("SELL rejected: no position held.")
            return v
        notional = abs(held.qty) * snap.close
        if _turnover_exceeded(notional, equity, limits, ctx, v):
            return v
        v.approved, v.qty = True, held.qty
        v.reasons.append(f"SELL approved: closing {held.qty} shares.")
        ctx.turnover_used += notional
        return v

    # ----------------------------------------------------------------- BUY
    if decision.conviction < limits.min_conviction:
        v.reasons.append(
            f"BUY rejected: conviction {decision.conviction} < minimum {limits.min_conviction}."
        )
        return v

    if ctx.peak_equity and equity < ctx.peak_equity * (1 - limits.max_drawdown_halt_pct):
        v.reasons.append(
            f"BUY rejected: portfolio drawdown beyond {limits.max_drawdown_halt_pct:.0%} — new buys halted."
        )
        return v

    if snap.hist_cvar_95_20d is not None and snap.hist_cvar_95_20d < limits.max_cvar_95_20d:
        v.reasons.append(
            f"BUY rejected: empirical CVaR95(20d) {snap.hist_cvar_95_20d:.1%} worse than "
            f"limit {limits.max_cvar_95_20d:.1%}."
        )
        return v

    if held is None and len(account.positions) >= limits.max_positions:
        v.reasons.append(f"BUY rejected: already at max positions ({limits.max_positions}).")
        return v

    if snap.atr_14 is None or snap.atr_14 <= 0:
        v.reasons.append("BUY rejected: no valid ATR — cannot size the position.")
        return v

    # --- Sizing: risk-parity per trade, scaled by conviction, capped by weight.
    # Fractional shares (Alpaca supports them) so small accounts can hold
    # expensive stocks; a $200 account buying 0.05 JPM is by design.
    stop_distance = limits.stop_atr_mult * snap.atr_14
    conviction_scale = decision.conviction / 5.0
    risk_budget = equity * limits.per_trade_risk_pct * conviction_scale
    qty = _round_qty(risk_budget / stop_distance)

    max_notional = equity * limits.max_position_pct
    current_notional = (held.market_value if held else 0.0)
    if current_notional + qty * snap.close > max_notional:
        qty = _round_qty((max_notional - current_notional) / snap.close)

    gross = sum(abs(p.market_value) for p in account.positions.values())
    if gross + qty * snap.close > equity * limits.max_gross_exposure:
        qty = _round_qty((equity * limits.max_gross_exposure - gross) / snap.close)

    # --- Sector concentration cap. Unknown sectors are per-symbol buckets so
    # they are never lumped together into a fake "sector".
    sector = ctx.sectors.get(decision.symbol, f"UNKNOWN:{decision.symbol}")
    sector_used = ctx.sector_notional.get(sector, 0.0)
    sector_room = equity * limits.max_sector_exposure_pct - sector_used
    if qty * snap.close > sector_room:
        qty = _round_qty(sector_room / snap.close)
        if qty <= 0:
            v.reasons.append(
                f"BUY rejected: sector '{sector}' already at "
                f"{sector_used / equity:.1%} of equity (cap {limits.max_sector_exposure_pct:.0%})."
            )
            return v
        v.reasons.append(f"Size trimmed by sector cap on '{sector}'.")

    notional = qty * snap.close
    if qty <= 0 or notional < MIN_ORDER_NOTIONAL:
        v.reasons.append("BUY rejected: sizing below minimum order size (caps or tiny capital).")
        return v

    if _turnover_exceeded(notional, equity, limits, ctx, v):
        return v

    ctx.sector_notional[sector] = sector_used + notional
    v.approved, v.qty = True, float(qty)
    v.stop_price = round(snap.close - stop_distance, 2)
    v.reasons.append(
        f"BUY approved: {qty} shares (~{notional / equity:.1%} of equity, ${notional:.2f}), "
        f"stop at {v.stop_price} ({limits.stop_atr_mult}x ATR), conviction {decision.conviction}/5."
    )
    ctx.turnover_used += notional
    return v


def check_stops(account: AccountState, snaps: dict[str, TechnicalSnapshot], limits: RiskLimits,
                entries: dict[str, float]) -> list[RiskVerdict]:
    """Deterministic stop-loss sweep — runs without any LLM involvement.

    `entries` maps symbol -> entry price recorded at buy time.
    """
    verdicts = []
    for symbol, pos in account.positions.items():
        snap = snaps.get(symbol)
        entry = entries.get(symbol, pos.avg_entry)
        if snap is None or snap.atr_14 is None:
            continue
        stop = entry - limits.stop_atr_mult * snap.atr_14
        if snap.close <= stop:
            verdicts.append(
                RiskVerdict(
                    symbol=symbol, approved=True, action="SELL", qty=pos.qty,
                    reasons=[f"STOP hit: close {snap.close} <= stop {round(stop, 2)}."],
                )
            )
    return verdicts


def _turnover_exceeded(notional: float, equity: float, limits: RiskLimits,
                       ctx: CycleContext, v: RiskVerdict) -> bool:
    if ctx.turnover_used + notional > equity * limits.max_daily_turnover_pct:
        v.reasons.append(
            f"Rejected: cycle turnover budget exhausted "
            f"({limits.max_daily_turnover_pct:.0%} of equity)."
        )
        return True
    return False
