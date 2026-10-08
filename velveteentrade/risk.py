"""The deterministic risk gate — the most important file in this repository.

Every ExecutiveDecision passes through here. This code, not the LLM, decides
whether an order exists and how large it is. No prompt can override it.

Formal model and proofs: `verification/lean/` (see `verification/README.md`).
The invariants proved there, stated informally:

  (G1) An approved BUY never pushes the symbol, its sector, the portfolio's
       gross exposure, the position count, or the cycle's turnover above
       their caps.
  (G2) An approved BUY risks at most equity * per_trade_risk_pct * conviction/5
       between the reference price and its (frozen) stop.
  (G3) HOLD never orders; SELL only closes a position that exists.
  (C)  Over a whole cycle — ANY sequence of decisions, i.e. ANY LLM output —
       if the portfolio starts within its caps, it ends within its caps.

(C) is why the gate keeps a *shadow portfolio* in CycleContext: the broker's
account snapshot does not change while a cycle runs, so checking caps against
it lets several individually-valid BUYs jointly break them (the bug fixed in
Oct 2026: 9 positions / 105.8% gross with caps 8 / 100%).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .broker import AccountState
from .config import RiskLimits
from .schemas import ExecutiveDecision, RiskVerdict, TechnicalSnapshot

log = logging.getLogger(__name__)

MIN_ORDER_NOTIONAL = 1.0  # Alpaca's minimum for fractional orders


def _round_qty(x: float) -> float:
    """Fractional share quantity, floored to 3 decimals (never round up past a cap)."""
    return max(0.0, math.floor(x * 1000) / 1000)


def _stop_price(close: float, stop_distance: float) -> float:
    """Stop rounded UP to the cent, so the real stop distance never exceeds the
    one used for sizing (rounding down could add up to $0.005/share of risk)."""
    return math.ceil((close - stop_distance) * 100) / 100


@dataclass
class CycleContext:
    """State threaded through one decision cycle.

    Besides the budgets (turnover, sectors), it holds a SHADOW PORTFOLIO —
    per-symbol exposure and gross exposure — seeded from the broker snapshot
    and updated by every approved order, so that caps are checked against the
    portfolio as it will be, not as it was when the cycle started.
    """

    turnover_used: float = 0.0
    peak_equity: float | None = None
    sectors: dict[str, str] = field(default_factory=dict)          # symbol -> sector
    sector_notional: dict[str, float] = field(default_factory=dict)  # sector -> $ exposure
    held_notional: dict[str, float] = field(default_factory=dict)    # symbol -> $ exposure (shadow)
    gross: float = 0.0                                                # sum of |exposure| (shadow)
    seeded: bool = False
    notes: list[str] = field(default_factory=list)

    def sector_of(self, symbol: str) -> str:
        # Unknown sectors are per-symbol buckets, never lumped into a fake sector.
        return self.sectors.get(symbol, f"UNKNOWN:{symbol}")

    def seed(self, account: AccountState) -> None:
        """Initialize the shadow portfolio and sector exposure from the broker. Idempotent."""
        if self.seeded:
            return
        for sym, pos in account.positions.items():
            mv = abs(pos.market_value)
            self.held_notional[sym] = self.held_notional.get(sym, 0.0) + mv
            self.gross += mv
            sector = self.sector_of(sym)
            self.sector_notional[sector] = self.sector_notional.get(sector, 0.0) + mv
        self.seeded = True

    # Backwards-compatible name used by older callers/tests.
    seed_sector_exposure = seed

    @property
    def n_positions(self) -> int:
        return len(self.held_notional)

    def _apply_buy(self, symbol: str, notional: float) -> None:
        sector = self.sector_of(symbol)
        self.held_notional[symbol] = self.held_notional.get(symbol, 0.0) + notional
        self.gross += notional
        self.sector_notional[sector] = self.sector_notional.get(sector, 0.0) + notional
        self.turnover_used += notional

    def _apply_sell(self, symbol: str, traded_notional: float) -> None:
        exposure = self.held_notional.pop(symbol, 0.0)
        sector = self.sector_of(symbol)
        self.gross = max(0.0, self.gross - exposure)
        self.sector_notional[sector] = max(0.0, self.sector_notional.get(sector, 0.0) - exposure)
        self.turnover_used += traded_notional


def evaluate(
    decision: ExecutiveDecision,
    snap: TechnicalSnapshot,
    account: AccountState,
    limits: RiskLimits,
    ctx: CycleContext,
) -> RiskVerdict:
    ctx.seed(account)  # no-op after the first call of the cycle
    v = RiskVerdict(symbol=decision.symbol, approved=False, action=decision.action)
    equity = account.equity
    held = account.positions.get(decision.symbol)

    # ---------------------------------------------------------------- HOLD
    if decision.action == "HOLD":
        v.reasons.append("HOLD — no order.")
        return v

    # ---------------------------------------------------------------- SELL
    if decision.action == "SELL":
        if held is None or held.qty <= 0 or decision.symbol not in ctx.held_notional:
            v.reasons.append("SELL rejected: no position held.")
            return v
        notional = abs(held.qty) * snap.close
        if _turnover_exceeded(notional, equity, limits, ctx, v):
            return v
        v.approved, v.qty = True, held.qty
        v.reasons.append(f"SELL approved: closing {held.qty} shares.")
        ctx._apply_sell(decision.symbol, notional)
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

    is_new = decision.symbol not in ctx.held_notional
    if is_new and ctx.n_positions >= limits.max_positions:
        v.reasons.append(f"BUY rejected: already at max positions ({limits.max_positions}).")
        return v

    if snap.atr_14 is None or snap.atr_14 <= 0 or snap.close <= 0:
        v.reasons.append("BUY rejected: no valid ATR/price — cannot size the position.")
        return v

    # --- Sizing: risk-parity per trade, scaled by conviction, capped by weight.
    # Fractional shares (Alpaca supports them) so small accounts can hold
    # expensive stocks; a $200 account buying 0.05 JPM is by design.
    # Each cap below can only SHRINK qty (it is re-sized only when exceeded,
    # and _round_qty floors), which is what makes the caps compose.
    stop_distance = limits.stop_atr_mult * snap.atr_14
    conviction_scale = decision.conviction / 5.0
    risk_budget = equity * limits.per_trade_risk_pct * conviction_scale
    qty = _round_qty(risk_budget / stop_distance)

    max_notional = equity * limits.max_position_pct
    current_notional = ctx.held_notional.get(decision.symbol, 0.0)
    if current_notional + qty * snap.close > max_notional:
        qty = _round_qty((max_notional - current_notional) / snap.close)

    gross_cap = equity * limits.max_gross_exposure
    if ctx.gross + qty * snap.close > gross_cap:
        qty = _round_qty((gross_cap - ctx.gross) / snap.close)

    sector = ctx.sector_of(decision.symbol)
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

    ctx._apply_buy(decision.symbol, notional)
    v.approved, v.qty = True, float(qty)
    v.stop_price = _stop_price(snap.close, stop_distance)
    v.reasons.append(
        f"BUY approved: {qty} shares (~{notional / equity:.1%} of equity, ${notional:.2f}), "
        f"stop at {v.stop_price} ({limits.stop_atr_mult}x ATR), conviction {decision.conviction}/5."
    )
    return v


def check_stops(account: AccountState, snaps: dict[str, TechnicalSnapshot], limits: RiskLimits,
                entries: dict[str, float],
                stops: dict[str, float] | None = None) -> list[RiskVerdict]:
    """Deterministic stop-loss sweep — runs without any LLM involvement.

    `stops` maps symbol -> stop price FROZEN at buy time (preferred). The stop
    must not move with today's ATR: a stop recomputed from current volatility
    drifts down exactly when volatility rises, so the loss per trade would no
    longer be bounded by the risk budget it was sized with (G2).
    Positions without a frozen stop (opened before this rule) fall back to
    `entries[symbol] - stop_atr_mult * ATR`.
    """
    stops = stops or {}
    verdicts = []
    for symbol, pos in account.positions.items():
        snap = snaps.get(symbol)
        if snap is None:
            continue
        if symbol in stops:
            stop = stops[symbol]
        else:
            if snap.atr_14 is None:
                continue
            entry = entries.get(symbol, pos.avg_entry)
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
