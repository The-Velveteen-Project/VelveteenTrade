"""The deterministic risk gate — the most important file in this repository.

Every ExecutiveDecision passes through here. This code, not the LLM, decides
whether an order exists and how large it is. No prompt can override it.

Formal model and proofs: `verification/lean/` (see `verification/README.md`).
The invariants proved there, stated informally:

  (G1) An approved BUY never pushes the symbol, its sector, the portfolio's
       gross exposure, the position count, or the session's turnover above
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

The same failure exists BETWEEN cycles (B2, fixed 24 Oct 2026): the snapshot
shows only FILLED positions, so orders still queued from an earlier cycle —
market closed, several /ciclo the same evening — were invisible and every new
cycle got the full budget again (12 positions / 120% gross / 120% turnover
with caps 8 / 100% / 25%). The shadow is therefore seeded with pending orders
as well, and turnover is a per-SESSION ledger, not a per-cycle counter. This is
model P3 in `verification/`: each cycle must start from the previous one's
shadow, not from the broker's photograph alone.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .broker import AccountState, OpenOrder
from .config import RiskLimits
from .schemas import ExecutiveDecision, RiskVerdict, TechnicalSnapshot

log = logging.getLogger(__name__)

MIN_ORDER_NOTIONAL = 1.0  # Alpaca's minimum for fractional orders


def _finite(*xs: float | None) -> bool:
    """True iff every value is a real number (B3: NaN makes every comparison
    False, so a NaN input silently disables caps and stops)."""
    return all(x is not None and math.isfinite(x) for x in xs)


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
    current_value: float | None = None   # satellite value V for the drawdown halt (B6); None = equity
    sectors: dict[str, str] = field(default_factory=dict)          # symbol -> sector
    sector_notional: dict[str, float] = field(default_factory=dict)  # sector -> $ exposure
    held_notional: dict[str, float] = field(default_factory=dict)    # symbol -> $ exposure (shadow)
    gross: float = 0.0                                                # sum of |exposure| (shadow)
    seeded: bool = False
    notes: list[str] = field(default_factory=list)
    unvalued: list[str] = field(default_factory=list)       # symbols we could not value
    traded: set[str] = field(default_factory=set)            # symbols with an order this cycle

    def sector_of(self, symbol: str) -> str:
        # Unknown sectors are per-symbol buckets, never lumped into a fake sector.
        return self.sectors.get(symbol, f"UNKNOWN:{symbol}")

    def _add_exposure(self, symbol: str, notional: float) -> None:
        self.held_notional[symbol] = self.held_notional.get(symbol, 0.0) + notional
        self.gross += notional
        sector = self.sector_of(symbol)
        self.sector_notional[sector] = self.sector_notional.get(sector, 0.0) + notional

    def seed(self, account: AccountState, pending: list[OpenOrder] | tuple = (),
             prices: dict[str, float] | None = None,
             booked: dict[str, float] | None = None) -> None:
        """Initialize the shadow from the broker. Idempotent.

        - Filled positions at market value (as before).
        - PENDING orders (B2): a pending BUY is exposure already committed — it
          adds `remaining qty x reference price` to the symbol, its sector and
          gross, and takes a position slot if the symbol is new. A pending SELL
          does NOT reduce exposure (it may never fill): conservative.
        - Turnover starts from what this session has already spent: `booked`
          (order id -> notional, from the journal's ledger) plus every pending
          order the ledger does not know, each counted once.
        - A pending order that cannot be valued (no finite positive price) is
          recorded in `unvalued`, and evaluate() then halts new buys:
          if we cannot price what is committed, we do not add to it.
        """
        if self.seeded:
            return
        prices = prices or {}
        booked = booked or {}
        for sym, pos in account.positions.items():
            if not _finite(pos.market_value):
                self.unvalued.append(sym)      # unknown exposure: halt buys (fail closed)
                continue
            self._add_exposure(sym, abs(pos.market_value))
        self.turnover_used += sum(abs(n) for n in booked.values())
        for o in pending:
            px = prices.get(o.symbol)
            if o.remaining_qty > 0:
                if px is None or not math.isfinite(px) or px <= 0:
                    self.unvalued.append(o.symbol)
                    continue
                value = o.remaining_qty * px
            elif o.notional:
                value = abs(o.notional)     # dollar order: conservative, ignores partial fills
            else:
                continue                    # fully filled: already in positions
            if o.side == "buy":
                self._add_exposure(o.symbol, value)
            if o.id not in booked:
                self.turnover_used += value
        self.seeded = True

    # Backwards-compatible name used by older callers/tests.
    seed_sector_exposure = seed

    @property
    def n_positions(self) -> int:
        return len(self.held_notional)

    def mark_traded(self, symbol: str) -> None:
        """Record an order sent OUTSIDE evaluate() this cycle (the stop sweep)."""
        self.traded.add(symbol)

    def _apply_buy(self, symbol: str, notional: float) -> None:
        self._add_exposure(symbol, notional)
        self.turnover_used += notional
        self.traded.add(symbol)

    def _apply_sell(self, symbol: str, traded_notional: float) -> None:
        exposure = self.held_notional.pop(symbol, 0.0)
        sector = self.sector_of(symbol)
        self.gross = max(0.0, self.gross - exposure)
        self.sector_notional[sector] = max(0.0, self.sector_notional.get(sector, 0.0) - exposure)
        self.turnover_used += traded_notional
        self.traded.add(symbol)


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

    # ------------------------------- B5/D3: one order per symbol per cycle
    # The shadow models each symbol's position as ONE quantity; a second order
    # on the same symbol in the same cycle (a SELL after the stop sweep already
    # sold it, or BUY-then-SELL) makes shadow and account diverge: a double
    # sale, or a sale that "closes" exposure the account still holds.
    if decision.symbol in ctx.traded:
        v.reasons.append(f"Rejected: {decision.symbol} already has an order this cycle "
                         "(stop sweep or an earlier decision) — one order per symbol per cycle.")
        return v

    # ------------------------------------------- B3: fail closed on NaN/inf
    # Every check below is a comparison; with NaN they are all False and the
    # gate would approve. Nothing is mutated before this point.
    if not (_finite(snap.close, equity, ctx.gross, ctx.turnover_used)
            and (snap.atr_14 is None or _finite(snap.atr_14))
            and (held is None or _finite(held.qty))):
        v.reasons.append("Rejected: non-finite input (price, ATR, equity or budget) — "
                         "cannot evaluate caps.")
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
    if ctx.unvalued:
        v.reasons.append(
            "BUY rejected: pending orders on "
            f"{sorted(set(ctx.unvalued))} could not be valued — new buys halted this cycle."
        )
        return v

    if decision.conviction < limits.min_conviction:
        v.reasons.append(
            f"BUY rejected: conviction {decision.conviction} < minimum {limits.min_conviction}."
        )
        return v

    if equity <= 0:
        v.reasons.append("BUY rejected: no capital allocated to the satellite (or it is spent).")
        return v

    current = ctx.current_value if ctx.current_value is not None else equity
    if ctx.peak_equity and current < ctx.peak_equity * (1 - limits.max_drawdown_halt_pct):
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

    # The stop is computed BEFORE the shadow is touched (B3c: it used to raise
    # after _apply_buy, leaving the shadow mutated for an order never sent).
    stop_price = _stop_price(snap.close, stop_distance)
    ctx._apply_buy(decision.symbol, notional)
    v.approved, v.qty = True, float(qty)
    v.stop_price = stop_price
    v.reasons.append(
        f"BUY approved: {qty} shares (~{notional / equity:.1%} of equity, ${notional:.2f}), "
        f"stop at {v.stop_price} ({limits.stop_atr_mult}x ATR), conviction {decision.conviction}/5."
    )
    return v


def check_stops(account: AccountState, snaps: dict[str, TechnicalSnapshot], limits: RiskLimits,
                entries: dict[str, float],
                stops: dict[str, float] | None = None,
                unchecked: list[str] | None = None) -> list[RiskVerdict]:
    """Deterministic stop-loss sweep — runs without any LLM involvement.

    `stops` maps symbol -> stop price FROZEN at buy time (preferred). The stop
    must not move with today's ATR: a stop recomputed from current volatility
    drifts down exactly when volatility rises, so the loss per trade would no
    longer be bounded by the risk budget it was sized with (G2).
    Positions without a frozen stop (opened before this rule) fall back to
    `entries[symbol] - stop_atr_mult * ATR`.

    Price (B3): today's close if finite, else the broker's own mark
    (market_value / qty). A stop is never skipped because one data source is
    blank; if NO valid price or stop exists the position is logged as
    unchecked (and listed in `unchecked`, when a list is passed).
    """
    stops = stops or {}
    verdicts = []
    for symbol, pos in account.positions.items():
        snap = snaps.get(symbol)
        price = snap.close if snap is not None and _finite(snap.close) else None
        if price is None and _finite(pos.market_value, pos.qty) and pos.qty:
            price = abs(pos.market_value / pos.qty)
        if symbol in stops and _finite(stops[symbol]):
            stop = stops[symbol]
        else:
            atr = snap.atr_14 if snap is not None else None
            entry = entries.get(symbol, pos.avg_entry)
            stop = entry - limits.stop_atr_mult * atr if _finite(atr, entry) else None
        if price is None or stop is None:
            log.error("STOP NOT CHECKED for %s: price=%r stop=%r — no valid data.",
                      symbol, price, stop)
            if unchecked is not None:
                unchecked.append(symbol)
            continue
        if price <= stop:
            verdicts.append(
                RiskVerdict(
                    symbol=symbol, approved=True, action="SELL", qty=pos.qty,
                    reasons=[f"STOP hit: price {price} <= stop {round(stop, 2)}"
                             + ("" if snap is not None and price == snap.close
                                else " (broker mark; no valid close today)") + "."],
                )
            )
    return verdicts


def _turnover_exceeded(notional: float, equity: float, limits: RiskLimits,
                       ctx: CycleContext, v: RiskVerdict) -> bool:
    if ctx.turnover_used + notional > equity * limits.max_daily_turnover_pct:
        v.reasons.append(
            f"Rejected: session turnover budget exhausted "
            f"({limits.max_daily_turnover_pct:.0%} of equity)."
        )
        return True
    return False
