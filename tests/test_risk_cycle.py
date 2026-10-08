"""Cycle-level invariants of the risk gate — the properties proved in verification/lean.

The tests in test_risk.py check one decision at a time. These check what the
Lean theorem `cycle_safe` states: over ANY sequence of decisions (= any LLM
output), a portfolio that starts within its caps ends within its caps.
"""
import random

import pytest

from velveteentrade import risk
from velveteentrade.broker import AccountState, Position
from velveteentrade.config import RiskLimits
from velveteentrade.schemas import ExecutiveDecision, TechnicalSnapshot

EPS = 1e-9  # relative float tolerance: the Lean proofs are over exact rationals


def dec(symbol, action="BUY", conviction=5):
    return ExecutiveDecision(symbol=symbol, action=action, conviction=conviction,
                             thesis="t", invalidation="i", horizon_days=10)


def snap(symbol, close=100.0, atr=1.0, cvar=-0.10):
    return TechnicalSnapshot(symbol=symbol, close=close, atr_14=atr, hist_cvar_95_20d=cvar)


def caps_hold(ctx: risk.CycleContext, equity: float, L: RiskLimits) -> list[str]:
    tol = EPS * equity
    bad = []
    if ctx.n_positions > L.max_positions:
        bad.append(f"positions {ctx.n_positions} > {L.max_positions}")
    if ctx.gross > equity * L.max_gross_exposure + tol:
        bad.append(f"gross {ctx.gross / equity:.4f} > {L.max_gross_exposure}")
    if ctx.turnover_used > equity * L.max_daily_turnover_pct + tol:
        bad.append("turnover over budget")
    for sym, x in ctx.held_notional.items():
        if x > equity * L.max_position_pct + tol:
            bad.append(f"{sym} {x / equity:.4f} > position cap")
    for sec, x in ctx.sector_notional.items():
        if x > equity * L.max_sector_exposure_pct + tol:
            bad.append(f"sector {sec} {x / equity:.4f} > cap")
    return bad


def test_regression_oct_2026_counterexample():
    """The pattern that broke the old gate: caps checked against the broker
    snapshot, which does not change during the cycle. Start: 7 positions at
    68.6% gross (caps 8 / 80%). The old gate approves BUY A and BUY B (each
    looks fine against the snapshot) → 9 positions, 88.6% gross."""
    L, eq = RiskLimits(max_gross_exposure=0.8), 100_000.0
    pos = {f"H{i}": Position(f"H{i}", 98, 100, 9_800) for i in range(7)}
    acct = AccountState(equity=eq, cash=eq, positions=pos)
    ctx = risk.CycleContext(peak_equity=eq)
    approved = [s for s in "ABCDE"
                if risk.evaluate(dec(s), snap(s), acct, L, ctx).approved]
    assert approved == ["A"]                     # one slot left; the rest vetoed
    assert caps_hold(ctx, eq, L) == []


def test_gross_cap_trims_across_the_cycle():
    L, eq = RiskLimits(max_gross_exposure=0.8, max_positions=20), 100_000.0
    pos = {f"H{i}": Position(f"H{i}", 98, 100, 9_800) for i in range(7)}
    acct = AccountState(equity=eq, cash=eq, positions=pos)
    ctx = risk.CycleContext(peak_equity=eq)
    qtys = [risk.evaluate(dec(s), snap(s), acct, L, ctx).qty for s in "AB"]
    assert qtys == [100.0, 14.0]                 # B trimmed to the 1.4k left
    assert ctx.gross <= eq * 0.8 + 1e-6


def test_sell_frees_room_within_the_same_cycle():
    L, eq = RiskLimits(max_positions=1), 100_000.0
    acct = AccountState(equity=eq, cash=eq,
                        positions={"OLD": Position("OLD", 10, 100, 1_000)})
    ctx = risk.CycleContext(peak_equity=eq)
    assert not risk.evaluate(dec("NEW"), snap("NEW"), acct, L, ctx).approved
    assert risk.evaluate(dec("OLD", "SELL"), snap("OLD"), acct, L, ctx).approved
    assert risk.evaluate(dec("NEW"), snap("NEW"), acct, L, ctx).approved
    # and the closed position cannot be sold twice
    assert not risk.evaluate(dec("OLD", "SELL"), snap("OLD"), acct, L, ctx).approved


def test_frozen_stop_does_not_drift_with_volatility():
    L = RiskLimits()
    acct = AccountState(equity=1e5, cash=1e5, positions={"X": Position("X", 100, 100, 9_000)})
    # stop frozen at 96; today ATR tripled and close fell to 89
    out = risk.check_stops(acct, {"X": snap("X", close=89.0, atr=6.0)}, L,
                           entries={"X": 100.0}, stops={"X": 96.0})
    assert len(out) == 1
    # the legacy rule (no frozen stop) would NOT have fired: 100 - 2*6 = 88 < 89
    legacy = risk.check_stops(acct, {"X": snap("X", close=89.0, atr=6.0)}, L,
                              entries={"X": 100.0})
    assert legacy == []


def test_stop_rounds_up_so_risk_never_exceeds_budget():
    L, eq = RiskLimits(max_position_pct=1.0, max_sector_exposure_pct=1.0,
                       max_daily_turnover_pct=1.0), 100_000.0
    ctx = risk.CycleContext(peak_equity=eq)
    v = risk.evaluate(dec("Y", conviction=5), snap("Y", close=100.0, atr=1.3333),
                      AccountState(equity=eq, cash=eq), L, ctx)
    assert v.approved
    budget = eq * L.per_trade_risk_pct
    assert v.qty * (100.0 - v.stop_price) <= budget + 1e-6


@pytest.mark.parametrize("seed", range(300))
def test_random_cycles_preserve_caps(seed):
    """Fuzz: random starting portfolio within caps, random decisions (BUY/SELL/
    HOLD, repeated symbols, any conviction) — caps must hold after every step,
    and every approved BUY must respect its risk budget (G2)."""
    rng = random.Random(seed)
    L = RiskLimits(
        max_position_pct=rng.choice([0.05, 0.10, 0.25]),
        max_positions=rng.randint(1, 10),
        max_gross_exposure=rng.choice([0.5, 1.0]),
        max_daily_turnover_pct=rng.choice([0.1, 0.25, 1.0]),
        max_sector_exposure_pct=rng.choice([0.15, 0.3, 1.0]),
        min_conviction=rng.randint(1, 4),
    )
    eq = rng.choice([200.0, 5_000.0, 100_000.0])
    symbols = [f"S{i}" for i in range(12)]
    sectors = {s: rng.choice(["Tech", "Fin", "Energy"]) for s in symbols[:8]}
    prices = {s: rng.choice([3.0, 47.5, 100.0, 612.3]) for s in symbols}

    # starting portfolio built by the gate itself, so it is within caps
    acct = AccountState(equity=eq, cash=eq)
    boot = risk.CycleContext(peak_equity=eq, sectors=sectors)
    for s in rng.sample(symbols, 6):
        risk.evaluate(dec(s), snap(s, prices[s], prices[s] * 0.02), acct, L, boot)
    acct.positions = {s: Position(s, x / prices[s], prices[s], x)
                      for s, x in boot.held_notional.items()}
    ctx = risk.CycleContext(peak_equity=eq, sectors=sectors)
    ctx.seed(acct)
    assert caps_hold(ctx, eq, L) == []

    for _ in range(rng.randint(1, 15)):
        s = rng.choice(symbols)
        action = rng.choice(["BUY", "BUY", "SELL", "HOLD"])
        conv = rng.randint(1, 5)
        sn = snap(s, prices[s], prices[s] * rng.choice([0.005, 0.02, 0.08]))
        v = risk.evaluate(dec(s, action, conv), sn, acct, L, ctx)
        assert caps_hold(ctx, eq, L) == [], (seed, s, action, v.reasons)
        if v.approved and action == "BUY":
            budget = eq * L.per_trade_risk_pct * conv / 5
            assert v.qty * (sn.close - v.stop_price) <= budget * (1 + EPS) + 1e-9
        if action == "HOLD":
            assert not v.approved and v.qty == 0
