import pytest

from velveteentrade.broker import AccountState, Position
from velveteentrade.config import RiskLimits
from velveteentrade import risk
from velveteentrade.schemas import ExecutiveDecision, TechnicalSnapshot


def snap(symbol="AAPL", close=100.0, atr=2.0, cvar=-0.10):
    return TechnicalSnapshot(symbol=symbol, close=close, atr_14=atr, hist_cvar_95_20d=cvar)


def decision(symbol="AAPL", action="BUY", conviction=4):
    return ExecutiveDecision(
        symbol=symbol, action=action, conviction=conviction,
        thesis="t", invalidation="i", horizon_days=10,
    )


def account(equity=100_000.0, positions=None):
    return AccountState(equity=equity, cash=equity, positions=positions or {})


@pytest.fixture
def limits():
    return RiskLimits()


@pytest.fixture
def ctx():
    return risk.CycleContext(peak_equity=100_000.0)


def test_buy_approved_and_sized_by_risk_budget(limits, ctx):
    v = risk.evaluate(decision(), snap(), account(), limits, ctx)
    assert v.approved
    # risk budget = 100k * 1% * (4/5) = 800; stop distance = 2 * 2.0 = 4 → 200 shares,
    # then the 10% position cap (10k / $100) trims it to 100 shares. The cap wins.
    assert v.qty == 100
    assert v.stop_price == 96.0
    assert v.qty * 100.0 <= 100_000 * limits.max_position_pct


def test_low_conviction_rejected(limits, ctx):
    v = risk.evaluate(decision(conviction=2), snap(), account(), limits, ctx)
    assert not v.approved


def test_position_cap_limits_size(limits, ctx):
    # Full-conviction cheap stock would exceed 10% cap → capped at 10% of equity.
    v = risk.evaluate(decision(conviction=5), snap(close=10.0, atr=0.05), account(), limits, ctx)
    assert v.approved
    assert v.qty * 10.0 <= 100_000 * limits.max_position_pct + 10.0


def test_max_positions_blocks_new_buys(limits, ctx):
    positions = {
        f"S{i}": Position(symbol=f"S{i}", qty=10, avg_entry=50, market_value=500)
        for i in range(limits.max_positions)
    }
    v = risk.evaluate(decision(symbol="NEW"), snap("NEW"), account(positions=positions), limits, ctx)
    assert not v.approved


def test_drawdown_halts_buys(limits):
    ctx = risk.CycleContext(peak_equity=120_000.0)  # equity 100k → 16.7% drawdown
    v = risk.evaluate(decision(), snap(), account(), limits, ctx)
    assert not v.approved
    assert any("drawdown" in r for r in v.reasons)


def test_fat_tail_cvar_rejected(limits, ctx):
    v = risk.evaluate(decision(), snap(cvar=-0.40), account(), limits, ctx)
    assert not v.approved
    assert any("CVaR" in r for r in v.reasons)


def test_sell_without_position_rejected(limits, ctx):
    v = risk.evaluate(decision(action="SELL"), snap(), account(), limits, ctx)
    assert not v.approved


def test_sell_closes_full_position(limits, ctx):
    positions = {"AAPL": Position(symbol="AAPL", qty=50, avg_entry=90, market_value=5000)}
    v = risk.evaluate(decision(action="SELL"), snap(), account(positions=positions), limits, ctx)
    assert v.approved and v.qty == 50


def test_turnover_budget_exhausts(limits, ctx):
    ctx.turnover_used = 100_000 * limits.max_daily_turnover_pct  # budget spent
    v = risk.evaluate(decision(), snap(), account(), limits, ctx)
    assert not v.approved
    assert any("turnover" in r for r in v.reasons)


def test_hold_never_orders(limits, ctx):
    v = risk.evaluate(decision(action="HOLD"), snap(), account(), limits, ctx)
    assert not v.approved and v.qty == 0


def test_stop_sweep_triggers_sell(limits):
    positions = {"AAPL": Position(symbol="AAPL", qty=50, avg_entry=100, market_value=4500)}
    acct = account(positions=positions)
    s = snap(close=95.9, atr=2.0)  # stop = 100 - 4 = 96 → close below
    verdicts = risk.check_stops(acct, {"AAPL": s}, limits, entries={"AAPL": 100.0})
    assert len(verdicts) == 1 and verdicts[0].action == "SELL" and verdicts[0].qty == 50


def test_sector_cap_blocks_concentration(limits):
    ctx = risk.CycleContext(peak_equity=100_000.0,
                            sectors={"JPM": "Financials", "V": "Financials"})
    positions = {"JPM": Position(symbol="JPM", qty=100, avg_entry=280, market_value=28_000)}
    acct = account(positions=positions)
    ctx.seed_sector_exposure(acct)
    # Financials already at 28% of equity; cap is 30% → almost no room left.
    v = risk.evaluate(decision(symbol="V", conviction=5), snap("V", close=300.0), acct, limits, ctx)
    if v.approved:
        assert v.qty * 300.0 <= 100_000 * limits.max_sector_exposure_pct - 28_000 + 300.0
    else:
        assert any("sector" in r for r in v.reasons)


def test_unknown_sectors_not_lumped_together(limits, ctx):
    # Two symbols with no sector info must NOT share a concentration bucket.
    ctx.sectors = {}
    ctx.sector_notional = {"UNKNOWN:AAA": 29_000.0}
    v = risk.evaluate(decision(symbol="BBB"), snap("BBB"), account(), limits, ctx)
    assert v.approved  # BBB has its own UNKNOWN:BBB bucket


def test_stop_sweep_no_trigger_above_stop(limits):
    positions = {"AAPL": Position(symbol="AAPL", qty=50, avg_entry=100, market_value=5000)}
    verdicts = risk.check_stops(account(positions=positions), {"AAPL": snap(close=98.0)}, limits,
                                entries={"AAPL": 100.0})
    assert verdicts == []
