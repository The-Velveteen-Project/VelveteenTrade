"""End-to-end dry run with mock broker, mock LLM, and synthetic data — no network."""
import numpy as np
import pandas as pd

from velveteentrade.broker import MockBroker
from velveteentrade.config import Settings
from velveteentrade.llm import MockLLM
from velveteentrade.memory import Journal
from velveteentrade.pipeline import run_cycle
from velveteentrade.schemas import AnalystReport, ExecutiveDecision


class FakeData:
    """Synthetic uptrending market data; no network, no cache."""

    def daily_bars(self, symbol, days=320):
        rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
        rets = rng.normal(0.001, 0.01, days)
        close = 100 * np.exp(np.cumsum(rets))
        return pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.full(days, 1e6),
        }, index=pd.date_range("2024-01-01", periods=days, freq="B"))

    def fundamentals(self, symbol):
        return {"sector": "Technology", "trailingPE": 25.0}

    def news(self, symbol):
        return [{"headline": "Quarterly results in line", "created_at": "2026-08-28", "summary": ""}]


def test_full_cycle_dry(tmp_path):
    settings = Settings(universe=["AAA", "BBB", "CCC"], candidates_per_cycle=2,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path)
    broker = MockBroker(equity=100_000)
    llm = MockLLM(responses={
        "AnalystReport": AnalystReport(symbol="AAA", stance="BULLISH",
                                       summary="Strong.", data_quality="FULL"),
        "ExecutiveDecision": ExecutiveDecision(symbol="AAA", action="BUY", conviction=4,
                                               thesis="Aligned.", invalidation="Breaks 50d SMA.",
                                               horizon_days=15),
    })
    journal = Journal(settings.db_path)

    summary = run_cycle(settings, broker, FakeData(), llm, journal)

    assert len(summary["analyzed"]) == 2          # screened candidates only
    assert len(broker.orders) >= 1                # BUY approved and "executed"
    assert all(side == "buy" for _, _, side in broker.orders)
    assert journal.recent()                       # audit trail written
    assert journal.entry_prices()                 # entry recorded for stop sweeps


def test_cycle_rejects_low_conviction(tmp_path):
    settings = Settings(universe=["AAA"], candidates_per_cycle=1,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path)
    broker = MockBroker()
    llm = MockLLM(responses={
        "AnalystReport": AnalystReport(symbol="AAA", stance="NEUTRAL",
                                       summary="Meh.", data_quality="PARTIAL"),
        "ExecutiveDecision": ExecutiveDecision(symbol="AAA", action="BUY", conviction=2,
                                               thesis="Weak.", invalidation="n/a", horizon_days=5),
    })
    summary = run_cycle(settings, broker, FakeData(), llm, Journal(settings.db_path))
    assert broker.orders == []
    assert summary["rejected"]


def test_pending_orders_block_reentry(tmp_path):
    """A queued unfilled order must make its symbol untouchable this cycle."""
    from velveteentrade.broker import OpenOrder

    settings = Settings(universe=["AAA", "BBB"], candidates_per_cycle=2,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path)
    broker = MockBroker(equity=100_000)
    broker.pending = [OpenOrder(id="o1", symbol="AAA", qty=27, side="buy")]
    llm = MockLLM(responses={
        "AnalystReport": AnalystReport(symbol="AAA", stance="BULLISH",
                                       summary="Strong.", data_quality="FULL"),
        "ExecutiveDecision": ExecutiveDecision(symbol="AAA", action="BUY", conviction=5,
                                               thesis="Aligned.", invalidation="Breaks 50d SMA.",
                                               horizon_days=15),
    })
    summary = run_cycle(settings, broker, FakeData(), llm, Journal(settings.db_path))
    assert "AAA" not in summary["analyzed"]           # skipped entirely
    assert summary["pending_skipped"] == ["AAA"]
    assert all(sym != "AAA" for sym, _, _ in broker.orders)  # no duplicate order


def test_capital_base_from_profile_shrinks_sizing(tmp_path):
    """A $100k demo account must trade like a $200 one when the profile says so."""
    from velveteentrade import profile as pm
    from velveteentrade.profile import RiskProfile

    settings = Settings(universe=["AAA"], candidates_per_cycle=1,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path)
    settings.profile_path = tmp_path / "profile.yaml"
    pm.save(RiskProfile(name="Carlos", tolerance=4, horizon_years=5, experience="BASIC",
                        max_drawdown_comfort=0.20, income_stability="STABLE",
                        has_emergency_fund=True, capital=200), settings.profile_path)
    settings.risk = pm.limits_for(pm.load(settings.profile_path))

    broker = MockBroker(equity=100_000)
    llm = MockLLM(responses={
        "AnalystReport": AnalystReport(symbol="AAA", stance="BULLISH",
                                       summary="Strong.", data_quality="FULL"),
        "ExecutiveDecision": ExecutiveDecision(symbol="AAA", action="BUY", conviction=5,
                                               thesis="Aligned.", invalidation="x",
                                               horizon_days=10),
    })
    data = FakeData()
    run_cycle(settings, broker, data, llm, Journal(settings.db_path))
    assert broker.orders, "expected a (tiny) buy"
    sym, qty, side = broker.orders[0]
    close = data.daily_bars(sym)["close"].iloc[-1]
    assert side == "buy"
    assert 1.0 <= qty * close <= 200 * settings.risk.max_position_pct + 1  # ~$20 max


def test_agent_failure_skips_symbol_not_cycle(tmp_path):
    settings = Settings(universe=["AAA"], candidates_per_cycle=1,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path)

    class ExplodingLLM(MockLLM):
        def complete(self, *a, **kw):
            raise RuntimeError("boom")

    broker = MockBroker()
    summary = run_cycle(settings, broker, FakeData(), ExplodingLLM(), Journal(settings.db_path))
    assert broker.orders == []
    assert summary["analyzed"] == []
