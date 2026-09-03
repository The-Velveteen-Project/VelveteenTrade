import numpy as np
import pandas as pd
import pytest

from velveteentrade import indicators


def make_df(n=300, drift=0.0005, vol=0.01, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": rng.integers(1e5, 1e6, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_snapshot_complete():
    snap = indicators.snapshot("TEST", make_df())
    assert snap.symbol == "TEST"
    assert snap.sma_50 is not None and snap.sma_200 is not None
    assert 0 <= snap.rsi_14 <= 100
    assert snap.atr_14 > 0
    assert snap.realized_vol_21d > 0
    assert snap.hist_cvar_95_20d is not None and snap.hist_cvar_95_20d < 0
    assert snap.dist_from_52w_high <= 0.0001


def test_snapshot_short_history_degrades_gracefully():
    snap = indicators.snapshot("TEST", make_df(n=30))
    assert snap.sma_200 is None
    assert snap.momentum_63d is None
    assert snap.close > 0


def test_rsi_bounds_on_monotonic_series():
    up = pd.Series(np.linspace(100, 200, 100))
    val = indicators.rsi(up).iloc[-1]
    assert val > 70


def test_cvar_is_tail_mean():
    close = make_df(n=400, vol=0.02)["close"]
    cvar = indicators.historical_cvar(close)
    rets = close.pct_change(20).dropna()
    var = rets.quantile(0.05)
    assert cvar <= var  # expected shortfall is at least as bad as VaR


def test_screen_ranks_by_risk_adjusted_momentum():
    strong = indicators.snapshot("UP", make_df(drift=0.002, vol=0.008, seed=1))
    weak = indicators.snapshot("DOWN", make_df(drift=-0.002, vol=0.02, seed=2))
    top = indicators.screen([weak, strong], top_n=1)
    assert top[0].symbol == "UP"
