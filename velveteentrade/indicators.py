"""Deterministic technical indicators. Pure pandas/numpy — no LLM anywhere near this file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import TechnicalSnapshot

TRADING_DAYS = 252


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0)  # all-gain windows → RSI 100


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    rets = np.log(close / close.shift(1))
    return rets.rolling(window).std() * np.sqrt(TRADING_DAYS)


def historical_cvar(close: pd.Series, horizon: int = 20, alpha: float = 0.95) -> float | None:
    """Empirical CVaR (expected shortfall) of overlapping `horizon`-day returns.

    Honest, assumption-free tail estimate. Negative number, e.g. -0.12 means the
    average of the worst 5% of 20-day windows lost 12%.
    """
    rets = close.pct_change(horizon).dropna()
    if len(rets) < 60:
        return None
    var = rets.quantile(1 - alpha)
    tail = rets[rets <= var]
    return float(tail.mean()) if len(tail) else None


def snapshot(symbol: str, df: pd.DataFrame) -> TechnicalSnapshot:
    """Compute a full indicator snapshot from an OHLCV daily dataframe."""
    close = df["close"]
    n = len(close)

    def last(series: pd.Series) -> float | None:
        v = series.iloc[-1] if len(series) else np.nan
        return None if pd.isna(v) else float(v)

    high_52w = close.tail(252).max() if n >= 20 else np.nan
    return TechnicalSnapshot(
        symbol=symbol,
        close=float(close.iloc[-1]),
        sma_50=last(close.rolling(50).mean()) if n >= 50 else None,
        sma_200=last(close.rolling(200).mean()) if n >= 200 else None,
        rsi_14=last(rsi(close)) if n >= 15 else None,
        atr_14=last(atr(df)) if n >= 15 else None,
        momentum_63d=float(close.iloc[-1] / close.iloc[-64] - 1) if n >= 64 else None,
        momentum_21d=float(close.iloc[-1] / close.iloc[-22] - 1) if n >= 22 else None,
        realized_vol_21d=last(realized_vol(close)) if n >= 22 else None,
        hist_cvar_95_20d=historical_cvar(close),
        dist_from_52w_high=float(close.iloc[-1] / high_52w - 1) if not pd.isna(high_52w) else None,
    )


def screen(snapshots: list[TechnicalSnapshot], top_n: int) -> list[TechnicalSnapshot]:
    """Deterministic pre-screen so the LLM only analyzes plausible candidates.

    Rank by 3-month momentum adjusted by volatility (a crude Sharpe of the recent
    trend). Cheap, transparent, and it saves ~80% of LLM calls.
    """
    scored = []
    for s in snapshots:
        if s.momentum_63d is None or s.realized_vol_21d in (None, 0):
            continue
        scored.append((s.momentum_63d / max(s.realized_vol_21d, 0.05), s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in scored[:top_n]]
