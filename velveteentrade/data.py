"""Market data with a local CSV cache.

Primary source: Alpaca Market Data (IEX feed — free with any Alpaca account).
Fundamentals: yfinance, best-effort (the system degrades gracefully to
technical-only analysis when fundamentals are unavailable).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_TTL_HOURS = 12


class MarketData:
    def __init__(self, api_key: str, secret_key: str, cache_dir: Path) -> None:
        self._key = api_key
        self._secret = secret_key
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ bars
    def daily_bars(self, symbol: str, days: int = 320) -> pd.DataFrame:
        """Daily OHLCV, cached. Columns: open, high, low, close, volume."""
        cached = self._read_cache(symbol)
        if cached is not None:
            return cached.tail(days)
        df = self._fetch_alpaca(symbol, days)
        self._write_cache(symbol, df)
        return df

    def _fetch_alpaca(self, symbol: str, days: int) -> pd.DataFrame:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(self._key, self._secret)
        start = datetime.now(timezone.utc) - timedelta(days=int(days * 1.6))
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
        bars = client.get_stock_bars(req).df
        if bars.empty:
            raise ValueError(f"No bars returned for {symbol}")
        df = bars.reset_index()
        df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        return df.tail(days)

    # ----------------------------------------------------------------- cache
    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}.csv"

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        p = self._cache_path(symbol)
        if not p.exists():
            return None
        age_hours = (time.time() - p.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception:  # corrupted cache → refetch
            return None

    def _write_cache(self, symbol: str, df: pd.DataFrame) -> None:
        df.to_csv(self._cache_path(symbol))

    # ---------------------------------------------------------- fundamentals
    def fundamentals(self, symbol: str) -> dict:
        """Best-effort fundamental snapshot via yfinance. Returns {} on failure."""
        p = self.cache_dir / f"{symbol.upper()}_fund.json"
        if p.exists() and (time.time() - p.stat().st_mtime) / 3600 < 24 * 5:
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        try:
            import yfinance as yf

            info = yf.Ticker(symbol).info or {}
            keys = [
                "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                "priceToBook", "profitMargins", "operatingMargins", "returnOnEquity",
                "debtToEquity", "revenueGrowth", "earningsGrowth", "freeCashflow",
                "totalCash", "totalDebt", "dividendYield", "recommendationKey",
            ]
            out = {k: info.get(k) for k in keys if info.get(k) is not None}
            if out:
                p.write_text(json.dumps(out))
            return out
        except Exception as exc:
            log.warning("Fundamentals unavailable for %s: %s", symbol, exc)
            return {}

    def news(self, symbol: str, limit: int = 8) -> list[dict]:
        """Recent headlines via Alpaca News API (Benzinga, free tier). Best-effort."""
        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest

            client = NewsClient(self._key, self._secret)
            req = NewsRequest(symbols=symbol, limit=limit)
            items = client.get_news(req).data.get("news", [])
            return [
                {"headline": n.headline, "created_at": str(n.created_at), "summary": (n.summary or "")[:300]}
                for n in items
            ]
        except Exception as exc:
            log.warning("News unavailable for %s: %s", symbol, exc)
            return []
