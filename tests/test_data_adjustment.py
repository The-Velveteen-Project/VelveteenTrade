"""R30: daily bars must be requested split/dividend-adjusted (no network)."""
import sys
import types

import pandas as pd

from velveteentrade.data import MarketData


def test_fetch_requests_adjusted_bars(monkeypatch, tmp_path):
    seen = {}

    class Adjustment:
        ALL = "all"
        RAW = "raw"

    class StockBarsRequest:
        def __init__(self, **kw):
            seen.update(kw)

    class _Bars:
        df = pd.DataFrame({"symbol": ["X"], "open": [1.0], "high": [1.0], "low": [1.0],
                           "close": [1.0], "volume": [1.0]},
                          index=pd.Index([pd.Timestamp("2026-10-23", tz="UTC")], name="timestamp"))

    class StockHistoricalDataClient:
        def __init__(self, *a, **k):
            pass

        def get_stock_bars(self, req):
            return _Bars()

    mods = {
        "alpaca": types.ModuleType("alpaca"),
        "alpaca.data": types.ModuleType("alpaca.data"),
        "alpaca.data.enums": types.SimpleNamespace(Adjustment=Adjustment),
        "alpaca.data.historical": types.SimpleNamespace(StockHistoricalDataClient=StockHistoricalDataClient),
        "alpaca.data.requests": types.SimpleNamespace(StockBarsRequest=StockBarsRequest),
        "alpaca.data.timeframe": types.SimpleNamespace(TimeFrame=types.SimpleNamespace(Day="1Day")),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    md = MarketData("k", "s", tmp_path)
    md._fetch_alpaca("X", 10)
    assert seen.get("adjustment") == "all"
