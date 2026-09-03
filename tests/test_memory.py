from velveteentrade.broker import Position
from velveteentrade.memory import Journal


def test_peak_equity_persists_and_only_rises(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.update_peak_equity(100_000) == 100_000
    assert j.update_peak_equity(110_000) == 110_000
    assert j.update_peak_equity(90_000) == 110_000  # drawdown does not lower the peak
    j.close()
    j2 = Journal(tmp_path / "j.db")  # survives restart
    assert j2.get_float("peak_equity") == 110_000


def test_reconcile_entries_uses_broker_fills(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_entry_price("JPM", 310.0)   # provisional last-close price at order time
    j.set_entry_price("GONE", 50.0)   # position no longer held

    positions = {
        "JPM": Position(symbol="JPM", qty=27, avg_entry=312.4, market_value=8434.8),
    }
    j.reconcile_entries(positions)

    entries = j.entry_prices()
    assert entries["JPM"] == 312.4    # corrected to the real fill
    assert "GONE" not in entries      # cleared


def test_lessons_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.add_lesson("AAPL", "Momentum entries near 52w high needed wider stops.")
    assert j.lessons("AAPL") == ["Momentum entries near 52w high needed wider stops."]
    assert j.lessons("MSFT") == []
