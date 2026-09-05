from velveteentrade.bot import format_informe
from velveteentrade.broker import MockBroker, OpenOrder, Position
from velveteentrade.memory import Journal
from velveteentrade.schemas import ExecutiveDecision, JournalEntry, RiskVerdict


def test_informe_with_positions_and_decisions(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.record_equity(200.0)
    import time
    time.sleep(0.002)
    j.record_equity(201.2)
    j.record(JournalEntry(
        symbol="MSFT",
        decision=ExecutiveDecision(symbol="MSFT", action="BUY", conviction=4,
                                   thesis="Momentum alineado con fundamentales. Riesgo acotado.",
                                   invalidation="Cierre bajo SMA200.", horizon_days=12),
        verdict=RiskVerdict(symbol="MSFT", approved=True, action="BUY", qty=0.024),
        executed=True))
    broker = MockBroker(equity=100_000)
    broker.state.positions["MSFT"] = Position("MSFT", 0.024, 499.7, 12.05)
    broker.pending = [OpenOrder("o1", "JNJ", 0.05, "buy")]

    text = format_informe(broker, j, 200.0)
    assert "$201.20" in text and "+$1.20" in text          # real P&L, no brackets
    assert "MSFT: 0.024" in text
    assert "BUY JNJ" in text                                # queued order visible
    assert "✅ MSFT BUY conv 4/5" in text
    assert "Momentum alineado" in text
    assert "[pendiente" not in text


def test_informe_empty_state(tmp_path):
    j = Journal(tmp_path / "j.db")
    text = format_informe(MockBroker(), j, 200.0)
    assert "sin ciclos registrados" in text
    assert "Sin posiciones abiertas" in text
