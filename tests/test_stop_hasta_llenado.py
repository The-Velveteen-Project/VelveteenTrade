"""Endurecimiento R3: el stop congelado sobrevive a una SELL que no se llena.

pipeline.py borraba entrada y stop (journal.clear_entry) al enviar la venta.
Si no se llenaba (rechazo, cancelación o parcial que vence), el lote quedaba
sin stop congelado y check_stops volvía al stop móvil entry - k·ATR de hoy,
que rompe G2. MockBroker no llena nada, así que modela justo ese caso.
"""
from velveteentrade.broker import MockBroker, Position

from test_brechas import EQ, buy_all, make_settings, run_cycle, snap


def test_stop_no_llenado_conserva_el_stop_congelado(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=100.0, atr=2.0)})
    assert j.stop_prices() == {"X": 96.0}
    broker.state.positions = {"X": Position("X", 100, 100.0, 9_000.0)}
    broker.orders.clear()
    # A cae a 90 con ATR 7,2: el stop congelado (96) dispara y se envía la venta.
    run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=90.0, atr=7.2)}, journal=j)
    assert ("X", 100, "sell") in [(a, int(q), sd) for a, q, sd in broker.orders]
    # La venta no se llenó: la posición sigue y el stop congelado también.
    assert j.stop_prices() == {"X": 96.0}
    broker.orders.clear()
    run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=90.0, atr=7.2)}, journal=j)
    # Con el stop congelado, el ciclo siguiente a 90 vuelve a vender (antes, el
    # respaldo 100 - 2·7,2 = 85,6 no disparaba).
    assert [(a, sd) for a, q, sd in broker.orders] == [("X", "sell")]


def test_stop_se_limpia_cuando_la_posicion_desaparece(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=100.0, atr=2.0)})
    broker.state.positions = {}                # la posición ya no está (venta llenada)
    run_cycle(s, broker, buy_all(conv=0), monkeypatch, {"X": snap("X", close=100.0, atr=2.0)},
              journal=j)
    assert "X" not in j.stop_prices()
