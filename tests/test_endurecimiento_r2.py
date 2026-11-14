"""Endurecimiento, ronda 2 (14 nov 2026): defectos heredados de main que el
barrido de stops no cubría. Cada test reproduce el caso del hallazgo."""
from __future__ import annotations

import pytest

from velveteentrade import pipeline
from velveteentrade.broker import MockBroker, Position
from velveteentrade.memory import Journal

from tests.test_brechas import EQ, ScriptedLLM, buy_all, make_settings, run_cycle, snap


class ExplodingSellBroker(MockBroker):
    """La venta de `bad` lanza (p. ej. un APIError síncrono de Alpaca)."""

    def __init__(self, bad, **kw):
        super().__init__(**kw)
        self.bad = bad

    def submit_order(self, symbol, qty, side):
        if symbol == self.bad and side == "sell":
            raise RuntimeError(f"APIError simulado al vender {symbol}")
        return super().submit_order(symbol, qty, side)


def _dos_bajo_el_stop(tmp_path, bad):
    broker = ExplodingSellBroker(bad, equity=EQ)
    broker.state.positions = {"A": Position("A", 100, 100.0, 9_000.0),
                              "B": Position("B", 100, 100.0, 9_000.0)}
    s = make_settings(tmp_path, f"x{bad}", ["A", "B", "C"])
    j = Journal(s.db_path)
    j.set_stop("A", 96.0)
    j.set_stop("B", 96.0)
    snaps = {"A": snap("A", close=90.0), "B": snap("B", close=90.0), "C": snap("C", close=50.0)}
    return broker, s, j, snaps


@pytest.mark.parametrize("bad", ["A", "B"])
def test_una_venta_que_lanza_no_bloquea_los_stops_siguientes(tmp_path, monkeypatch, bad):
    broker, s, j, snaps = _dos_bajo_el_stop(tmp_path, bad)
    for _ in range(3):   # la causa persiste: el corte se repetía en cada ciclo
        with pytest.raises(pipeline.StopsNotSent) as e:
            run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
        assert bad in str(e.value)
    otro = "B" if bad == "A" else "A"
    vendidos = [sym for sym, _, sd in broker.orders if sd == "sell"]
    # Antes del arreglo, con bad="A": vendidos == [] en los tres ciclos.
    assert vendidos == [otro] * 3
    # Falla cerrado para las compras: la puerta no corre con un stop sin enviar.
    assert not any(sd == "buy" for _, _, sd in broker.orders)


def test_sin_excepcion_salen_los_dos_stops(tmp_path, monkeypatch):
    broker, s, j, snaps = _dos_bajo_el_stop(tmp_path, bad="ninguno")
    summary, _ = run_cycle(s, broker, ScriptedLLM(lambda x: {}), monkeypatch, snaps, journal=j)
    assert sorted(summary["stops"]) == ["A", "B"]
