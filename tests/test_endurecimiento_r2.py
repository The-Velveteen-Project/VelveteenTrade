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


# ------------------------------------------------------------------ splits
from velveteentrade import risk                      # noqa: E402
from velveteentrade.broker import AccountState       # noqa: E402
from velveteentrade.config import RiskLimits         # noqa: E402


def _barrido_tras(tmp_path, nombre, q1, e1, marca):
    """Posición A x100 a 100 con stop congelado en 96; luego el bróker
    reporta (q1, e1) y el precio de hoy es `marca` en la escala nueva."""
    j = Journal(tmp_path / f"{nombre}.db")
    j.reconcile_entries({"A": Position("A", 100.0, 100.0, 10_000.0)})
    j.set_stop("A", 96.0)
    pos = Position("A", q1, e1, q1 * marca)
    j.reconcile_entries({"A": pos})
    cuenta = AccountState(equity=EQ, cash=EQ - pos.market_value, positions={"A": pos})
    v = risk.check_stops(cuenta, {"A": snap("A", close=marca)}, RiskLimits(),
                         j.entry_prices(), stops=j.stop_prices())
    return j.stop_prices()["A"], [(x.symbol, x.qty) for x in v]


def test_split_2x1_no_vende_una_posicion_que_no_perdio_valor(tmp_path):
    stop, ventas = _barrido_tras(tmp_path, "s2", 200.0, 50.0, marca=50.0)
    # Antes: stop 96 y venta de A x200 (el valor seguía en 10 000).
    assert stop == pytest.approx(48.0)
    assert ventas == []


def test_contrasplit_1x10_sigue_protegiendo(tmp_path):
    stop, ventas = _barrido_tras(tmp_path, "s10", 10.0, 1_000.0, marca=900.0)
    # 900 en la escala nueva = 90 en la vieja, bajo el stop de 96.
    # Antes: stop 96 contra 900, ninguna venta (G2 roto).
    assert stop == pytest.approx(960.0)
    assert ventas == [("A", 10.0)]


@pytest.mark.parametrize("q1,e1", [
    (150.0, 100.0),    # compra adicional al mismo precio: la base sube
    (200.0, 52.0),     # compra grande muy por debajo: base 10 400, no 10 000
    (50.0, 100.0),     # venta parcial: la base baja
    (100.0, 100.0),    # nada cambió
])
def test_compras_y_ventas_no_se_confunden_con_un_split(tmp_path, q1, e1):
    j = Journal(tmp_path / "n.db")
    j.reconcile_entries({"A": Position("A", 100.0, 100.0, 10_000.0)})
    j.set_stop("A", 96.0)
    j.reconcile_entries({"A": Position("A", q1, e1, q1 * e1)})
    assert j.stop_prices()["A"] == 96.0
