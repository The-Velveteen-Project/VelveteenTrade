"""C1: el satélite no abre su base sobre posiciones heredadas.

Escenario R1 de la revisión del 14 nov 2026 (revision/salida_redespliegue.txt):
el motor operaba toda la cuenta (modo "profile" sin perfil, S = equity) y
compró $10.000 de X y $10.000 de Y. Al entrar separate_account (S = $2.000),
el satélite heredaba esa cartera: en el segundo ciclo una caída de $650 en la
cuenta bajaba V de 2000 a 1350 y el freno quedaba puesto 126 sesiones.

Con la protección, la base nueva se pospone mientras las posiciones valgan más
que S (o haya compras pendientes). No se compra; stops y ventas siguen. Al
cerrar las posiciones a mano, el ciclo siguiente abre la base limpia.
Sin red: bróker, datos y LLM son falsos.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from velveteentrade import capital
from velveteentrade.bot import format_cycle_summary, format_estado
from velveteentrade.broker import AccountState, MockBroker, OpenOrder, Position
from velveteentrade.config import Allocation, RiskLimits
from velveteentrade.memory import Journal

from test_brechas import buy_all, make_settings, run_cycle, save_profile, snap

SEP = dict(mode="separate_account", total_capital=10_000, satellite_pct_paper=0.20,
           satellite_pct_live=0.10)
FP_SEP = {"capital": 2_000.0, "core": 0.0, "paper": True}
FP_MAIN = {"capital": None, "core": 0.0}          # modo "profile" sin perfil: toda la cuenta


class Cuenta(MockBroker):
    """Cuenta simulada: llena al precio de la sesión y recalcula la equity."""

    def __init__(self, cash=100_000.0):
        super().__init__(equity=cash)
        self.cash, self.qty, self.entry, self.px = cash, {}, {}, {}

    def account(self):
        pos = {s: Position(s, q, self.entry[s], q * self.px[s]) for s, q in self.qty.items()}
        eq = self.cash + sum(p.market_value for p in pos.values())
        return AccountState(equity=eq, cash=self.cash, positions=pos)

    def submit_order(self, symbol, qty, side):
        oid = super().submit_order(symbol, qty, side)
        q0 = self.qty.get(symbol, 0.0)
        q = q0 + (qty if side == "buy" else -qty)
        self.cash += (-1 if side == "buy" else 1) * qty * self.px[symbol]
        if side == "buy":
            self.entry[symbol] = self.px[symbol]
        if abs(q) < 1e-12:
            self.qty.pop(symbol, None)
            self.entry.pop(symbol, None)
        else:
            self.qty[symbol] = q
        return oid


def ciclo(s, cuenta, monkeypatch, j, dia, px, llm=None):
    cuenta.px = dict(px)
    cuenta.session = date(2026, 11, dia)
    s.universe = list(px)
    n0 = len(cuenta.orders)
    summary, j = run_cycle(s, cuenta, llm or buy_all(conv=5), monkeypatch,
                           {k: snap(k, close=v, atr=1.0) for k, v in px.items()}, journal=j)
    return summary, cuenta.orders[n0:], j


def peaks(j):
    return j._conn.execute("SELECT session, peak FROM session_peaks ORDER BY session").fetchall()


# ------------------------------------------------------- escenario R1
def test_r1_posiciones_heredadas_bloquean_compras_y_no_rebasan(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "r1", ["X"], limits=RiskLimits(max_drawdown_halt_pct=0.10))
    s.alpaca_paper = True
    cuenta = Cuenta()
    # R1.1 y R1.2: el motor opera toda la cuenta (como main sin perfil).
    _, o1, j = ciclo(s, cuenta, monkeypatch, None, 12, {"X": 50.0})
    _, o2, j = ciclo(s, cuenta, monkeypatch, j, 13, {"X": 50.0, "Y": 20.0})
    assert o1 == [("X", 200.0, "buy")] and o2 == [("Y", 500.0, "buy")]          # $10.000 cada una
    assert capital.current_base(j)["fp"] == FP_MAIN

    # R1.3: entra separate_account. Posiciones por $20.000 > S = $2.000.
    s.allocation = Allocation(**SEP)
    sm3, o3, j = ciclo(s, cuenta, monkeypatch, j, 16, {"X": 50.0, "Y": 20.0, "Z": 30.0})
    assert o3 == []
    assert sm3["buys_blocked"] and "$20,000.00" in sm3["buys_blocked"]
    assert "cerrar esas posiciones a mano" in sm3["buys_blocked"]
    assert "reiniciar la cuenta paper" in sm3["buys_blocked"]
    assert len(sm3["rejected"]) == 3
    assert all("satellite base on hold" in r for r in sm3["rejected"]), sm3["rejected"]
    assert capital.current_base(j)["fp"] == FP_MAIN                             # no re-basó
    bl = capital.rebase_blocked(j)
    assert bl["positions_value"] == pytest.approx(20_000.0)
    assert bl["allocated"] == pytest.approx(2_000.0)
    assert "2026-11-16" not in dict(peaks(j))                                   # ni pico ni V

    # R1.4: X −4 %, Y −2,5 %. El stop de X vende; sigue sin comprar y sin freno
    # (antes: V = 1350 y «drawdown beyond 10%» en todas las compras).
    sm4, o4, j = ciclo(s, cuenta, monkeypatch, j, 17,
                       {"X": 48.0, "Y": 19.5, "Z": 30.0, "Q": 10.0})
    assert o4 == [("X", 200.0, "sell")]
    assert sm4["stops"] == ["X"]
    # Se mide al empezar el ciclo, antes del stop: X 200 x 48 + Y 500 x 19,5.
    assert sm4["buys_blocked"] and "$19,350.00" in sm4["buys_blocked"]
    assert cuenta.qty == {"Y": 500.0}                                           # Y sigue ahí
    assert not any("drawdown" in r for r in sm4["rejected"]), sm4["rejected"]
    assert capital.current_base(j)["fp"] == FP_MAIN

    # El usuario cierra Y a mano. El ciclo siguiente abre la base limpia.
    cuenta.submit_order("Y", 500.0, "sell")
    assert cuenta.account().positions == {}
    eq = cuenta.account().equity
    sm5, o5, j = ciclo(s, cuenta, monkeypatch, j, 18, {"X": 48.0, "Y": 19.5, "Z": 30.0, "Q": 10.0})
    base = capital.current_base(j)
    assert base["fp"] == FP_SEP and base["allocated"] == pytest.approx(2_000.0)
    assert base["start_equity"] == pytest.approx(eq)                            # pérdidas viejas fuera
    assert capital.rebase_blocked(j) is None and not sm5["buys_blocked"]
    assert peaks(j) == [("2026-11-18", 2_000.0)]
    assert o5 and all(side == "buy" for _, _, side in o5)
    for sym, qty, _ in o5:
        assert qty * cuenta.px[sym] <= 0.10 * 2_000.0 + 1e-9                    # tope de config


def test_r1_estado_y_resumen_dicen_que_hacer(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "r1e", ["X"])
    s.alpaca_paper = True
    cuenta = Cuenta()
    _, _, j = ciclo(s, cuenta, monkeypatch, None, 12, {"X": 50.0})
    s.allocation = Allocation(**SEP)
    sm, _, j = ciclo(s, cuenta, monkeypatch, j, 16, {"X": 50.0, "Z": 30.0})
    txt = format_estado(cuenta, j, 2_000.0, base_current=capital.base_is_current(
        j, None, s.allocation, True))
    assert "⛔" in txt and "en espera" in txt
    assert "cerrar esas posiciones a mano" in txt and "reiniciar la cuenta paper" in txt
    assert "se renueva en el próximo ciclo" not in txt
    resumen = format_cycle_summary(sm, j)
    assert "⛔ Compras bloqueadas" in resumen


# ------------------------------------------------------- cuenta plana
def test_cuenta_plana_rebasa_limpio(tmp_path, monkeypatch):
    """Escenario R2: el motor con un perfil sintético de tolerancia 1 y capital
    150 no compró nada (S = $15). Al entrar separate_account la cuenta está
    plana: re-basa a S = $2.000 en el primer ciclo y compra."""
    s = make_settings(tmp_path, "r2", ["X"], limits=RiskLimits(max_drawdown_halt_pct=0.10))
    s.alpaca_paper = True
    save_profile(s.profile_path, tolerance=1, max_drawdown_comfort=0.05, capital=150.0)
    cuenta = Cuenta()
    _, o1, j = ciclo(s, cuenta, monkeypatch, None, 12, {"X": 50.0})
    assert o1 == [] and capital.current_base(j)["allocated"] == pytest.approx(15.0)
    s.allocation = Allocation(**SEP)
    sm, o2, j = ciclo(s, cuenta, monkeypatch, j, 16, {"X": 50.0})
    assert not sm["buys_blocked"] and capital.rebase_blocked(j) is None
    assert capital.current_base(j)["fp"] == FP_SEP
    assert o2 == [("X", 2.0, "buy")]                                            # $100: 5 % del perfil
    assert peaks(j) == [("2026-11-16", 2_000.0)]


def test_estado_sin_bloqueo_no_muestra_aviso(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "plana", ["X"])
    s.alpaca_paper = True
    s.allocation = Allocation(**SEP)
    cuenta = Cuenta()
    _, _, j = ciclo(s, cuenta, monkeypatch, None, 16, {"X": 50.0})
    assert "⛔" not in format_estado(cuenta, j, 2_000.0)


# ------------------------------------------------------- capital.resolve
def j_(tmp_path):
    return Journal(tmp_path / "j.db")


def test_posiciones_que_valen_s_o_menos_no_bloquean(tmp_path):
    for v in (0.0, 1_500.0, 2_000.0):
        j = Journal(tmp_path / f"j{v}.db")
        b = capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP), positions_value=v)
        assert b.rebased and b.blocked is None and b.sizing == pytest.approx(2_000.0)


def test_posiciones_por_encima_de_s_posponen_la_base(tmp_path):
    j = j_(tmp_path)
    b = capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP), positions_value=2_000.01)
    assert b.blocked and not b.rebased and b.sizing == 0.0
    assert capital.current_base(j) is None
    first = capital.rebase_blocked(j)["since"]
    capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP), positions_value=5_000.0)
    assert capital.rebase_blocked(j)["since"] == first                          # conserva el inicio
    b = capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP), positions_value=0.0)
    assert b.rebased and not b.blocked and capital.rebase_blocked(j) is None


def test_compras_pendientes_posponen_la_base(tmp_path):
    j = j_(tmp_path)
    b = capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP), pending_buys=1)
    assert b.blocked and "orden(es) de compra pendiente(s)" in b.blocked and b.sizing == 0.0


def test_pipeline_con_compra_pendiente_no_rebasa(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "pend", ["X"])
    s.alpaca_paper = True
    s.allocation = Allocation(**SEP)
    cuenta = Cuenta()
    cuenta.pending.append(OpenOrder(id="o1", symbol="Y", qty=500.0, side="buy"))
    sm, o, j = ciclo(s, cuenta, monkeypatch, None, 16, {"X": 50.0, "Y": 20.0})
    assert o == [] and sm["buys_blocked"] and capital.current_base(j) is None


def test_valor_no_finito_falla_cerrado(tmp_path):
    b = capital.resolve(j_(tmp_path), None, 100_000.0, allocation=Allocation(**SEP),
                        positions_value=math.nan)
    assert b.blocked and b.sizing == 0.0


def test_dentro_de_la_base_las_posiciones_no_bloquean(tmp_path):
    """El guardia mira solo al abrir una base nueva. Con la base en vigor, las
    posiciones del propio satélite (aunque valgan más que S tras ganar) no frenan."""
    j = j_(tmp_path)
    capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP))
    b = capital.resolve(j, None, 101_000.0, allocation=Allocation(**SEP), positions_value=2_900.0)
    assert not b.rebased and not b.blocked and b.value == pytest.approx(3_000.0)


def test_modo_profile_no_aplica_el_guardia(tmp_path):
    """En modo "profile" el núcleo de ETF vive en esta cuenta: sus posiciones
    valen más que S por diseño y no deben bloquear."""
    j = j_(tmp_path)
    b = capital.resolve(j, None, 100_000.0, positions_value=60_000.0, pending_buys=2)
    assert b.rebased and not b.blocked
