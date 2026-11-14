"""R17-L3: tests que matan los mutantes que sobreviven a la suite del PR #2 (R31).

Cada test nombra el mutante de mutacion/mutar.py (R17_anexos/l3_mutar.py) que
mata. Sin red: journal en tmp_path, puerta pura (risk.evaluate) y perfil en
tmp_path. Copiados de R17_anexos/l3_test_r17_l3_mutacion.py; único cambio:
test_siembra_no_crea_sesiones_posteriores_a_la_actual usa W = 2 en vez de 1,
porque W = 1 ya no es válido (R17 m3). Con W = 2 sigue matando a M28.
"""
from __future__ import annotations

import math

import pytest

from velveteentrade import profile as pm
from velveteentrade import risk
from velveteentrade.broker import AccountState
from velveteentrade.config import RiskLimits, Settings
from velveteentrade.memory import Journal
from velveteentrade.schemas import ExecutiveDecision, TechnicalSnapshot

from test_brechas import save_profile
from test_r31_freno_ventana import W, halted, sess


def _hist(j: Journal, rows) -> None:
    for ts, eq in rows:
        j._conn.execute("INSERT INTO equity_history (ts, equity) VALUES (?, ?)", (ts, eq))
    j._conn.commit()


# ---------------------------------------------------------------- risk.py
def test_frontera_exacta_del_umbral():
    """M05 ('<' -> '<=') y M09 (umbral + 1 punto). Caer EXACTAMENTE el 10 % no
    frena (la regla es 'más allá del umbral'); un centavo más, sí."""
    assert 1_000.0 * (1 - 0.10) == 900.0          # la frontera es exacta en binario
    assert not halted(1_000.0, 900.0)
    assert halted(1_000.0, 899.99)


def test_el_freno_mira_el_valor_del_satelite_no_la_equity_de_dimensionamiento():
    """M08. Tras una ganancia el valor V supera lo asignado E y el dimensionamiento
    usa min(V, E). El freno compara V (950) contra el pico (1 000), no E (850)."""
    lim = RiskLimits(max_drawdown_halt_pct=0.10)
    acct = AccountState(equity=850.0, cash=850.0)
    ctx = risk.CycleContext(peak_equity=1_000.0, current_value=950.0)
    d = ExecutiveDecision(symbol="X", action="BUY", conviction=5, thesis="t",
                          invalidation="i", horizon_days=10)
    s = TechnicalSnapshot(symbol="X", close=10.0, atr_14=0.2, hist_cvar_95_20d=-0.05)
    v = risk.evaluate(d, s, acct, lim, ctx)
    assert not any("drawdown" in r for r in v.reasons), v.reasons


# -------------------------------------------------------- memory.py: siembra
def test_siembra_agrupa_por_fecha_de_nueva_york(tmp_path):
    """M29. 03:00 UTC del 6 de enero son las 22:00 del 5 en Nueva York: ese pico
    es de la sesión del 5 y con W=2 ya salió de la ventana el 7."""
    j = Journal(tmp_path / "j.db")
    _hist(j, [("2026-01-06T03:00:00+00:00", 1_000.0),
              ("2026-01-06T15:00:00+00:00", 850.0)])
    assert j.update_window_peak("2026-01-07", 850.0, 2) == 850.0


def test_siembra_no_crea_sesiones_posteriores_a_la_actual(tmp_path):
    """M28. Una fila de historia con fecha posterior a `session` se cuenta como
    de la sesión actual; sin el recorte, el pico se perdería de la ventana."""
    j = Journal(tmp_path / "j.db")
    _hist(j, [("2026-01-10T15:00:00+00:00", 1_000.0)])
    assert j.update_window_peak("2026-01-05", 850.0, 2) == 1_000.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_siembra_ignora_pico_heredado_no_finito(tmp_path, bad):
    """M31. Un kv peak_equity no finito no entra a la ventana."""
    j = Journal(tmp_path / "j.db")
    j.set_float("peak_equity", bad)
    assert j.update_window_peak(sess(0), 1_000.0, W) == 1_000.0


def test_siembra_ignora_equity_no_finita_en_la_historia(tmp_path):
    """M34. Un inf en equity_history no se vuelve el pico de la ventana (si lo
    fuera, la puerta fallaría cerrada W sesiones seguidas)."""
    j = Journal(tmp_path / "j.db")
    _hist(j, [("2026-01-05T15:00:00+00:00", math.inf),
              ("2026-01-05T16:00:00+00:00", 900.0)])
    assert j.update_window_peak("2026-01-06", 1_000.0, W) == 1_000.0


# --------------------------------------------------------------- profile.py
def test_apply_profile_conserva_la_ventana_de_config(tmp_path):
    """M43. Al arrancar (cli -> apply_profile) los límites del perfil
    reemplazan a los de config.yaml, pero W sigue saliendo de config.yaml."""
    s = Settings(risk=RiskLimits(drawdown_window_sessions=63))
    s.profile_path = tmp_path / "profile.yaml"
    save_profile(s.profile_path)
    pm.apply_profile(s)
    assert s.risk.drawdown_window_sessions == 63


def test_siembra_incluye_la_fila_que_coincide_con_since(tmp_path):
    """M27. La base actual empieza EN `since` (como capital.pnl_since_base, que
    filtra ts >= since): una fila con ts == since es de la base nueva."""
    j = Journal(tmp_path / "j.db")
    ts = "2026-01-05T15:00:00+00:00"
    _hist(j, [("2026-01-02T15:00:00+00:00", 5_000.0), (ts, 1_000.0)])
    assert j.update_window_peak("2026-01-06", 850.0, W, since=ts) == 1_000.0


def test_siembra_toma_el_maximo_de_los_ciclos_de_un_dia(tmp_path):
    """M32. Dos ciclos el mismo día en Nueva York: la sesión vale su máximo,
    igual que en la tabla viva (ON CONFLICT ... MAX)."""
    j = Journal(tmp_path / "j.db")
    _hist(j, [("2026-01-05T15:00:00+00:00", 1_000.0),
              ("2026-01-05T18:00:00+00:00", 850.0)])
    assert j.update_window_peak("2026-01-06", 800.0, 2) == 1_000.0
