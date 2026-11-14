"""R17-L1: casos borde del freno por drawdown con ventana (PR #2, R31).

Selección de trabajo_R17/l1_logica/tests/test_r17_l1_bordes.py. Entran los
tests que fijan comportamiento deseado. Los que fijaban un defecto como si
fuera correcto quedan así:
- W = 1, W enorme y subir W (m3, m2, m1): corregidos en este PR; sus tests
  están en test_r17_cotas_ventana.py y test_r17_sin_poda.py.
- Caída lenta (M1) y suelta en plena caída (M2): defectos de diseño que la
  especificación Lean admite. Van como xfail estrictos con el comportamiento
  que se esperaría de un freno que acota la pérdida; si un cambio futuro los
  hace pasar, la suite lo avisa (XPASS estricto = fallo).
- Patrimonio NaN que tumba el ciclo antes de los stops (m9, viene de main):
  xfail estricto; el arreglo va aparte, desde main.
- No entran: depósitos y retiros (i4, pregunta abierta), sesión que retrocede
  (m8, semántica sin decidir), journal borrado (hecho del entorno, no del
  código) y los duplicados de tests del PR o de L3.

Sin red: journal en tmp_path, broker y LLM falsos.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date

import pytest

from velveteentrade import capital as capital_mod
from velveteentrade.broker import MockBroker
from velveteentrade.config import WINDOW_MAX, RiskLimits
from velveteentrade.memory import Journal

from test_brechas import buy_all, make_settings, run_cycle, save_profile, snap as snap_b
from test_r31_freno_ventana import buy_verdict, halted, sess

W = 126


# ======================================================= W > historia
def test_w_mayor_que_la_historia_sembrada_desde_equity_history(tmp_path):
    j = Journal(tmp_path / "j.db")
    for ts, eq in [("2026-01-05T15:00:00+00:00", 1_000.0),
                   ("2026-01-06T15:00:00+00:00", 870.0)]:
        j._conn.execute("INSERT INTO equity_history (ts, equity) VALUES (?, ?)", (ts, eq))
    j._conn.commit()
    peak = j.update_window_peak("2026-01-07", 880.0, WINDOW_MAX)
    assert peak == 1_000.0 and halted(peak, 880.0)


# ===================================================== historia vacía
def test_historia_vacia_el_primer_valor_es_el_pico(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.update_window_peak(sess(0), 600.0, W) == 600.0
    assert not halted(600.0, 600.0)


def test_historia_vacia_y_valor_nan_da_none_y_la_puerta_rechaza(tmp_path):
    j = Journal(tmp_path / "j.db")
    peak = j.update_window_peak(sess(0), math.nan, W)
    assert peak is None
    v = buy_verdict(peak, math.nan)
    assert not v.approved and "cannot be evaluated" in v.reasons[-1]


# ===================================================== NaN e infinito
def test_infinito_se_ignora_en_la_ventana_y_la_puerta_rechaza(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, W)
    peak = j.update_window_peak(sess(1), math.inf, W)
    assert peak == 1_000.0
    assert not buy_verdict(peak, math.inf).approved
    assert not buy_verdict(peak, -math.inf).approved


@pytest.mark.xfail(raises=sqlite3.IntegrityError, strict=True,
                   reason="R17 m9 (viene de main): record_equity(NaN) viola NOT NULL "
                          "antes del barrido de stops. Arreglo aparte, desde main.")
def test_equity_nan_no_deberia_tumbar_el_ciclo(tmp_path, monkeypatch):
    """Lo deseado: con un patrimonio NaN del broker el ciclo termina, la compra
    se rechaza (fallo cerrado) y el barrido de stops corre. Hoy el ciclo aborta
    con IntegrityError en record_equity, antes de los stops."""
    s = make_settings(tmp_path, "x", ["X"], limits=RiskLimits(drawdown_window_sessions=3))
    save_profile(s.profile_path, capital=1_000.0)
    snaps = {"X": snap_b("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=100_000.0)
    broker.session = date(2026, 3, 2)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps)
    broker.orders.clear()
    broker.state.equity = math.nan
    broker.session = date(2026, 3, 3)
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert broker.orders == []


# ==================================================== re-base del capital
def test_rebase_legado_sin_perfil_conserva_el_pico_kv(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_float("peak_equity", 120_000.0)                    # pico de main
    b = capital_mod.resolve(j, None, 100_000.0)
    assert b.rebased
    peak = j.update_window_peak(sess(0), b.value, 3, since=capital_mod.current_base(j)["since"])
    assert peak == 120_000.0 and halted(peak, b.value)


# =================================================== reinicio del proceso
def test_reinicio_conserva_la_ventana(tmp_path):
    p = tmp_path / "j.db"
    j = Journal(p)
    j.update_window_peak(sess(0), 1_000.0, W)
    j.update_window_peak(sess(1), 850.0, W)
    j._conn.close()
    j2 = Journal(p)                                           # proceso nuevo
    peak = j2.update_window_peak(sess(1), 850.0, W)          # mismo día, otro ciclo
    assert peak == 1_000.0 and halted(peak, 850.0)
    assert j2._conn.execute("SELECT COUNT(*) FROM session_peaks").fetchone()[0] == 2


# ============================================ sesión y zona horaria
def test_siembra_por_fecha_de_nueva_york_deja_una_fila_por_sesion(tmp_path):
    j = Journal(tmp_path / "j.db")
    rows = [("2026-01-05T15:00:00+00:00", 900.0),            # 10:00 NY, 5 ene
            ("2026-01-06T03:00:00+00:00", 1_000.0),          # 22:00 NY, 5 ene
            ("2026-01-06T15:00:00+00:00", 950.0)]            # 10:00 NY, 6 ene
    for ts, eq in rows:
        j._conn.execute("INSERT INTO equity_history (ts, equity) VALUES (?, ?)", (ts, eq))
    j._conn.commit()
    j.update_window_peak("2026-01-07", 940.0, 3)
    got = dict(j._conn.execute("SELECT session, peak FROM session_peaks").fetchall())
    assert got == {"2026-01-05": 1_000.0, "2026-01-06": 950.0, "2026-01-07": 940.0}


def test_varios_ciclos_en_fin_de_semana_caen_en_la_misma_sesion(tmp_path):
    """Con el reloj de Alpaca, sábado y domingo dan next_open = lunes: no suman
    sesiones (se simula con la misma etiqueta)."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak("2026-03-09", 1_000.0, 2)           # viernes 21:30 UTC -> lunes 9
    j.update_window_peak("2026-03-09", 850.0, 2)             # sábado -> lunes 9
    j.update_window_peak("2026-03-09", 850.0, 2)             # domingo -> lunes 9
    assert j.update_window_peak("2026-03-10", 850.0, 2) == 1_000.0


def test_la_ventana_cuenta_sesiones_con_registro_no_calendario(tmp_path):
    """Si el bot no corre (despliegue caído), la ventana abarca más calendario:
    el pico dura más. Conservador."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak("2026-01-02", 1_000.0, 3)
    j.update_window_peak("2026-01-05", 850.0, 3)
    assert j.update_window_peak("2026-06-30", 850.0, 3) == 1_000.0


# ======================================================= empates en el pico
def test_empate_el_pico_repetido_sigue_en_la_ventana(tmp_path):
    j = Journal(tmp_path / "j.db")
    for i, v in enumerate([1_000.0, 900.0, 1_000.0]):
        j.update_window_peak(sess(i), v, 3)
    assert j.update_window_peak(sess(3), 850.0, 3) == 1_000.0   # sale la 0, queda la 2
    assert j.update_window_peak(sess(4), 850.0, 3) == 1_000.0
    assert j.update_window_peak(sess(5), 850.0, 3) == 850.0


# ================================================ suelta y caída rápida
def test_suelta_tras_recuperacion_parcial_dentro_del_umbral(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, W)
    assert halted(j.update_window_peak(sess(1), 850.0, W), 850.0)
    peak = j.update_window_peak(sess(2), 905.0, W)            # 9,5 % bajo el pico
    assert peak == 1_000.0 and not halted(peak, 905.0)


def test_sigue_frenado_si_la_caida_continua_rapido(tmp_path):
    """W = 5 y caída del 3 % por sesión: 1 - 0,97**4 = 11,5 % > 10 %, sigue frenado."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 5)
    v = 850.0
    for i in range(1, 30):
        peak = j.update_window_peak(sess(i), v, 5)
        assert halted(peak, v), i
        v *= 0.97


# ============================ defectos de diseño M1 y M2 (xfail documentados)
@pytest.mark.xfail(strict=True,
                   reason="R17 M1 (diseño): una caída repartida en más de W sesiones "
                          "nunca pone el freno. La especificación Lean lo admite (L2 lo "
                          "demostró en stair_never_halts, VelveteenDrawdownGaps.lean, que "
                          "no está en el repositorio). Lo cierra R32 o una decisión del mantenedor.")
def test_m1_una_caida_lenta_acumulada_deberia_poner_el_freno(tmp_path):
    """Lo deseado: con W = 126, θ = 10 % y −0,08 % por sesión, la pérdida
    acumulada pasa del 38 % en 600 sesiones; un freno que acote la pérdida se
    pondría al menos una vez. Hoy la caída dentro de la ventana es
    1 − 0,9992**125 = 9,5 % < 10 % y el freno no se pone nunca."""
    j = Journal(tmp_path / "j.db")
    v, frenos = 1_000.0, 0
    for i in range(600):
        peak = j.update_window_peak(sess(i), v, W)
        frenos += halted(peak, v)
        v *= 0.9992
    assert 1 - v / 1_000.0 > 0.38
    assert frenos > 0


@pytest.mark.xfail(strict=True,
                   reason="R17 M2 (diseño): el freno se suelta en plena caída cuando "
                          "el pico viejo sale de la ventana. La especificación Lean lo "
                          "admite (window_halt_releases). Lo cierra R32 o una decisión del mantenedor.")
def test_m2_no_deberia_soltarse_en_plena_caida(tmp_path):
    """Lo deseado: con W = 5, un choque del 15 % y luego −2 % por sesión, el
    valor nunca vuelve al 90 % del máximo inicial, así que el freno seguiría
    puesto. Hoy se suelta en la quinta sesión (1 − 0,98**4 = 7,76 % < 10 %) y
    en la sesión 11 el valor es 694,51 sin freno."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 5)
    v, estados = 850.0, []
    for i in range(1, 12):
        peak = j.update_window_peak(sess(i), v, 5)
        estados.append(halted(peak, v))
        v *= 0.98
    assert estados == [True] * 11
