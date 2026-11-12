"""R31: el freno por drawdown se mide contra el máximo de una VENTANA de sesiones.

Antes el pico era el histórico: después de un stop a efectivo el valor del
satélite ya no podía volver a él y el freno no se soltaba nunca (exposición 0
para siempre). Ahora el pico es el máximo de las últimas N sesiones con
registro (config.yaml: risk.drawdown_window_sessions, 126 por defecto,
provisional: lo decide Carlos).

Sin red: journal en tmp_path, broker y LLM falsos (ver test_brechas.py).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from velveteentrade import risk
from velveteentrade.broker import AccountState, MockBroker
from velveteentrade.config import RiskLimits, check_window, load_settings
from velveteentrade.memory import Journal
from velveteentrade.schemas import ExecutiveDecision, TechnicalSnapshot

from test_brechas import buy_all, make_settings, run_cycle, save_profile, snap as snap_b

W = 126
HALT = 0.10


def sess(i: int) -> str:
    """Sesión i como fecha ISO (días consecutivos: el journal solo las ordena)."""
    return (date(2026, 1, 1) + timedelta(days=i)).isoformat()


def buy_verdict(peak, value, limits=None):
    lim = limits or RiskLimits(max_drawdown_halt_pct=HALT)
    acct = AccountState(equity=value if math.isfinite(value) else 1_000.0, cash=1_000.0)
    ctx = risk.CycleContext(peak_equity=peak, current_value=value)
    d = ExecutiveDecision(symbol="X", action="BUY", conviction=5, thesis="t",
                          invalidation="i", horizon_days=10)
    s = TechnicalSnapshot(symbol="X", close=10.0, atr_14=0.2, hist_cvar_95_20d=-0.05)
    return risk.evaluate(d, s, acct, lim, ctx)


def halted(peak, value) -> bool:
    v = buy_verdict(peak, value)
    return not v.approved and any("drawdown" in r for r in v.reasons)


# ------------------------------------------------------------- el freno se activa
def test_freno_se_activa_dentro_de_la_ventana(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.update_window_peak(sess(0), 1_000.0, W) == 1_000.0
    peak = j.update_window_peak(sess(1), 850.0, W)          # caída del 15 % > 10 %
    assert peak == 1_000.0
    assert halted(peak, 850.0)
    assert not halted(peak, 950.0)                           # caída del 5 %: compra


# ------------------------------------------- se suelta cuando sale el pico viejo
def test_freno_se_suelta_cuando_la_ventana_deja_atras_el_pico(tmp_path):
    """El caso R31: pico 1 000, stop a efectivo en 850 y valor plano para siempre."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, W)
    for i in range(1, W):                                    # sesiones 1..125: el pico aún está
        peak = j.update_window_peak(sess(i), 850.0, W)
        assert peak == 1_000.0 and halted(peak, 850.0), i
    peak = j.update_window_peak(sess(W), 850.0, W)           # sesiones 1..126: ya no
    assert peak == 850.0
    assert not halted(peak, 850.0)
    assert j.get_float("peak_equity") == 850.0               # kv refleja el pico vigente


def test_contraste_el_pico_historico_no_se_suelta_nunca(tmp_path):
    """Documenta R31 con el método histórico (que el freno ya no usa)."""
    j = Journal(tmp_path / "j.db")
    j.update_peak_equity(1_000.0)
    for _ in range(3 * W):
        peak = j.update_peak_equity(850.0)
    assert peak == 1_000.0 and halted(peak, 850.0)


def test_la_ventana_cuenta_sesiones_no_ciclos(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 3)
    for _ in range(10):                                      # muchos ciclos en la sesión 1
        j.update_window_peak(sess(1), 850.0, 3)
    assert j.update_window_peak(sess(2), 850.0, 3) == 1_000.0   # sesiones 0,1,2
    assert j.update_window_peak(sess(3), 850.0, 3) == 850.0     # sesiones 1,2,3


def test_pico_de_la_sesion_es_el_maximo_de_sus_ciclos(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 900.0, 2)
    j.update_window_peak(sess(0), 1_000.0, 2)
    assert j.update_window_peak(sess(0), 800.0, 2) == 1_000.0   # bajar no borra el máximo
    assert j.update_window_peak(sess(1), 800.0, 2) == 1_000.0


def test_un_pico_nuevo_dentro_de_la_ventana_reinicia_la_cuenta(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 3)
    j.update_window_peak(sess(1), 1_100.0, 3)                # pico nuevo
    j.update_window_peak(sess(2), 900.0, 3)
    j.update_window_peak(sess(3), 900.0, 3)
    assert j.update_window_peak(sess(3), 900.0, 3) == 1_100.0  # sesiones 1,2,3
    assert j.update_window_peak(sess(4), 900.0, 3) == 900.0


# ------------------------------------------------------------------ casos límite
def test_ventana_mas_larga_que_la_historia_usa_toda_la_historia(tmp_path):
    j = Journal(tmp_path / "j.db")
    vals = [1_000.0, 1_200.0, 900.0, 950.0]
    for i, v in enumerate(vals):
        peak = j.update_window_peak(sess(i), v, 10_000)
    assert peak == 1_200.0                                   # = pico histórico de la base
    assert halted(peak, 950.0)


def test_ventana_de_una_sesion_es_el_maximo_de_hoy(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 1)
    assert j.update_window_peak(sess(1), 500.0, 1) == 500.0


def test_valor_nan_no_se_guarda_y_la_puerta_falla_cerrada(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, W)
    peak = j.update_window_peak(sess(1), math.nan, W)
    assert peak == 1_000.0                                   # el NaN no envenena el pico
    assert j.update_window_peak(sess(2), math.inf, W) == 1_000.0
    v = buy_verdict(peak, math.nan)
    assert not v.approved and "cannot be evaluated" in v.reasons[-1]


def test_sin_ningun_valor_finito_no_hay_pico(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.update_window_peak(sess(0), math.nan, W) is None


@pytest.mark.parametrize("peak", [math.nan, math.inf, -math.inf])
def test_pico_no_finito_bloquea_compras(peak):
    v = buy_verdict(peak, 1_000.0)
    assert not v.approved and "cannot be evaluated" in v.reasons[-1]


def test_sin_pico_no_hay_freno():
    assert buy_verdict(None, 1_000.0).approved


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "126", None])
def test_ventana_invalida_se_rechaza(tmp_path, bad):
    with pytest.raises(ValueError):
        check_window(bad)
    with pytest.raises(ValueError):
        Journal(tmp_path / "j.db").update_window_peak(sess(0), 1.0, bad)
    if bad is not None:
        with pytest.raises(ValueError):
            RiskLimits(drawdown_window_sessions=bad)


def test_config_yaml_trae_la_ventana_por_defecto():
    s = load_settings()
    assert s.risk.drawdown_window_sessions == 126 == RiskLimits().drawdown_window_sessions


def test_config_yaml_ventana_configurable(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  drawdown_window_sessions: 63\n")
    assert load_settings(p).risk.drawdown_window_sessions == 63


def test_reset_descarta_la_ventana_de_la_base_anterior(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 100_000.0, W)
    j.reset_window_peak(600.0)                               # base nueva (B6c)
    assert j.update_window_peak(sess(1), 590.0, W) == 600.0
    assert not halted(600.0, 590.0)


def test_migracion_reconstruye_la_ventana_desde_equity_history(tmp_path):
    """Primera corrida con este código: la ventana sale de equity_history, así
    que un pico viejo FUERA de la ventana no retiene el freno."""
    j = Journal(tmp_path / "j.db")
    start = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)   # 10:00 en Nueva York
    rows = [(start, 1_000.0)] + [(start + timedelta(days=d), 850.0) for d in range(1, 6)]
    for ts, eq in rows:
        j._conn.execute("INSERT INTO equity_history (ts, equity) VALUES (?, ?)",
                        (ts.isoformat(), eq))
    j._conn.commit()
    j.set_float("peak_equity", 1_000.0)                      # pico histórico heredado
    today = (start + timedelta(days=6)).date().isoformat()
    assert j.update_window_peak(today, 850.0, 3) == 850.0    # el 1 000 quedó fuera
    j2 = Journal(tmp_path / "k.db")
    for ts, eq in rows:
        j2._conn.execute("INSERT INTO equity_history (ts, equity) VALUES (?, ?)",
                         (ts.isoformat(), eq))
    j2._conn.commit()
    assert j2.update_window_peak(today, 850.0, 10) == 1_000.0  # ventana larga: lo incluye


def test_migracion_sin_historia_conserva_el_pico_guardado(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_float("peak_equity", 120_000.0)
    assert j.update_window_peak(sess(0), 100_000.0, 3) == 120_000.0
    j.update_window_peak(sess(1), 100_000.0, 3)
    j.update_window_peak(sess(2), 100_000.0, 3)
    assert j.update_window_peak(sess(3), 100_000.0, 3) == 100_000.0  # caduca a las 3 sesiones


# --------------------------------------------------------- de punta a punta
def test_ciclo_completo_el_freno_se_suelta_tras_la_ventana(tmp_path, monkeypatch):
    """Pipeline real con broker falso: pérdida del 16,7 % del satélite, freno
    puesto; tras W sesiones con el valor plano, la compra vuelve a salir."""
    w = 3
    s = make_settings(tmp_path, "x", ["X"], limits=RiskLimits(drawdown_window_sessions=w))
    save_profile(s.profile_path, capital=1_000.0)            # satélite 600 (tolerancia 4)
    snaps = {"X": snap_b("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=100_000.0)
    broker.session = date(2026, 3, 2)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps)
    broker.orders.clear()
    broker.state.equity = 100_000.0 - 100.0                  # el satélite pierde 100
    for i in range(1, w):
        broker.session = date(2026, 3, 2) + timedelta(days=i)
        summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
        assert summary["orders"] == [] and "drawdown" in summary["rejected"][0], i
    broker.session = date(2026, 3, 2) + timedelta(days=w)
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert summary["orders"] and not any("drawdown" in r for r in summary["rejected"])
    assert j.get_float("peak_equity") == 500.0
