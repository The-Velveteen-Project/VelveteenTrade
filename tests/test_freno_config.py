"""El umbral del freno del satélite sale de config.yaml aunque exista un perfil.

Decisión del 14 nov 2026: el satélite frena al 10 % de config.yaml. Antes, con
un perfil de tolerancia efectiva 1, la fila del perfil lo bajaba al 5 % (M3 de
R17). El perfil sigue fijando los demás límites (tope por posición, número de
posiciones, convicción mínima...). Todos los perfiles de este archivo son
sintéticos.

Los cuatro puntos que recalculan límites usan profile.gate_limits:
apply_profile (cli.main al arrancar), pipeline._run_cycle, cli profile y
advisor.explain_plan. Un test por punto.
"""
from __future__ import annotations

from datetime import date

import pytest

from velveteentrade import advisor, cli
from velveteentrade import profile as pm
from velveteentrade.config import Allocation, RiskLimits, Settings, load_settings

from test_brechas import MockBroker, buy_all, make_settings, run_cycle, save_profile, snap

CFG_HALT = 0.10
SEP = dict(mode="separate_account", total_capital=10_000, satellite_pct_paper=0.20,
           satellite_pct_live=0.10)


def perfil_t1(path):
    """Perfil sintético de tolerancia 1: su fila de la tabla dice freno del 5 %."""
    prof = save_profile(path, tolerance=1, max_drawdown_comfort=0.05)
    assert pm.effective_tolerance(prof) == 1
    return prof


def settings_sep(tmp_path, name="s", universe=("X",)):
    s = make_settings(tmp_path, name, list(universe), limits=RiskLimits(max_drawdown_halt_pct=CFG_HALT))
    s.allocation = Allocation(**SEP)
    s.alpaca_paper = True
    return s


# ------------------------------------------------------------ profile.py
def test_la_fila_de_tolerancia_1_dice_5_pero_la_puerta_usa_10(tmp_path):
    prof = perfil_t1(tmp_path / "p.yaml")
    assert pm.limits_for(prof).max_drawdown_halt_pct == pytest.approx(0.05)   # la tabla no cambia
    lim = pm.gate_limits(prof, RiskLimits(max_drawdown_halt_pct=CFG_HALT, drawdown_window_sessions=63))
    assert lim.max_drawdown_halt_pct == pytest.approx(CFG_HALT)
    assert lim.drawdown_window_sessions == 63
    # El perfil sigue fijando los demás límites.
    fila = pm.limits_for(prof)
    for campo in ("max_position_pct", "max_positions", "per_trade_risk_pct", "min_conviction",
                  "max_sector_exposure_pct", "max_daily_turnover_pct"):
        assert getattr(lim, campo) == getattr(fila, campo), campo
    assert lim.max_position_pct == pytest.approx(0.05) and lim.min_conviction == 4


def test_sin_perfil_rigen_los_limites_de_config():
    cfg = RiskLimits(max_drawdown_halt_pct=CFG_HALT)
    assert pm.gate_limits(None, cfg) is cfg


def test_el_umbral_de_config_manda_con_cualquier_tolerancia(tmp_path):
    """Con 0,07 en config, ningún perfil (tolerancia 1 a 5) lo cambia."""
    cfg = RiskLimits(max_drawdown_halt_pct=0.07)
    for t in range(1, 6):
        prof = save_profile(tmp_path / f"p{t}.yaml", tolerance=t)
        assert pm.gate_limits(prof, cfg).max_drawdown_halt_pct == pytest.approx(0.07), t


def test_config_owned_son_ventana_y_umbral():
    assert set(pm.CONFIG_OWNED) == {"drawdown_window_sessions", "max_drawdown_halt_pct"}


def test_apply_profile_conserva_el_umbral_de_config_y_es_idempotente(tmp_path):
    """cli.main llama a apply_profile al arrancar; luego pipeline, cli profile y
    advisor leen settings.risk. El umbral debe seguir siendo el de config."""
    s = Settings(risk=RiskLimits(max_drawdown_halt_pct=CFG_HALT))
    s.profile_path = tmp_path / "profile.yaml"
    prof = perfil_t1(s.profile_path)
    pm.apply_profile(s)
    assert s.risk.max_drawdown_halt_pct == pytest.approx(CFG_HALT)
    assert s.risk.max_position_pct == pm.limits_for(prof).max_position_pct      # 5 %, del perfil
    pm.apply_profile(s)
    assert s.risk.max_drawdown_halt_pct == pytest.approx(CFG_HALT)


def test_config_yaml_del_repo_trae_10_por_ciento():
    assert load_settings().risk.max_drawdown_halt_pct == pytest.approx(0.10)


# ------------------------------------------------------------- pipeline
def _ciclos(tmp_path, monkeypatch, con_perfil):
    """Tres ciclos sobre la misma base: S = $2.000 (20 % de 10.000).
    Ciclo 1: V = 2000 (pico). Ciclo 2: la cuenta pierde $140, V = 1860 (−7 %).
    Ciclo 3: la cuenta pierde $220, V = 1780 (−11 %)."""
    s = settings_sep(tmp_path)
    if con_perfil:
        perfil_t1(s.profile_path)
        pm.apply_profile(s)                      # como cli.main al arrancar
    broker = MockBroker(equity=100_000.0)
    out = []
    j = None
    for dia, equity in ((16, 100_000.0), (17, 99_860.0), (18, 99_780.0)):
        broker.state.equity = broker.state.cash = equity
        broker.session = date(2026, 11, dia)
        n0 = len(broker.orders)
        summary, j = run_cycle(s, broker, buy_all(conv=5), monkeypatch,
                               {"X": snap("X", close=50.0, atr=1.0)}, journal=j)
        out.append((broker.orders[n0:], summary["rejected"]))
    return out


@pytest.mark.parametrize("con_perfil", [True, False], ids=["perfil_t1", "sin_perfil"])
def test_pipeline_frena_al_10_y_no_al_5(tmp_path, monkeypatch, con_perfil):
    (o1, r1), (o2, r2), (o3, r3) = _ciclos(tmp_path, monkeypatch, con_perfil)
    assert o1 and not r1
    # −7 %: con el 5 % del perfil esta compra se rechazaba.
    assert o2 and not any("drawdown" in r for r in r2), r2
    # −11 %: pasa del 10 % de config.yaml y el freno se pone.
    assert not o3
    assert any("drawdown beyond 10%" in r for r in r3), r3


def test_pipeline_con_perfil_sigue_usando_el_tope_del_perfil(tmp_path, monkeypatch):
    (o1, _), _, _ = _ciclos(tmp_path, monkeypatch, con_perfil=True)
    (sym, qty, side), = o1
    assert side == "buy" and qty * 50.0 <= 0.05 * 2_000.0 + 1e-9                 # 5 % del perfil


# ------------------------------------------------------- cli y advisor
def test_cli_profile_muestra_el_freno_de_config(tmp_path, capsys):
    s = settings_sep(tmp_path)
    perfil_t1(s.profile_path)
    pm.apply_profile(s)
    cli.cmd_profile(s)
    out = capsys.readouterr().out
    assert "freno drawdown 10%" in out
    assert "pos máx 5%" in out                                                   # del perfil


def test_advisor_explica_el_freno_de_config(tmp_path, monkeypatch):
    s = settings_sep(tmp_path)
    prof = perfil_t1(s.profile_path)

    class SinCanon:
        def __init__(self, *a, **k):
            pass

        def excerpts(self, *a, **k):
            return []

    class Eco:
        payload = ""

        def chat(self, model, system, messages):
            Eco.payload = messages[0]["content"]
            return "ok"

    monkeypatch.setattr(advisor, "Retriever", SinCanon)
    advisor.explain_plan(Eco(), "m", prof, s)
    assert "cae 10% desde su máximo" in Eco.payload
    assert "máx 5% por posición" in Eco.payload
