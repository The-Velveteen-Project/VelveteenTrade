"""R17 m2 y m3: cotas de la ventana W del freno por drawdown (R31).

Antes, check_window aceptaba cualquier entero >= 1:
- m2: un W enorme (10**19) pasaba la carga y cada ciclo lanzaba OverflowError
  en update_window_peak, antes del barrido de stops;
- m3: W = 1 dejaba el pico en el máximo de la sesión actual y, con un ciclo
  diario, ninguna caída entre sesiones ponía el freno.
Ahora W debe estar en [WINDOW_MIN, WINDOW_MAX] = [2, 2520] y un valor fuera
del rango detiene el arranque (load_settings), no un ciclo.
"""
from __future__ import annotations

import pytest

from velveteentrade import profile as profile_mod
from velveteentrade.config import WINDOW_MAX, WINDOW_MIN, RiskLimits, load_settings
from velveteentrade.memory import Journal

from test_r31_freno_ventana import halted, sess


def _yaml(tmp_path, raw: str):
    p = tmp_path / "c.yaml"
    p.write_text(f"risk:\n  drawdown_window_sessions: {raw}\n")
    return p


def test_cotas_documentadas():
    assert (WINDOW_MIN, WINDOW_MAX) == (2, 2520)
    assert WINDOW_MAX < 2**63 - 1                     # sqlite3 liga cualquier W válido


@pytest.mark.parametrize("raw", ["0", "-1", "1", "2521", "10000000000000000000",
                                 "126.0", "'126'", "true", "null", "[126]"])
def test_w_fuera_de_rango_o_mal_tipado_detiene_la_carga(tmp_path, raw):
    """m2 y m3: 1, 2521 y 10**19 ahora fallan al cargar, igual que los
    valores mal tipados que ya fallaban (L1, test_w_invalido_en_yaml)."""
    with pytest.raises(ValueError):
        load_settings(_yaml(tmp_path, raw))


@pytest.mark.parametrize("w", [WINDOW_MIN, 126, WINDOW_MAX])
def test_w_en_el_rango_carga_y_corre_un_ciclo(tmp_path, w):
    s = load_settings(_yaml(tmp_path, str(w)))
    assert s.risk.drawdown_window_sessions == w
    j = Journal(tmp_path / "j.db")
    assert j.update_window_peak(sess(0), 1_000.0, w) == 1_000.0
    assert j.update_window_peak(sess(1), 850.0, w) == 1_000.0


def test_w_enorme_ya_no_llega_al_ciclo(tmp_path):
    """m2: el OverflowError de sqlite3 ya no es alcanzable desde la configuración,
    ni por config.yaml ni por limits_for (perfil)."""
    with pytest.raises(ValueError):
        load_settings(_yaml(tmp_path, "10000000000000000000"))
    with pytest.raises(ValueError):
        RiskLimits(drawdown_window_sessions=10**19)
    with pytest.raises(ValueError):
        Journal(tmp_path / "j.db").update_window_peak(sess(0), 1.0, 10**19)


def test_limits_for_respeta_las_cotas():
    prof = profile_mod.RiskProfile(name="U", tolerance=4, horizon_years=10,
                                   experience="INTERMEDIATE", max_drawdown_comfort=0.20,
                                   income_stability="STABLE", has_emergency_fund=True,
                                   capital=1_000.0)
    assert profile_mod.limits_for(prof, window=WINDOW_MAX).drawdown_window_sessions == WINDOW_MAX
    for bad in (1, WINDOW_MAX + 1):
        with pytest.raises(ValueError):
            profile_mod.limits_for(prof, window=bad)


def test_w_minimo_frena_una_caida_entre_sesiones(tmp_path):
    """m3: con W = 1 una caída del 50 % entre dos sesiones no frenaba (L1,
    test_w1_una_caida_entre_sesiones_nunca_frena). Con la ventana mínima sí."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, WINDOW_MIN)
    peak = j.update_window_peak(sess(1), 500.0, WINDOW_MIN)
    assert peak == 1_000.0 and halted(peak, 500.0)
