"""m3: la suite no depende de ALPACA_PAPER ni de SATELLITE_TOTAL_CAPITAL del
shell o del .env (ver conftest.py). Estos tests fallan si el aislamiento se
rompe; la prueba de fondo es correr la suite con las variables exportadas."""
import os

import pytest

from velveteentrade.config import Settings, load_settings


def test_el_entorno_de_la_suite_es_paper_sin_total_externo():
    assert os.environ["ALPACA_PAPER"] == "true"
    assert os.environ["SATELLITE_TOTAL_CAPITAL"] == ""
    assert Settings().alpaca_paper is True                 # el valor por defecto se lee al importar


def test_load_settings_usa_el_total_de_config_yaml():
    s = load_settings()
    assert s.allocation.total_capital == pytest.approx(10_000.0)
    assert s.allocation.satellite_capital(paper=True) == pytest.approx(2_000.0)


def test_un_test_puede_cambiar_las_variables_y_no_contamina_al_siguiente(monkeypatch):
    monkeypatch.setenv("SATELLITE_TOTAL_CAPITAL", "15000")
    assert load_settings().allocation.total_capital == pytest.approx(15_000.0)


def test_el_siguiente_test_vuelve_a_ver_el_entorno_fijo():
    assert os.environ["SATELLITE_TOTAL_CAPITAL"] == ""
    assert load_settings().allocation.total_capital == pytest.approx(10_000.0)
