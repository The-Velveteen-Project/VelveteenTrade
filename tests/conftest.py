"""Test isolation: no test may read or write the real journal, profile or cache.

config.DATA_DIR is resolved when velveteentrade.config is first imported, and
Settings() defaults its db/profile/cache paths from it. Point it at a fresh
temporary directory BEFORE anything imports the package (pytest imports this
file first). Without this, tests that build Settings() without overriding
profile_path read the deployment's real profile.yaml (found 24 Oct 2026).

m3 (revisión del 14 nov 2026): la suite tampoco depende del .env ni del shell.
config.py llama a load_dotenv, que no pisa variables ya definidas; Settings lee
ALPACA_PAPER al importarse y load_settings lee SATELLITE_TOTAL_CAPITAL. Con
ALPACA_PAPER=false o SATELLITE_TOTAL_CAPITAL=20000 exportadas, varios tests
fallaban sin que hubiera regresión. Se fijan aquí, antes de importar el paquete,
y otra vez en cada test, porque un test puede cambiarlas con monkeypatch. Un
SATELLITE_TOTAL_CAPITAL vacío vale como ausente (config.load_settings).
"""
import os
import tempfile

import pytest

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="velveteen-tests-")

TEST_ENV = {"ALPACA_PAPER": "true", "SATELLITE_TOTAL_CAPITAL": ""}
os.environ.update(TEST_ENV)


@pytest.fixture(autouse=True)
def _entorno_fijo(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
