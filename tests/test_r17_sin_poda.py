"""R17 m1: session_peaks ya no se poda por la W vigente.

Antes, update_window_peak borraba las sesiones fuera de la ventana actual.
Subir W no recuperaba esas sesiones (el pico quedaba por debajo del máximo de
las últimas W sesiones) y bajar W un solo ciclo perdía el pico para siempre.
Ahora solo se borra lo que excede WINDOW_MAX sesiones, que ninguna W válida
puede mirar, así que cambiar W dentro de las cotas es reversible.
"""
from __future__ import annotations

from velveteentrade.config import WINDOW_MAX
from velveteentrade.memory import Journal

from test_r31_freno_ventana import halted, sess


def _filas(j: Journal) -> int:
    return j._conn.execute("SELECT COUNT(*) FROM session_peaks").fetchone()[0]


def test_subir_w_recupera_las_sesiones_anteriores(tmp_path):
    """Caso de L1 (test_hallazgo_subir_w_no_recupera_las_sesiones_podadas), ahora
    con el resultado que pide la especificación: W = 3 y luego W = 10 da el
    máximo de las últimas 10 sesiones, 1 000, y el freno queda puesto."""
    j = Journal(tmp_path / "j.db")
    for i, v in enumerate([1_000.0, 850.0, 850.0, 850.0]):
        j.update_window_peak(sess(i), v, 3)
    assert j.update_window_peak(sess(3), 850.0, 3) == 850.0      # con W = 3 el 1 000 salió
    peak = j.update_window_peak(sess(4), 850.0, 10)               # ahora W = 10
    assert peak == 1_000.0 and halted(peak, 850.0)


def test_bajar_w_un_ciclo_y_volver_no_pierde_el_pico(tmp_path):
    """Caso de L4: un ciclo con W = 20 y luego volver a 126 bajaba el pico de
    1 000 a 900 de forma irreversible. Ahora vuelve a 1 000."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, 126)
    for i in range(1, 101):
        j.update_window_peak(sess(i), 900.0, 126)
    assert j.update_window_peak(sess(100), 900.0, 126) == 1_000.0
    assert j.update_window_peak(sess(101), 900.0, 20) == 900.0    # W = 20: el 1 000 no cuenta
    assert j.update_window_peak(sess(102), 900.0, 126) == 1_000.0  # W = 126 otra vez
    assert _filas(j) == 103


def test_la_tabla_se_acota_a_window_max_sesiones(tmp_path):
    """La poda por tamaño deja exactamente las WINDOW_MAX sesiones más recientes,
    y con W = WINDOW_MAX el pico más viejo sale de la ventana cuando corresponde."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak(sess(0), 1_000.0, WINDOW_MAX)
    for i in range(1, WINDOW_MAX):
        j.update_window_peak(sess(i), 850.0, 2)
    assert _filas(j) == WINDOW_MAX
    assert j.update_window_peak(sess(WINDOW_MAX - 1), 850.0, WINDOW_MAX) == 1_000.0
    assert j.update_window_peak(sess(WINDOW_MAX), 850.0, WINDOW_MAX) == 850.0
    assert _filas(j) == WINDOW_MAX
    primera = j._conn.execute("SELECT MIN(session) FROM session_peaks").fetchone()[0]
    assert primera == sess(1)


def test_la_poda_no_borra_sesiones_posteriores_a_la_actual(tmp_path):
    """La poda mira las sesiones <= `session`; una fila con etiqueta posterior
    (m8, semántica sin decidir) no se borra por la poda."""
    j = Journal(tmp_path / "j.db")
    j.update_window_peak("2026-03-06", 1_000.0, 2)
    j.update_window_peak("2026-03-05", 850.0, 2)
    assert _filas(j) == 2
