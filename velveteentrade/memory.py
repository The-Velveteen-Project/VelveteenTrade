"""Decision journal (SQLite) — audit trail + raw material for post-trade reflection."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .schemas import JournalEntry

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload TEXT NOT NULL,
    executed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS entries (
    symbol TEXT PRIMARY KEY,
    entry_price REAL NOT NULL,
    entry_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stops (
    symbol TEXT PRIMARY KEY,
    stop_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    lesson TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_history (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS session_peaks (
    session TEXT PRIMARY KEY,
    peak REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turnover (
    session TEXT NOT NULL,
    order_id TEXT NOT NULL,
    notional REAL NOT NULL,
    PRIMARY KEY (session, order_id)
);
"""


class Journal:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)

    def record(self, entry: JournalEntry) -> None:
        self._conn.execute(
            "INSERT INTO journal (ts, symbol, payload, executed) VALUES (?, ?, ?, ?)",
            (entry.ts.isoformat(), entry.symbol, entry.model_dump_json(), int(entry.executed)),
        )
        self._conn.commit()

    def set_entry_price(self, symbol: str, price: float) -> None:
        self._conn.execute(
            "INSERT INTO entries (symbol, entry_price, entry_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET entry_price=excluded.entry_price, entry_ts=excluded.entry_ts",
            (symbol, price, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def clear_entry(self, symbol: str) -> None:
        self._conn.execute("DELETE FROM entries WHERE symbol = ?", (symbol,))
        self._conn.execute("DELETE FROM stops WHERE symbol = ?", (symbol,))
        self._conn.commit()

    def set_stop(self, symbol: str, stop_price: float) -> float:
        """Freeze the stop computed by the risk gate at buy time (see risk.check_stops).

        TIGHTEN-ONLY (B4): there is one stop per symbol, shared by every lot.
        An add-on buy computes its stop from today's price and ATR, which can
        be far LOWER than the stop the first lot was sized with; overwriting
        it broke G2 for that lot (1 500 lost on a 1 000 budget). Keeping the
        max of old and new bounds the loss of EVERY lot by its own budget:
        a higher stop only shortens each lot's distance to it (model P4).
        Returns the stop in force after the call."""
        self._conn.execute(
            "INSERT INTO stops (symbol, stop_price) VALUES (?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET stop_price=MAX(stops.stop_price, excluded.stop_price)",
            (symbol, float(stop_price)),
        )
        self._conn.commit()
        return self.stop_prices()[symbol]

    def stop_prices(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT symbol, stop_price FROM stops").fetchall()
        return {s: p for s, p in rows}

    def entry_prices(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT symbol, entry_price FROM entries").fetchall()
        return {s: p for s, p in rows}

    def add_lesson(self, symbol: str, lesson: str) -> None:
        self._conn.execute(
            "INSERT INTO lessons (ts, symbol, lesson) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), symbol, lesson),
        )
        self._conn.commit()

    def lessons(self, symbol: str | None = None, limit: int = 5) -> list[str]:
        if symbol:
            rows = self._conn.execute(
                "SELECT lesson FROM lessons WHERE symbol = ? ORDER BY id DESC LIMIT ?", (symbol, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT lesson FROM lessons ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------- turnover
    def book_turnover(self, session: str, order_id: str, notional: float) -> None:
        """Book an order's notional against the trading SESSION it executes in.

        Keyed by order id so a cycle can tell its own booked orders from
        pending orders it has never seen (placed by hand, or before this
        ledger existed) and count each exactly once.

        A non-finite notional (e.g. a NaN price) is booked as +inf: an
        unknown spend must exhaust the session budget, never vanish from it
        (SQLite would store NaN as NULL)."""
        notional = abs(float(notional))
        if not math.isfinite(notional):
            log.error("Turnover for order %s is not finite (%r) — booking +inf: no more "
                      "turnover this session.", order_id, notional)
            notional = math.inf
        self._conn.execute(
            "INSERT OR REPLACE INTO turnover (session, order_id, notional) VALUES (?, ?, ?)",
            (session, str(order_id), notional),
        )
        self._conn.commit()

    def booked_turnover(self, session: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT order_id, notional FROM turnover WHERE session = ?", (session,)
        ).fetchall()
        return {oid: n for oid, n in rows}

    # ------------------------------------------------------------------- kv
    def get_float(self, key: str, default: float | None = None) -> float | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return float(row[0]) if row else default

    def set_float(self, key: str, value: float) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self._conn.commit()

    def get_str(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_str(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def record_equity(self, equity: float) -> None:
        """One equity point per cycle — the raw material of the public track record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO equity_history (ts, equity) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), float(equity)),
        )
        self._conn.commit()

    def equity_history(self) -> list[tuple[str, float]]:
        return self._conn.execute(
            "SELECT ts, equity FROM equity_history ORDER BY ts"
        ).fetchall()

    def update_peak_equity(self, equity: float) -> float:
        """Persist the all-time peak equity; returns the current peak.

        HISTÓRICO: el freno por drawdown ya no lo usa (R31, ver
        update_window_peak). Se conserva para lectura y compatibilidad."""
        peak = max(self.get_float("peak_equity", equity) or equity, equity)
        self.set_float("peak_equity", peak)
        return peak

    # ------------------------------------------------ R31: pico con ventana
    def update_window_peak(self, session: str, value: float, window: int,
                           since: str | None = None) -> float | None:
        """Pico del freno por drawdown: máximo del valor del satélite en las
        últimas `window` sesiones con registro, incluida `session` (R31).

        Antes el pico era el histórico: tras un stop a efectivo el valor ya no
        puede volver a él y el freno no se soltaba nunca. Con la ventana, el
        pico viejo sale de la ventana a las `window` sesiones y el freno se
        suelta si el satélite no ha caído más que el umbral dentro de ella.

        - Una fila por sesión (`session`, ISO) con el MÁXIMO de los ciclos de
          esa sesión: varios ciclos el mismo día no acortan la ventana.
        - Se cuentan sesiones CON REGISTRO: si el bot no corre unos días, la
          ventana abarca más calendario (conservador: el pico dura más).
        - Un valor no finito no se guarda (SQLite lo volvería NULL); la puerta
          rechaza las compras con un valor no finito (fail-closed, B3).
        - Tabla vacía (primera corrida con este código, o recién re-basado por
          capital.reset_window_peak): se reconstruye desde equity_history de
          la base actual (`since`), agrupando por fecha de Nueva York; si no
          hay historia, se toma el pico guardado en kv `peak_equity`, como
          pico de esta sesión (caduca a las `window` sesiones).
        - `window` mayor que la historia: el máximo de toda la historia de la
          base (el comportamiento de antes, mientras la historia sea corta).
        - No se poda por `window` (R17 m1): se conservan las WINDOW_MAX
          sesiones más recientes, así que subir o bajar W (dentro de las cotas
          de config.check_window) es reversible y no pierde picos.

        Devuelve el pico de la ventana (None si no hay ningún valor finito) y
        lo deja en kv `peak_equity` para que se pueda consultar."""
        from .config import WINDOW_MAX, check_window
        check_window(window)
        empty = self._conn.execute("SELECT COUNT(*) FROM session_peaks").fetchone()[0] == 0
        if empty:
            self._seed_session_peaks(session, since)
        if math.isfinite(value):
            self._conn.execute(
                "INSERT INTO session_peaks (session, peak) VALUES (?, ?) "
                "ON CONFLICT(session) DO UPDATE SET peak=MAX(session_peaks.peak, excluded.peak)",
                (session, float(value)),
            )
        # Poda por tamaño, nunca por `window` (R17 m1): solo se borra lo que
        # queda fuera de las WINDOW_MAX sesiones más recientes hasta `session`,
        # que ninguna W válida puede alcanzar. Antes se podaba lo que quedaba
        # fuera de la W vigente y cambiar W perdía picos de forma irreversible.
        horizon = self._conn.execute(
            "SELECT session FROM session_peaks WHERE session <= ? "
            "ORDER BY session DESC LIMIT 1 OFFSET ?", (session, WINDOW_MAX - 1)).fetchone()
        if horizon is not None:
            self._conn.execute("DELETE FROM session_peaks WHERE session < ?", (horizon[0],))
        # La ventana: las `window` sesiones más recientes hasta `session`.
        keep = [r[0] for r in self._conn.execute(
            "SELECT session FROM session_peaks WHERE session <= ? ORDER BY session DESC LIMIT ?",
            (session, window)).fetchall()]
        row = self._conn.execute(
            "SELECT MAX(peak) FROM session_peaks WHERE session >= ? AND session <= ?",
            (min(keep), session)).fetchone() if keep else (None,)
        self._conn.commit()
        peak = row[0]
        if peak is None:
            return None
        peak = float(peak)
        self.set_float("peak_equity", peak)
        return peak

    def _seed_session_peaks(self, session: str, since: str | None) -> None:
        ny = ZoneInfo("America/New_York")
        per_session: dict[str, float] = {}
        for ts, eq in self.equity_history():
            if since is not None and ts < since:
                continue
            if eq is None or not math.isfinite(eq):
                continue
            try:
                day = datetime.fromisoformat(ts).astimezone(ny).date().isoformat()
            except ValueError:
                continue
            day = min(day, session)   # nunca una sesión posterior a la actual
            per_session[day] = max(per_session.get(day, -math.inf), float(eq))
        if not per_session:
            legacy = self.get_float("peak_equity")
            if legacy is not None and math.isfinite(legacy):
                # Sin historia: el pico guardado vale como pico de ESTA sesión
                # (conservador: retiene el freno `window` sesiones más).
                per_session[session] = legacy
        for day, peak in per_session.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO session_peaks (session, peak) VALUES (?, ?)", (day, peak))

    def reset_window_peak(self, value: float) -> None:
        """Base de capital nueva (B6c): el pico reinicia en `value` y la
        historia de la ventana de la base anterior se descarta."""
        self._conn.execute("DELETE FROM session_peaks")
        self._conn.commit()
        self.set_float("peak_equity", value)

    def reconcile_entries(self, positions: dict, pending_buys: set[str] | frozenset = frozenset()) -> None:
        """Sync journal entry prices with the broker's actual fills.

        The broker's avg_entry is the source of truth — it corrects the
        provisional last-close price recorded at order time, and clears
        entries for symbols no longer held (manual closes, full stops).

        Frozen stops of symbols that are neither held nor waiting on a queued
        BUY are cleared too: since stops are tighten-only (B4), an orphan stop
        left by a manual close would otherwise be inherited by a later re-buy
        at a lower price and sell it at once. A stop with a queued buy behind
        it survives — that buy is about to become the position it protects.
        """
        held = set(positions)
        for symbol in list(self.entry_prices()):
            if symbol not in held:
                self._conn.execute("DELETE FROM entries WHERE symbol = ?", (symbol,))
        for symbol in list(self.stop_prices()):
            if symbol not in held and symbol not in pending_buys:
                self._conn.execute("DELETE FROM stops WHERE symbol = ?", (symbol,))
        self._conn.commit()
        for symbol, pos in positions.items():
            if pos.avg_entry and pos.avg_entry > 0:
                self.set_entry_price(symbol, float(pos.avg_entry))

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, symbol, payload, executed FROM journal ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"ts": ts, "symbol": sym, "executed": bool(ex), **json.loads(payload)}
            for ts, sym, payload, ex in rows
        ]

    def close(self) -> None:
        self._conn.close()
