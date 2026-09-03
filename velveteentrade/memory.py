"""Decision journal (SQLite) — audit trail + raw material for post-trade reflection."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import JournalEntry

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
        self._conn.commit()

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
        """Persist the all-time peak equity; returns the current peak."""
        peak = max(self.get_float("peak_equity", equity) or equity, equity)
        self.set_float("peak_equity", peak)
        return peak

    def reconcile_entries(self, positions: dict) -> None:
        """Sync journal entry prices with the broker's actual fills.

        The broker's avg_entry is the source of truth — it corrects the
        provisional last-close price recorded at order time, and clears
        entries for symbols no longer held (manual closes, full stops).
        """
        held = set(positions)
        for symbol in list(self.entry_prices()):
            if symbol not in held:
                self.clear_entry(symbol)
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
