"""Capital base of the satellite the engine manages (B6, B7, B8 — 24 Oct 2026).

One number used to do three jobs — `min(account equity, declared capital)` —
and each job needed a different one:

  S  ALLOCATED satellite: (1 - etf_core_weight) x declared capital, or
     x account equity when no amount was declared. The ETF core is the
     user's; the swing engine never trades it (B8: the plan promised a 60/40
     split while the gate sized against 100% of the capital).
  V  VALUE of the satellite today: S + (account equity - equity when the
     base was set). Every gain and every LOSS of the account is the
     satellite's (the engine already treats every position as its own).
     The public track record and the drawdown halt read V (B6b: they used
     to read the declared capital, so losses were invisible).
  E  SIZING base: max(0, min(V, S, account equity)). Never more than the
     user allocated, less after losses, never more than the account has.
     A declared capital of 0 means "nothing to invest yet", not "the whole
     account" (B6a).

The base (S and the account equity it started from) is re-set whenever the
profile's capital or core weight changes, and the drawdown peak restarts in
the new base (B6c: a peak from the full account compared against a $200
capital halted buys forever). Within a base the peak is the maximum of the
last N sessions, not the all-time one (R31, Journal.update_window_peak).

R20 (14 nov 2026): with `allocation.mode: separate_account` in config.yaml the
ETF core lives in ANOTHER broker account, so this account is 100 % satellite:
core weight 0 and declared capital = total_capital x satellite pct (20 % in
paper, 10 % with real money). The profile still sets the limits, not the split.
In mode "profile" nothing changes.

Known limit: deposits and withdrawals in the broker account are counted as
P&L of the satellite. On a paper account there are none; for real money the
broker's cash-transfer activity should be subtracted.

Posiciones heredadas (C1 de la revisión del 14 nov 2026). V cuenta como del
satélite toda ganancia o pérdida de la cuenta desde que se abrió la base. Si al
abrir una base nueva la cuenta tiene posiciones compradas con otro presupuesto
(por ejemplo, $10.000 por compra cuando el motor operaba toda la cuenta), el
satélite de $2.000 hereda una cartera diez veces mayor: un −0,65 % de la cuenta
le resta un 32,5 % a V y el freno queda puesto meses. Por eso resolve no abre
una base nueva a ciegas. La pospone mientras:

  - las posiciones abiertas valgan más que S, o
  - haya órdenes de compra pendientes (se llenarían ya dentro de la base nueva).

Solo aplica con allocation.mode = separate_account, donde la cuenta es 100 %
satélite; en modo "profile" el núcleo vive en esta misma cuenta y nada cambia.
Mientras está pospuesta, la base anterior sigue guardada sin cambios, E = 0 y
`blocked` lleva el motivo: el pipeline rechaza toda compra con ese mensaje y no
registra V ni el pico; los stops y las ventas siguen. /estado lo muestra. La
salida la decide el usuario: cerrar esas posiciones a mano (o cancelar las
órdenes) o reiniciar la cuenta paper. El primer ciclo con la cuenta en regla
abre la base limpia. Posiciones que valen S o menos no bloquean: pasan a ser
del satélite, como antes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Allocation
from .memory import Journal
from .profile import RiskProfile, etf_core_weight

_KEY = "capital_base"
_BLOCK_KEY = "capital_base_blocked"   # re-base pospuesto (C1); "" = ninguno


@dataclass(frozen=True)
class CapitalBase:
    allocated: float      # S
    value: float          # V
    sizing: float         # E
    core_weight: float
    rebased: bool         # True if this call set a new base (peak restarts)
    blocked: str | None = None   # C1: base nueva pospuesta; motivo y qué hacer. E = 0.


def _fingerprint(profile: RiskProfile | None, allocation: Allocation | None = None,
                 paper: bool = True) -> dict:
    sat = allocation.satellite_capital(paper) if allocation is not None else None
    if sat is not None:
        # R20: núcleo en otra cuenta. La huella guarda también si la cuenta es
        # paper o real: el satélite puede valer lo mismo en los dos modos
        # (10 000 x 20 % = 20 000 x 10 %) y aun así son cuentas distintas, con
        # otra equity de partida. Cambiar de modo, de producto o de paper a
        # real abre una base nueva y vacía la ventana (B6c).
        return {"capital": float(sat), "core": 0.0, "paper": bool(paper)}
    if profile is None:
        return {"capital": None, "core": 0.0}
    cap = None if profile.capital is None else float(profile.capital)
    return {"capital": cap, "core": float(etf_core_weight(profile))}


def working_capital(allocation: Allocation | None, profile: RiskProfile | None,
                    paper: bool = True) -> float | None:
    """Capital de trabajo que muestran /estado, /informe, el chat y el track
    record. En separate_account es el satélite que de verdad opera el motor
    (total x pct); el capital del perfil ya no dimensiona nada y mostrarlo
    contradecía la cifra del satélite. En modo "profile", el capital declarado."""
    s = allocation.satellite_capital(paper) if allocation is not None else None
    if s is not None:
        return s
    return profile.capital if profile else None


def current_base(journal: Journal) -> dict | None:
    """The base in force: {'allocated', 'start_equity', 'since', 'fp'}, or None."""
    stored = journal.get_str(_KEY)
    return json.loads(stored) if stored else None


def base_is_current(journal: Journal, profile: RiskProfile | None,
                    allocation: Allocation | None = None, paper: bool = True) -> bool:
    """True si la base guardada corresponde a la configuración en vigor. Tras
    un despliegue que cambia la huella, la base vieja sigue guardada hasta que
    el próximo ciclo llama a resolve; mientras tanto no se debe mostrar."""
    base = current_base(journal)
    return base is not None and base.get("fp") == _fingerprint(profile, allocation, paper)


def pnl_since_base(journal: Journal) -> tuple[float, float, str | None] | None:
    """(start value, value now, since) of the equity series IN THE CURRENT BASE.

    The series records the satellite's value; when the profile changes the base
    the series jumps, and a P&L measured across that jump would mix two bases."""
    hist = journal.equity_history()
    if not hist:
        return None
    base = current_base(journal)
    since = base.get("since") if base else None
    seg = [e for ts, e in hist if since is None or ts >= since] or [hist[-1][1]]
    return seg[0], seg[-1], since


def split(profile: RiskProfile | None, allocation: Allocation | None = None,
          paper: bool = True) -> dict:
    """Reparto en vigor, para mostrarlo: {'mode', 'core', 'capital'}."""
    fp = _fingerprint(profile, allocation, paper)
    mode = allocation.mode if allocation is not None else "profile"
    return {"mode": mode, "core": fp["core"], "capital": fp["capital"]}


def rebase_blocked(journal: Journal) -> dict | None:
    """El re-base pospuesto del último ciclo (C1), o None:
    {'allocated', 'positions_value', 'pending_buys', 'since', 'message'}."""
    stored = journal.get_str(_BLOCK_KEY)
    return json.loads(stored) if stored else None


def blocked_message(allocated: float, positions_value: float, pending_buys: int) -> str:
    """Motivo y salida, en una línea, para el log, el resumen del ciclo y /estado."""
    why = []
    if positions_value > allocated:
        why.append(f"la cuenta tiene posiciones abiertas por ${positions_value:,.2f}, más que "
                   f"el satélite (${allocated:,.2f})")
    if pending_buys:
        why.append(f"hay {pending_buys} orden(es) de compra pendiente(s) de antes")
    return ("compras bloqueadas: " + " y ".join(why) + ". El satélite no abre su base sobre "
            "posiciones que no compró con su presupuesto; los stops y las ventas siguen. "
            "Qué hacer: cerrar esas posiciones a mano (y cancelar las órdenes pendientes), "
            "o reiniciar la cuenta paper. El ciclo siguiente abre la base limpia.")


def resolve(journal: Journal, profile: RiskProfile | None, account_equity: float,
            allocation: Allocation | None = None, paper: bool = True,
            positions_value: float = 0.0, pending_buys: int = 0) -> CapitalBase:
    """S, V y E de la base en vigor; abre una base nueva si la huella cambió.
    `positions_value` (suma de |valor de mercado| de las posiciones abiertas) y
    `pending_buys` (órdenes de compra pendientes) solo cuentan al abrir una base
    nueva: si las posiciones valen más que S o hay compras pendientes, la base
    nueva se pospone (C1; ver el docstring del módulo)."""
    fp = _fingerprint(profile, allocation, paper)
    stored = journal.get_str(_KEY)
    base = json.loads(stored) if stored else None
    rebased = base is None or base.get("fp") != fp
    if rebased:
        start = float(account_equity)
        gross = start if fp["capital"] is None else min(max(fp["capital"], 0.0), start)
        allocated = (1.0 - fp["core"]) * gross
        held = float(positions_value)
        # Solo en separate_account, donde la cuenta es 100 % satélite. En modo
        # "profile" el núcleo de ETF vive en esta misma cuenta y sus posiciones
        # valen más que S por diseño: ahí el guardia bloquearía para siempre.
        guard = fp.get("paper") is not None
        if guard and (held > allocated or pending_buys > 0 or held != held):   # NaN: falla cerrado
            msg = blocked_message(allocated, held, pending_buys)
            prev = rebase_blocked(journal) or {}
            journal.set_str(_BLOCK_KEY, json.dumps({
                "fp": fp, "allocated": allocated, "positions_value": held,
                "pending_buys": int(pending_buys), "message": msg,
                "since": prev.get("since") or datetime.now(timezone.utc).isoformat()}))
            return CapitalBase(allocated=allocated, value=allocated, sizing=0.0,
                               core_weight=fp["core"], rebased=False, blocked=msg)
        base = {"fp": fp, "start_equity": start, "allocated": allocated,
                "since": datetime.now(timezone.utc).isoformat()}
        journal.set_str(_KEY, json.dumps(base))
        legacy = stored is None and fp == {"capital": None, "core": 0.0}
        if not legacy:
            # New base, new peak. (The one exception: first run of this code
            # with no profile, where the old peak is in the same base.)
            # R31: también se vacía la ventana del pico (sesiones de la base
            # anterior no se comparan con la nueva).
            journal.reset_window_peak(allocated)
    if journal.get_str(_BLOCK_KEY):
        journal.set_str(_BLOCK_KEY, "")         # la base está en regla: sin bloqueo
    allocated = float(base["allocated"])
    value = allocated + (float(account_equity) - float(base["start_equity"]))
    sizing = max(0.0, min(value, allocated, float(account_equity)))
    return CapitalBase(allocated=allocated, value=value, sizing=sizing,
                       core_weight=fp["core"], rebased=rebased)
