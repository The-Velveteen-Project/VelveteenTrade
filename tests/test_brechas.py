"""Reproducciones mínimas de violaciones de promesas de VelveteenTrade.

Cada test PASA cuando la violación OCURRE (documenta el bug con números).
Cuando un bug se arregle, su test debe fallar: entonces se invierte la
aserción y queda como test de regresión.

Sin red: broker, datos y LLM son falsos en memoria (ver conftest.py).
Ninguna orden sale de este proceso.
"""
from __future__ import annotations

import json
import math
from datetime import date
from fractions import Fraction

import numpy as np
import pandas as pd

from velveteentrade import indicators, pipeline, profile as profile_mod, risk
from velveteentrade.broker import MockBroker, OpenOrder, Position
from velveteentrade.config import RiskLimits, Settings
from velveteentrade.llm import MockLLM
from velveteentrade.memory import Journal
from velveteentrade.schemas import (AnalystReport, ExecutiveDecision,
                                    TechnicalSnapshot)

EQ = 100_000.0
_REAL_SNAPSHOT = indicators.snapshot


# ----------------------------------------------------------------- fakes
def snap(sym, close=100.0, atr=2.0, mom=0.20, vol=0.20, cvar=-0.10):
    return TechnicalSnapshot(symbol=sym, close=close, atr_14=atr, momentum_63d=mom,
                             realized_vol_21d=vol, hist_cvar_95_20d=cvar)


class FakeData:
    """MarketData sin red. daily_bars devuelve `bars[symbol]` si existe."""

    def __init__(self, sectors=None, bars=None):
        self.sectors, self.bars = sectors or {}, bars or {}

    def daily_bars(self, symbol, days=320):
        return self.bars.get(symbol, pd.DataFrame({"close": [1.0]}))

    def fundamentals(self, symbol):
        s = self.sectors.get(symbol)
        return {"sector": s} if s else {}

    def news(self, symbol, limit=8):
        return []


class ScriptedLLM(MockLLM):
    """LLM falso: `decide(symbol_analizado) -> kwargs` de ExecutiveDecision."""

    def __init__(self, decide):
        super().__init__()
        self.decide = decide

    def complete(self, model, system, user, schema, max_retries=2):
        payload = json.loads(user)
        if schema is ExecutiveDecision:
            kw = dict(symbol=payload["symbol"], action="HOLD", conviction=3,
                      thesis="t", invalidation="i", horizon_days=10)
            kw.update(self.decide(payload["symbol"]))
            return ExecutiveDecision(**kw)
        sym = payload.get("symbol") or payload["snapshot"]["symbol"]
        return AnalystReport(symbol=sym, stance="NEUTRAL", summary="s", data_quality="FULL")


def buy_all(conv=5):
    return ScriptedLLM(lambda s: dict(action="BUY", conviction=conv))


class AfterHoursBroker(MockBroker):
    """Mercado cerrado: toda orden queda en cola (pending) y la cuenta no cambia."""

    def submit_order(self, symbol, qty, side):
        oid = super().submit_order(symbol, qty, side)
        self.pending.append(OpenOrder(id=oid, symbol=symbol, qty=qty, side=side))
        return oid


class FillingBroker(MockBroker):
    """Mercado abierto: las órdenes se llenan al precio de referencia `px`."""

    def __init__(self, equity, px):
        super().__init__(equity=equity)
        self.px = px

    def submit_order(self, symbol, qty, side):
        oid = super().submit_order(symbol, qty, side)
        pos = self.state.positions.get(symbol)
        q = (pos.qty if pos else 0.0) + (qty if side == "buy" else -qty)
        if abs(q) < 1e-12:
            self.state.positions.pop(symbol, None)
        else:
            self.state.positions[symbol] = Position(symbol, q, self.px[symbol], q * self.px[symbol])
        return oid


def make_settings(tmp_path, name, universe, limits=None):
    d = tmp_path / name
    d.mkdir()
    s = Settings(universe=universe, candidates_per_cycle=20, risk=limits or RiskLimits())
    s.db_path, s.cache_dir, s.profile_path = d / "journal.db", d / ".cache", d / "profile.yaml"
    return s


def run_cycle(settings, broker, llm, monkeypatch, snaps=None, journal=None, data=None):
    """pipeline.run_cycle con snapshots inyectados (los indicadores no son lo que se prueba)."""
    snaps = snaps or {}
    monkeypatch.setattr(indicators, "snapshot",
                        lambda symbol, df: snaps[symbol] if symbol in snaps
                        else _REAL_SNAPSHOT(symbol, df))
    j = journal or Journal(settings.db_path)
    return pipeline.run_cycle(settings, broker, data or FakeData(), llm, j), j


def notional_of(orders, px, side="buy"):
    return sum(q * px[s] for s, q, sd in orders if sd == side)


def save_profile(path, **over):
    base = dict(name="U", tolerance=4, horizon_years=10, experience="INTERMEDIATE",
                max_drawdown_comfort=0.20, income_stability="STABLE",
                has_emergency_fund=True, capital=None)
    base.update(over)
    prof = profile_mod.RiskProfile(**base)
    profile_mod.save(prof, path)
    return prof


# ======================================================================
# B1 — El símbolo de la decisión (LLM) y el de la orden (pipeline) no son
#      el mismo: la puerta evalúa decision.symbol; la orden sale con `symbol`.
# ======================================================================
def test_repro_B1a_lowercase_ticker_bypasses_position_cap(tmp_path, monkeypatch):
    held = {"AAPL": Position("AAPL", 100, 100.0, 10_000.0)}   # ya en el tope del 10 %
    snaps = {"AAPL": snap("AAPL", close=100.0, atr=2.0)}

    # Control: el LLM repite bien el ticker -> la puerta no deja comprar más.
    ok = MockBroker(equity=EQ); ok.state.positions = dict(held)
    run_cycle(make_settings(tmp_path, "ctrl", ["AAPL"]), ok, buy_all(), monkeypatch, snaps)
    assert [o for o in ok.orders if o[2] == "buy"] == []

    # Mismo estado, el LLM escribe "aapl": la puerta lo ve como símbolo nuevo.
    bad = MockBroker(equity=EQ); bad.state.positions = dict(held)
    llm = ScriptedLLM(lambda s: dict(symbol=s.lower(), action="BUY", conviction=5))
    run_cycle(make_settings(tmp_path, "bug", ["AAPL"]), bad, llm, monkeypatch, snaps)
    # Regresión B1 (parche 24 oct): "aapl" se canonicaliza a "AAPL" y la puerta
    # aplica el tope del 10 %: no sale ninguna compra.
    assert [o for o in bad.orders if o[2] == "buy"] == []


def test_repro_B1b_foreign_symbol_sell_opens_a_short(tmp_path, monkeypatch):
    broker = MockBroker(equity=EQ)
    broker.state.positions = {"MSFT": Position("MSFT", 100, 300.0, 30_000.0)}  # AAPL: 0 acciones
    llm = ScriptedLLM(lambda s: dict(symbol="MSFT", action="SELL", conviction=3))
    run_cycle(make_settings(tmp_path, "x", ["AAPL"]), broker, llm, monkeypatch,
              {"AAPL": snap("AAPL")})
    # Regresión B1 (parche 24 oct): un símbolo ajeno se rechaza; no sale ninguna orden.
    assert broker.orders == []


# ======================================================================
# B2 — Las órdenes pendientes no entran en la cartera sombra. Varios ciclos
#      con el mercado cerrado (bot: /ciclo repetido, o /ciclo + job 21:30)
#      apilan exposición: es la falla 1 del README, ahora entre ciclos.
# ======================================================================
def test_repro_B2_pending_orders_stack_across_cycles(tmp_path, monkeypatch):
    syms = [f"S{i}" for i in range(12)]
    snaps = {s: snap(s, mom=0.5 - i / 100) for i, s in enumerate(syms)}  # orden de screen fijo
    px = {s: 100.0 for s in syms}
    s = make_settings(tmp_path, "x", syms)
    broker = AfterHoursBroker(equity=EQ)
    j = None
    for _ in range(6):
        _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    gross = notional_of(broker.orders, px) / EQ
    n_pos = len({o[0] for o in broker.orders})
    # Regresión B2 (24 oct): antes 12 posiciones al 120 % (bruto y turnover).
    # Las pendientes entran en la sombra y el turnover es por sesión: el
    # primer ciclo gasta el presupuesto del día y los cinco siguientes no compran.
    assert gross <= RiskLimits().max_gross_exposure
    assert n_pos <= RiskLimits().max_positions
    assert gross <= RiskLimits().max_daily_turnover_pct
    assert [o[0] for o in broker.orders] == ["S0", "S1"] and gross == 0.2


def test_B2_pending_exposure_binds_caps_without_turnover_help(tmp_path, monkeypatch):
    """La siembra con pendientes, sola: con el turnover desactivado (200 %), el
    tope de posiciones debe seguir valiendo entre ciclos con el mercado cerrado."""
    syms = [f"S{i}" for i in range(12)]
    snaps = {s: snap(s, mom=0.5 - i / 100) for i, s in enumerate(syms)}
    px = {s: 100.0 for s in syms}
    L = RiskLimits(max_daily_turnover_pct=2.0, max_sector_exposure_pct=1.0)
    s = make_settings(tmp_path, "x", syms, limits=L)
    broker = AfterHoursBroker(equity=EQ)
    j = None
    for _ in range(6):
        _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    n_pos = len({o[0] for o in broker.orders})
    gross = notional_of(broker.orders, px) / EQ
    assert n_pos == L.max_positions == 8                     # antes: 12
    assert gross <= L.max_gross_exposure


def test_B2_seed_counts_pending_orders_once():
    acct = MockBroker(equity=EQ).state
    acct.positions = {"A": Position("A", 100, 100.0, 10_000.0)}
    pending = [
        OpenOrder(id="b", symbol="B", qty=50, side="buy"),                  # nueva: 5 000
        OpenOrder(id="s", symbol="A", qty=100, side="sell"),                # no resta exposición
        OpenOrder(id="p", symbol="A", qty=100, side="buy", filled_qty=60),  # solo lo que falta: 4 000
        OpenOrder(id="n", symbol="C", qty=0, side="buy", notional=2_500.0),  # orden en dólares
        OpenOrder(id="f", symbol="D", qty=10, side="buy", filled_qty=10),   # llena: ya está en positions
    ]
    prices = {"A": 100.0, "B": 100.0, "C": 50.0, "D": 10.0}
    ctx = risk.CycleContext(peak_equity=EQ)
    ctx.seed(acct, pending=pending, prices=prices, booked={"b": 5_000.0, "old": 1_000.0})
    assert ctx.held_notional == {"A": 14_000.0, "B": 5_000.0, "C": 2_500.0}
    assert ctx.gross == 21_500.0 and ctx.n_positions == 3
    # turnover: lo reservado (5 000 + 1 000) + pendientes que el libro no conoce
    # (venta 10 000 + parcial 4 000 + dólares 2 500); "b" no se cuenta dos veces.
    assert ctx.turnover_used == 6_000.0 + 10_000.0 + 4_000.0 + 2_500.0
    assert ctx.unvalued == []


def test_B2_unpriced_pending_order_halts_buys_not_sells():
    acct = MockBroker(equity=EQ).state
    acct.positions = {"A": Position("A", 10, 100.0, 1_000.0)}
    ctx = risk.CycleContext(peak_equity=EQ)
    ctx.seed(acct, pending=[OpenOrder(id="z", symbol="Z", qty=5, side="buy")],
             prices={"A": 100.0, "Z": float("nan")})
    assert ctx.unvalued == ["Z"]
    dec = lambda sym, act: ExecutiveDecision(symbol=sym, action=act, conviction=5, thesis="t",
                                             invalidation="i", horizon_days=10)
    v = risk.evaluate(dec("X", "BUY"), snap("X"), acct, RiskLimits(), ctx)
    assert not v.approved and "could not be valued" in v.reasons[-1]
    assert risk.evaluate(dec("A", "SELL"), snap("A"), acct, RiskLimits(), ctx).approved


def test_B2_turnover_is_per_session_across_open_market_cycles(tmp_path, monkeypatch):
    """Mercado abierto (las órdenes se llenan): dos ciclos en la misma sesión
    comparten el presupuesto; la sesión siguiente lo renueva."""
    syms = [f"S{i}" for i in range(8)]
    snaps = {x: snap(x, mom=0.5 - i / 100) for i, x in enumerate(syms)}
    s = make_settings(tmp_path, "x", syms)
    broker = FillingBroker(EQ, {x: 100.0 for x in syms})
    broker.session = date(2026, 10, 26)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps)
    assert len(broker.orders) == 2                                    # 20 % de 25 %
    run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert len(broker.orders) == 2                                    # misma sesión: nada
    assert sum(j.booked_turnover("2026-10-26").values()) == 20_000.0
    broker.session = date(2026, 10, 27)
    run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert len(broker.orders) == 4                                    # sesión nueva


def test_B2_dry_run_does_not_book_real_turnover(tmp_path, monkeypatch):
    from velveteentrade.broker import DryRunBroker
    syms = ["S0", "S1"]
    s = make_settings(tmp_path, "x", syms)
    inner = MockBroker(equity=EQ); inner.session = date(2026, 10, 26)
    dry = DryRunBroker(inner)
    _, j = run_cycle(s, dry, buy_all(), monkeypatch, {x: snap(x) for x in syms})
    assert dry.intended_orders and j.booked_turnover("2026-10-26") == {}


def test_R2b_second_concurrent_cycle_is_refused(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["S0"])
    broker = MockBroker(equity=EQ)
    assert pipeline._CYCLE_LOCK.acquire(blocking=False)      # «otro ciclo» en curso
    try:
        try:
            run_cycle(s, broker, buy_all(), monkeypatch, {"S0": snap("S0")})
            refused = False
        except pipeline.CycleInProgress:
            refused = True
        assert refused and broker.orders == []
    finally:
        pipeline._CYCLE_LOCK.release()
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, {"S0": snap("S0")})
    assert broker.orders and not pipeline._CYCLE_LOCK.locked()   # liberado al terminar


def test_B2_non_finite_turnover_is_booked_as_infinite(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.book_turnover("2026-10-26", "x", float("nan"))
    assert j.booked_turnover("2026-10-26") == {"x": math.inf}


# ======================================================================
# B3 — Un close NaN (p. ej. una celda vacía en la caché CSV) apaga las
#      comparaciones: el stop no puede dispararse y, tras una SELL, el
#      turnover queda en NaN y deja de limitar las compras del ciclo.
# ======================================================================
def _bars_with_nan_last_close(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    c = np.linspace(50, 100, n)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)
    df.iloc[-1, df.columns.get_loc("close")] = np.nan
    return df


def test_repro_B3a_nan_close_reaches_the_gate_and_disables_the_stop():
    # Regresión B3 (24 oct). Antes: snapshot.close = nan, atr = 2.0, y
    # check_stops devolvía [] con el stop congelado en 96.
    # 1) El NaN ya no llega: el snapshot se niega a existir sin precio válido.
    try:
        indicators.snapshot("AAPL", _bars_with_nan_last_close())
        raised = False
    except ValueError:
        raised = True
    assert raised
    try:
        snap("AAPL", close=float("nan"))
        rejected = False
    except Exception:                                    # pydantic ValidationError
        rejected = True
    assert rejected
    # 2) Sin snapshot del día, el stop usa la marca del bróker y SÍ se dispara.
    acct = MockBroker(equity=EQ).state
    acct.positions = {"AAPL": Position("AAPL", 50, 100.0, 4_750.0)}       # marca 95
    v = risk.check_stops(acct, {}, RiskLimits(), {}, stops={"AAPL": 96.0})
    assert [(x.symbol, x.qty) for x in v] == [("AAPL", 50)]
    # 3) Y si un NaN llegara por otro camino (model_construct salta pydantic):
    nan_snap = TechnicalSnapshot.model_construct(symbol="AAPL", close=float("nan"), atr_14=2.0)
    v = risk.check_stops(acct, {"AAPL": nan_snap}, RiskLimits(), {}, stops={"AAPL": 96.0})
    assert [(x.symbol, x.qty) for x in v] == [("AAPL", 50)]
    # 4) Sin precio ni stop válidos: no se calla, queda en `unchecked`.
    acct.positions = {"AAPL": Position("AAPL", 50, 100.0, float("nan"))}
    unchecked = []
    assert risk.check_stops(acct, {}, RiskLimits(), {}, stops={}, unchecked=unchecked) == []
    assert unchecked == ["AAPL"]


def test_repro_B3b_nan_sell_poisons_turnover_budget(tmp_path, monkeypatch):
    syms = ["AAPL"] + [f"S{i}" for i in range(6)]
    snaps = {x: snap(x, mom=0.5 - i / 100) for i, x in enumerate(syms[1:])}  # AAPL: indicadores reales
    broker = MockBroker(equity=EQ)
    broker.state.positions = {"AAPL": Position("AAPL", 50, 100.0, 5_000.0)}
    llm = ScriptedLLM(lambda x: dict(action="SELL" if x == "AAPL" else "BUY", conviction=5))
    summary, _ = run_cycle(make_settings(tmp_path, "x", syms), broker, llm, monkeypatch, snaps,
                           data=FakeData(bars={"AAPL": _bars_with_nan_last_close()}))
    bought = notional_of(broker.orders, {x: 100.0 for x in syms})
    # Regresión B3 (24 oct). Antes: SELL AAPL aprobada con NaN, turnover NaN y
    # 60 000 comprados con tope de 25 000. Ahora AAPL no tiene precio del día:
    # no se analiza ni se vende por el LLM, y el turnover sigue limitando.
    assert ("AAPL", 50, "sell") not in broker.orders
    assert bought == 20_000.0 and bought <= EQ * RiskLimits().max_daily_turnover_pct
    assert "AAPL" not in summary["analyzed"]


def test_repro_B3c_nan_buy_crashes_after_mutating_the_shadow():
    ctx = risk.CycleContext(peak_equity=EQ)
    acct = MockBroker(equity=EQ).state
    nan_snap = TechnicalSnapshot.model_construct(symbol="X", close=float("nan"), atr_14=2.0,
                                                 hist_cvar_95_20d=None)
    dec = ExecutiveDecision(symbol="X", action="BUY", conviction=5, thesis="t",
                            invalidation="i", horizon_days=10)
    # Regresión B3 (24 oct). Antes: ValueError en _stop_price con ctx.gross ya NaN.
    v = risk.evaluate(dec, nan_snap, acct, RiskLimits(), ctx)
    assert not v.approved and "non-finite" in v.reasons[-1]
    assert ctx.gross == 0.0 and ctx.turnover_used == 0.0 and ctx.held_notional == {}


def test_B3_non_finite_position_value_halts_buys():
    acct = MockBroker(equity=EQ).state
    acct.positions = {"A": Position("A", 10, 100.0, float("nan"))}
    ctx = risk.CycleContext(peak_equity=EQ)
    dec = ExecutiveDecision(symbol="X", action="BUY", conviction=5, thesis="t",
                            invalidation="i", horizon_days=10)
    v = risk.evaluate(dec, snap("X"), acct, RiskLimits(), ctx)
    assert ctx.unvalued == ["A"] and not v.approved and "could not be valued" in v.reasons[-1]


def test_B3_blank_cell_inside_history_gives_none_not_nan():
    df = _bars_with_nan_last_close()
    df.iloc[-1, df.columns.get_loc("close")] = 100.0          # hoy hay precio...
    df.iloc[-30, df.columns.get_loc("close")] = np.nan        # ...pero falta un día del pasado
    s = indicators.snapshot("AAPL", df)
    assert s.close == 100.0
    for name, v in s.model_dump().items():
        assert not (isinstance(v, float) and not math.isfinite(v)), name


# ======================================================================
# B4 — El stop congelado se guarda por SÍMBOLO, no por lote: una compra
#      adicional lo sobrescribe y el primer lote pierde su cota (G2).
# ======================================================================
def test_repro_B4_add_on_buy_overwrites_frozen_stop(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=100.0, atr=2.0)})
    assert broker.orders == [("X", 100.0, "buy")] and j.stop_prices() == {"X": 96.0}
    budget = EQ * RiskLimits().per_trade_risk_pct * 5 / 5            # 1.000 USD
    assert 100 * (100.0 - 96.0) <= budget                             # lote 1: 400 <= 1.000

    # Se llenó; al día siguiente X cierra en 97 y el ATR se triplica.
    broker.state.positions = {"X": Position("X", 100, 100.0, 9_700.0)}
    run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X", close=97.0, atr=6.0)}, journal=j)
    assert broker.orders[-1][2] == "buy"                              # compra adicional pequeña
    add_qty = broker.orders[-1][1]
    # Regresión B4 (24 oct). Antes: el stop bajaba 96 -> 85 y el lote 1 perdía
    # 1 500 con presupuesto de 1 000; a 86 el stop no saltaba.
    assert j.stop_prices() == {"X": 96.0}                             # se aprieta, nunca se afloja
    assert 100 * (100.0 - 96.0) <= budget                             # lote 1: 400 <= 1 000
    assert add_qty * (97.0 - 96.0) <= budget                          # lote 2, con su propio presupuesto
    held = {"X": Position("X", 100 + add_qty, 100.0, (100 + add_qty) * 86.0)}
    acct = broker.state; acct.positions = held
    v = risk.check_stops(acct, {"X": snap("X", close=86.0, atr=6.0)}, RiskLimits(),
                         j.entry_prices(), stops=j.stop_prices())
    assert [(x.symbol, x.qty) for x in v] == [("X", 100 + add_qty)]    # a 86 SÍ salta


def test_B4_set_stop_is_tighten_only(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.set_stop("X", 96.0) == 96.0
    assert j.set_stop("X", 85.0) == 96.0                              # no se afloja
    assert j.set_stop("X", 98.5) == 98.5                              # sí se aprieta
    assert j.stop_prices() == {"X": 98.5}


def test_B4_orphan_stop_cleared_unless_a_buy_is_queued(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_stop("GONE", 96.0)       # posición cerrada a mano: su stop quedaría huérfano
    j.set_stop("QUEUED", 50.0)     # compra en cola, aún no es posición
    j.set_stop("HELD", 10.0)
    j.reconcile_entries({"HELD": Position("HELD", 1, 11.0, 11.0)}, pending_buys={"QUEUED"})
    assert j.stop_prices() == {"QUEUED": 50.0, "HELD": 10.0}


def test_B4_dry_run_never_touches_real_stops(tmp_path, monkeypatch):
    from velveteentrade.broker import DryRunBroker
    s = make_settings(tmp_path, "x", ["X", "Y"])
    j = Journal(s.db_path); j.set_stop("X", 90.0)
    inner = MockBroker(equity=EQ)
    inner.state.positions = {"X": Position("X", 10, 100.0, 1_000.0)}
    llm = ScriptedLLM(lambda x: dict(action="SELL" if x == "X" else "BUY", conviction=5))
    run_cycle(s, DryRunBroker(inner), llm, monkeypatch,
              {"X": snap("X"), "Y": snap("Y")}, journal=j)
    assert j.stop_prices() == {"X": 90.0}          # ni el SELL de X lo borra ni el BUY de Y crea uno


# ======================================================================
# B5 — Un stop ejecutado no actualiza la sombra y el símbolo puede volver
#      por el screen: el LLM pide SELL y sale una segunda venta.
# ======================================================================
def test_repro_B5_stop_then_llm_sell_double_sells(tmp_path, monkeypatch):
    broker = MockBroker(equity=EQ)
    broker.state.positions = {"X": Position("X", 50, 100.0, 4_750.0)}
    s = make_settings(tmp_path, "x", ["X"])
    j = Journal(s.db_path); j.set_stop("X", 96.0)
    llm = ScriptedLLM(lambda x: dict(action="SELL", conviction=3))
    run_cycle(s, broker, llm, monkeypatch, {"X": snap("X", close=95.0, mom=0.30)}, journal=j)
    sold = sum(q for sym, q, sd in broker.orders if sym == "X" and sd == "sell")
    # Regresión B5 (24 oct). Antes: [("X", 50, "sell"), ("X", 50, "sell")], 100
    # vendidas teniendo 50. El símbolo detenido ya no vuelve por el screen.
    assert broker.orders == [("X", 50, "sell")]
    assert sold == 50


def test_B5_gate_refuses_second_order_on_a_stopped_symbol():
    """Defensa en profundidad: aunque el símbolo llegara a la puerta, la puerta
    sabe que el barrido ya lo vendió este ciclo."""
    acct = MockBroker(equity=EQ).state
    acct.positions = {"X": Position("X", 50, 100.0, 4_750.0)}
    ctx = risk.CycleContext(peak_equity=EQ)
    ctx.seed(acct)
    ctx.mark_traded("X")                                              # lo hizo el barrido
    for action in ("SELL", "BUY"):
        v = risk.evaluate(ExecutiveDecision(symbol="X", action=action, conviction=5, thesis="t",
                                            invalidation="i", horizon_days=10),
                          snap("X", close=95.0), acct, RiskLimits(), ctx)
        assert not v.approved and "one order per symbol" in v.reasons[-1]


# ======================================================================
# B6 — Capital del perfil: unidades y bases mezcladas.
# ======================================================================
def test_repro_B6a_capital_zero_means_full_account(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    save_profile(s.profile_path, capital=0.0)                         # "todavía no quiero invertir"
    broker = MockBroker(equity=EQ)
    summary, _ = run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X")})
    # Regresión B6a (24 oct). Antes: compra de 10 000 USD (10 % de la cuenta entera).
    assert broker.orders == []
    assert "no capital allocated" in summary["rejected"][0]


def test_repro_B6b_satellite_losses_invisible_to_halt_and_track_record(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    prof = save_profile(s.profile_path, capital=1_000.0)
    sat = (1 - profile_mod.etf_core_weight(prof)) * 1_000.0             # 600 (núcleo 40 %)
    snaps = {"X": snap("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, ScriptedLLM(lambda x: dict(action="HOLD")), monkeypatch, snaps)
    broker.state.equity = EQ - 100.0             # el satélite pierde 100 (16,7 %)
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    # Regresión B6b (24 oct). Antes: track record [1000, 1000] y la compra salía.
    assert [e for _, e in j.equity_history()] == [sat, sat - 100.0]   # la pérdida se ve
    assert summary["orders"] == [] and "drawdown" in summary["rejected"][0]  # 16,7 % > 12 %


def test_repro_B6c_peak_from_full_account_halts_forever(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    snaps = {"X": snap("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, ScriptedLLM(lambda x: dict(action="HOLD")), monkeypatch, snaps)
    save_profile(s.profile_path, capital=200.0)     # el usuario hace /start después
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    # Regresión B6c (24 oct). Antes: «drawdown» para siempre (200 frente a un
    # pico de 100 000). Ahora la base nueva reinicia el pico y la compra sale.
    assert summary["orders"] and not any("drawdown" in r for r in summary["rejected"])
    assert j.get_float("peak_equity") == 0.6 * 200.0


def test_B6_capital_base_rules(tmp_path):
    from velveteentrade import capital as capital_mod
    j = Journal(tmp_path / "j.db")
    prof = profile_mod.RiskProfile(name="U", tolerance=4, horizon_years=10, experience="INTERMEDIATE",
                                   max_drawdown_comfort=0.2, income_stability="STABLE",
                                   has_emergency_fund=True, capital=1_000.0)
    b = capital_mod.resolve(j, prof, 100_000.0)
    assert b.rebased and (b.allocated, b.value, b.sizing) == (600.0, 600.0, 600.0)
    b = capital_mod.resolve(j, prof, 100_300.0)                       # ganancia: se ve, no se apalanca
    assert not b.rebased and (b.value, b.sizing) == (900.0, 600.0)
    b = capital_mod.resolve(j, prof, 99_700.0)                        # pérdida: se ve y achica
    assert (b.value, b.sizing) == (300.0, 300.0)
    b = capital_mod.resolve(j, prof, 99_000.0)                        # se perdió todo: no compra
    assert (b.value, b.sizing) == (-400.0, 0.0)
    b = capital_mod.resolve(j, prof.model_copy(update={"capital": 2_000.0}), 99_000.0)
    assert b.rebased and b.allocated == 1_200.0 and j.get_float("peak_equity") == 1_200.0
    b = capital_mod.resolve(j, prof.model_copy(update={"capital": 10**9}), 50.0)
    assert b.rebased and b.sizing <= 50.0                             # nunca más que la cuenta
    try:
        profile_mod.RiskProfile(**dict(prof.model_dump(), capital=-5.0))
        negative_ok = True
    except Exception:
        negative_ok = False
    assert not negative_ok                                            # capital >= 0


def test_B6_no_profile_keeps_the_existing_peak(tmp_path):
    """Primera corrida de este código sin perfil: el pico viejo está en la misma
    base (la cuenta) y no se pierde."""
    from velveteentrade import capital as capital_mod
    j = Journal(tmp_path / "j.db")
    j.set_float("peak_equity", 120_000.0)
    b = capital_mod.resolve(j, None, 100_000.0)
    assert (b.value, b.sizing) == (100_000.0, 100_000.0)
    assert j.get_float("peak_equity") == 120_000.0


def test_B6_track_record_pnl_is_measured_within_the_current_base(tmp_path):
    from velveteentrade import capital as capital_mod
    j = Journal(tmp_path / "j.db")
    capital_mod.resolve(j, None, 100_000.0); j.record_equity(100_000.0)
    prof = profile_mod.RiskProfile(name="U", tolerance=4, horizon_years=10, experience="INTERMEDIATE",
                                   max_drawdown_comfort=0.2, income_stability="STABLE",
                                   has_emergency_fund=True, capital=1_000.0)
    capital_mod.resolve(j, prof, 100_000.0); j.record_equity(600.0)
    j.record_equity(550.0)
    start, now, since = capital_mod.pnl_since_base(j)
    assert (start, now) == (600.0, 550.0) and since                   # no -99 % por el salto de base


# ======================================================================
# B7 — El perfil cambia los límites "desde el próximo ciclo", pero el bot
#      calcula settings.risk una vez al arrancar (cli.main -> apply_profile).
# ======================================================================
def test_repro_B7_bot_keeps_startup_limits_after_new_profile(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    profile_mod.apply_profile(s)                                      # arranque sin perfil
    prof = save_profile(s.profile_path, tolerance=1, max_drawdown_comfort=0.05)  # /start después
    want = profile_mod.limits_for(prof)
    assert want.min_conviction == 4 and want.max_position_pct == 0.05
    broker = MockBroker(equity=EQ)
    summary, j = run_cycle(s, broker, buy_all(conv=3), monkeypatch, {"X": snap("X")})  # = _cycle_in_thread
    # Regresión B7 (24 oct). Antes: compra de 100 acciones (10 %) con convicción 3.
    assert broker.orders == [] and "conviction 3 < minimum 4" in summary["rejected"][0]
    broker.session = date(2026, 10, 27)
    run_cycle(s, broker, buy_all(conv=4), monkeypatch, {"X": snap("X")}, journal=j)
    sat = (1 - profile_mod.etf_core_weight(prof)) * EQ
    assert broker.orders and broker.orders[0][1] * 100.0 <= want.max_position_pct * sat + 1e-9


# ======================================================================
# B8 — El plan dice "núcleo ETF 60 % / satélite 40 %", pero la puerta
#      dimensiona contra el 100 % del capital.
# ======================================================================
def test_repro_B8_satellite_exceeds_core_satellite_split(tmp_path, monkeypatch):
    syms = [f"S{i}" for i in range(10)]
    s = make_settings(tmp_path, "x", syms)
    prof = save_profile(s.profile_path, tolerance=3, capital=10_000.0)
    profile_mod.apply_profile(s)                                      # límites del perfil
    satellite = 1 - profile_mod.etf_core_weight(prof)                 # 0.40
    px = {x: 50.0 for x in syms}
    snaps = {x: snap(x, close=50.0, atr=0.5, mom=0.5 - i / 100) for i, x in enumerate(syms)}
    broker = FillingBroker(EQ, px)
    j = None
    for day in range(6):                                              # seis días, órdenes llenas
        # Una sesión distinta por ciclo: desde B2 el turnover es por sesión, y
        # seis ciclos en la MISMA sesión ya no apilan compras (eso era B2).
        broker.session = date(2026, 10, 19 + day)
        _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    gross = sum(p.market_value for p in broker.state.positions.values()) / 10_000.0
    # Regresión B8 (24 oct). Antes: satélite al 64 % del capital con 40 % prometido.
    # Ahora se dimensiona contra el satélite (4 000): 8 posiciones de 8 % = 25,6 %.
    assert satellite == 0.40 and gross <= satellite
    assert abs(gross - 0.256) < 1e-9


# ======================================================================
# D3 — Fidelidad del modelo: BUY y luego SELL del mismo símbolo en un ciclo.
#      La sombra (lo que Lean demuestra) pone la exposición en 0, pero la
#      SELL solo cierra la cantidad del bróker: la cartera real conserva la
#      compra. cycle_safe sigue siendo cierto sobre la sombra, no sobre la cuenta.
# ======================================================================
def test_repro_D3_shadow_diverges_from_real_portfolio_at_gate_level():
    L = RiskLimits(max_positions=20, max_gross_exposure=0.30, max_daily_turnover_pct=1.0,
                   max_sector_exposure_pct=1.0)
    acct = MockBroker(equity=EQ).state
    acct.positions = {"X": Position("X", 10, 100.0, 1_000.0)}
    ctx = risk.CycleContext(peak_equity=EQ)
    real = {"X": 10.0}

    def go(sym, action):
        v = risk.evaluate(ExecutiveDecision(symbol=sym, action=action, conviction=5, thesis="t",
                                            invalidation="i", horizon_days=10),
                          snap(sym, close=100.0, atr=1.0), acct, L, ctx)
        if v.approved:
            real[sym] = real.get(sym, 0.0) + (v.qty if action == "BUY" else -v.qty)
        return v

    assert go("X", "BUY").approved
    assert not go("X", "SELL").approved            # Regresión D3 (24 oct): una orden por símbolo y ciclo
    for sym in "ABCDE":
        go(sym, "BUY")
    real_gross = sum(q * 100.0 for q in real.values()) / EQ
    # Antes: sombra <= 30 % y cuenta real 39 % (X conservaba 90 acciones).
    # Ahora la SELL se rechaza, la sombra no "cierra" lo que la cuenta tiene,
    # y las dos coinciden (es la hipótesis Nodup de la propuesta P2).
    assert ctx.gross / EQ <= 0.30 + 1e-12
    assert abs(real_gross - ctx.gross / EQ) < 1e-9 and real_gross <= 0.30 + 1e-12


# ======================================================================
# D5 — Las hipótesis de redondeo del modelo Lean son falsas sobre floats
#      (efecto ~1e-14 USD: divergencia de fidelidad, no riesgo económico).
# ======================================================================
def test_repro_D5_float_rounding_breaks_lean_hypotheses():
    x = 0.11699999999999999
    assert Fraction(risk._round_qty(x)) > Fraction(x)                 # RoundDown.le_of_pos es falso
    c, d = 392.85, 2.0 * 3.945
    assert Fraction(risk._stop_price(c, d)) < Fraction(c) - Fraction(d)   # stopPrice_ge es falso


# ======================================================================
# R12 — Autorización del bot (parche 24 oct): falla cerrada.
# ======================================================================
def test_R12_allowed_chat_ids_parsing():
    from velveteentrade.bot import allowed_chat_ids, is_authorised
    assert allowed_chat_ids("") == frozenset()
    assert allowed_chat_ids(" 123, -456 ,,abc") == frozenset({123, -456})
    allowed = allowed_chat_ids("123")
    assert is_authorised(123, allowed)
    assert not is_authorised(999, allowed)
    assert not is_authorised(None, allowed)
    assert not is_authorised(123, frozenset())   # vacío => nadie


def test_R12_env_unset_means_nobody(monkeypatch):
    from velveteentrade.bot import allowed_chat_ids
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    assert allowed_chat_ids() == frozenset()
