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
    assert gross == 1.2 and gross > RiskLimits().max_gross_exposure          # 120 % > 100 %
    assert n_pos == 12 and n_pos > RiskLimits().max_positions               # 12 > 8
    assert gross > RiskLimits().max_daily_turnover_pct                      # mismo día: 120 % > 25 %


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
    s = indicators.snapshot("AAPL", _bars_with_nan_last_close())
    assert math.isnan(s.close) and s.atr_14 == 2.0          # pydantic acepta NaN
    acct = MockBroker(equity=EQ).state
    acct.positions = {"AAPL": Position("AAPL", 50, 100.0, 5_000.0)}
    # stop congelado en 96: con close NaN la comparación `close <= stop` es False
    assert risk.check_stops(acct, {"AAPL": s}, RiskLimits(), {}, stops={"AAPL": 96.0}) == []


def test_repro_B3b_nan_sell_poisons_turnover_budget(tmp_path, monkeypatch):
    syms = ["AAPL"] + [f"S{i}" for i in range(6)]
    snaps = {x: snap(x, mom=0.5 - i / 100) for i, x in enumerate(syms[1:])}  # AAPL: indicadores reales
    broker = MockBroker(equity=EQ)
    broker.state.positions = {"AAPL": Position("AAPL", 50, 100.0, 5_000.0)}
    llm = ScriptedLLM(lambda x: dict(action="SELL" if x == "AAPL" else "BUY", conviction=5))
    run_cycle(make_settings(tmp_path, "x", syms), broker, llm, monkeypatch, snaps,
              data=FakeData(bars={"AAPL": _bars_with_nan_last_close()}))
    bought = notional_of(broker.orders, {x: 100.0 for x in syms})
    assert ("AAPL", 50, "sell") in broker.orders
    assert bought == 60_000.0 and bought > EQ * RiskLimits().max_daily_turnover_pct  # 60 % > 25 %


def test_repro_B3c_nan_buy_crashes_after_mutating_the_shadow():
    ctx = risk.CycleContext(peak_equity=EQ)
    acct = MockBroker(equity=EQ).state
    nan_snap = snap("X", close=float("nan"), atr=2.0)
    try:
        risk.evaluate(ExecutiveDecision(symbol="X", action="BUY", conviction=5, thesis="t",
                                        invalidation="i", horizon_days=10),
                      nan_snap, acct, RiskLimits(), ctx)
        raised = False
    except ValueError:                        # math.ceil(nan) en _stop_price
        raised = True
    assert raised and math.isnan(ctx.gross)   # la sombra quedó envenenada antes del error


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
    assert j.stop_prices() == {"X": 85.0}                             # 96 -> 85
    lot1_loss_at_stop = 100 * (100.0 - 85.0)
    assert lot1_loss_at_stop == 1_500.0 > budget                      # G2 roto para el lote 1
    held = {"X": Position("X", 103.092, 100.0, 103.092 * 86.0)}
    acct = broker.state; acct.positions = held
    assert risk.check_stops(acct, {"X": snap("X", close=86.0, atr=6.0)}, RiskLimits(),
                            j.entry_prices(), stops=j.stop_prices()) == []   # a 86 no salta


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
    assert broker.orders == [("X", 50, "sell"), ("X", 50, "sell")]
    assert sold == 100 > 50                                           # corto de 50 si ambas se llenan


# ======================================================================
# B6 — Capital del perfil: unidades y bases mezcladas.
# ======================================================================
def test_repro_B6a_capital_zero_means_full_account(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    save_profile(s.profile_path, capital=0.0)                         # "todavía no quiero invertir"
    broker = MockBroker(equity=EQ)
    run_cycle(s, broker, buy_all(), monkeypatch, {"X": snap("X")})
    assert broker.orders == [("X", 100.0, "buy")]                     # 10.000 USD = 10 % de la cuenta entera


def test_repro_B6b_satellite_losses_invisible_to_halt_and_track_record(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    save_profile(s.profile_path, capital=1_000.0)
    snaps = {"X": snap("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, ScriptedLLM(lambda x: dict(action="HOLD")), monkeypatch, snaps)
    broker.state.equity = EQ - 1_000.0           # se perdió el 100 % del capital declarado
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert [e for _, e in j.equity_history()] == [1_000.0, 1_000.0]   # el track record no lo ve
    assert summary["orders"] == ["BUY X x10.0"]                       # el freno por drawdown no actúa


def test_repro_B6c_peak_from_full_account_halts_forever(tmp_path, monkeypatch):
    s = make_settings(tmp_path, "x", ["X"])
    snaps = {"X": snap("X", close=10.0, atr=0.2)}
    broker = MockBroker(equity=EQ)
    _, j = run_cycle(s, broker, ScriptedLLM(lambda x: dict(action="HOLD")), monkeypatch, snaps)
    save_profile(s.profile_path, capital=200.0)     # el usuario hace /start después
    summary, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    assert summary["orders"] == [] and "drawdown" in summary["rejected"][0]  # 200 vs pico 100.000


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
    run_cycle(s, broker, buy_all(conv=3), monkeypatch, {"X": snap("X")})  # = _cycle_in_thread
    assert broker.orders == [("X", 100.0, "buy")]                     # conv 3 < 4 y 10 % > 5 %


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
    for _ in range(6):                                                # seis días, órdenes llenas
        _, j = run_cycle(s, broker, buy_all(), monkeypatch, snaps, journal=j)
    gross = sum(p.market_value for p in broker.state.positions.values()) / 10_000.0
    assert satellite == 0.40 and gross > satellite                   # 0.64 > 0.40
    assert abs(gross - 0.64) < 1e-9


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

    go("X", "BUY"); go("X", "SELL")
    for sym in "ABCDE":
        go(sym, "BUY")
    real_gross = sum(q * 100.0 for q in real.values()) / EQ
    assert ctx.gross / EQ <= 0.30 + 1e-12          # la sombra respeta el tope...
    assert real_gross > 0.30                        # ...la cuenta real no
    assert real["X"] == 90.0


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
