"""The daily cycle: data → analysts → executive → risk gate → broker → journal.

Structure: analysis (LLM-bound) runs in parallel threads; everything that
touches money — the risk gate and order submission — runs strictly
sequentially, in screen-rank order, so budgets are deterministic.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from . import agents, indicators, risk
from .broker import AccountState, BrokerAdapter
from .config import Settings
from .data import MarketData
from .llm import StructuredLLM
from .memory import Journal
from .schemas import AnalystReport, ExecutiveDecision, JournalEntry, TechnicalSnapshot

log = logging.getLogger(__name__)

MAX_WORKERS = 4

# R2b: the bot runs /ciclo and the 21:30 job in worker threads of ONE process.
# Two cycles at once would both seed from the same broker photograph and
# each see at most part of the other's orders. One cycle at a time.
_CYCLE_LOCK = threading.Lock()


class CycleInProgress(RuntimeError):
    """Raised instead of starting a cycle while another one is running."""


def _canon_queries(snap: TechnicalSnapshot, sector: str | None) -> dict[str, str]:
    """Deterministic, low-cardinality queries so embeddings cache well across symbols."""
    mom = "positive" if (snap.momentum_63d or 0) > 0 else "negative"
    trend = "above" if snap.sma_200 and snap.close > snap.sma_200 else "below"
    vol = "high" if (snap.realized_vol_21d or 0) > 0.30 else "moderate"
    return {
        "technical": (
            f"swing trading interpretation of {mom} 3-month momentum, price {trend} the "
            f"200-day average, {vol} volatility, tail risk and trend-following entries and exits"
        ),
        "fundamental": (
            f"valuation multiples, profit margins, leverage and quality assessment, "
            f"margin of safety, {sector or 'general'} sector"
        ),
        "executive": (
            "decision discipline: conviction and bet sizing, cost of overtrading, "
            "selling discipline and thesis invalidation, when to hold"
        ),
    }


def _analyze_symbol(
    settings: Settings,
    data: MarketData,
    llm: StructuredLLM,
    snap: TechnicalSnapshot,
    held_qty: float,
    past_lessons: list[str],
    canon: dict[str, list[str]],
) -> tuple[str, AnalystReport, AnalystReport, ExecutiveDecision]:
    symbol = snap.symbol
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_f = pool.submit(
            agents.run_fundamental, llm, settings.models.analyst, symbol,
            data.fundamentals(symbol), data.news(symbol), canon.get("fundamental"),
        )
        fut_t = pool.submit(agents.run_technical, llm, settings.models.analyst, snap,
                            canon.get("technical"))
        fundamental, technical = fut_f.result(), fut_t.result()
    decision = agents.run_executive(
        llm, settings.models.executive, symbol, fundamental, technical,
        held_qty=held_qty, past_lessons=past_lessons, canon=canon.get("executive"),
    )
    return symbol, fundamental, technical, decision


def run_cycle(
    settings: Settings,
    broker: BrokerAdapter,
    data: MarketData,
    llm: StructuredLLM,
    journal: Journal,
) -> dict:
    """One full decision cycle. Returns a summary dict for logging/notification.

    Never runs concurrently with another cycle in this process: a second call
    while one is running raises CycleInProgress immediately (it does not wait
    and then trade on a stale view)."""
    if not _CYCLE_LOCK.acquire(blocking=False):
        raise CycleInProgress("Another trading cycle is already running.")
    try:
        return _run_cycle(settings, broker, data, llm, journal)
    finally:
        _CYCLE_LOCK.release()


def _run_cycle(
    settings: Settings,
    broker: BrokerAdapter,
    data: MarketData,
    llm: StructuredLLM,
    journal: Journal,
) -> dict:
    account = broker.account()

    # Pending (unfilled) orders — e.g. queued while the market is closed — are
    # exposure the account() snapshot doesn't show. A symbol with an order in
    # flight is untouchable this cycle: no re-analysis, no second order.
    pending = broker.open_orders()
    pending_symbols = {o.symbol for o in pending}

    # ------------------------------------------- 0. Reconcile with the broker
    # Broker fills are the source of truth for entry prices (the price recorded
    # at order time was the prior close, not the actual fill). A dry run must
    # not edit the real journal's entries or frozen stops.
    if broker.sends_orders:
        journal.reconcile_entries(account.positions,
                                  pending_buys={o.symbol for o in pending if o.side == "buy"})

    # Profile: read EVERY cycle (B7 — the bot used to keep the limits it
    # computed at startup, so a new /start profile never reached the gate).
    # Its deterministic limits replace config.yaml's.
    from . import capital as capital_mod
    from . import profile as profile_mod

    prof = profile_mod.load(settings.profile_path)
    window = settings.risk.drawdown_window_sessions   # R31: de config.yaml, no del perfil
    limits = profile_mod.limits_for(prof, window=window) if prof else settings.risk

    # Capital base (B6/B8, see capital.py): size against the satellite the
    # user allocated (E); track record and drawdown halt read what it is
    # worth today (V), losses included — a $100k demo account trading like a
    # $200 one, honestly.
    # R20: con allocation.mode = separate_account el núcleo vive en otra
    # cuenta y esta es 100 % satélite (20 % del total en paper, 10 % en real).
    base = capital_mod.resolve(journal, prof, account.equity,
                               allocation=settings.allocation, paper=settings.alpaca_paper)
    if base.rebased:
        log.info("New capital base: satellite $%.2f (core %.0f%%, account $%.2f).",
                 base.allocated, base.core_weight * 100, account.equity)
    log.info("Satellite value $%.2f, sizing against $%.2f (account equity $%.2f)",
             base.value, base.sizing, account.equity)
    account = AccountState(equity=base.sizing, cash=account.cash, positions=account.positions)
    # R31: el pico del freno es el máximo de las últimas N sesiones de la base
    # actual, no el histórico (que no se soltaba nunca tras quedar en efectivo).
    session = broker.session_date().isoformat()
    since = (capital_mod.current_base(journal) or {}).get("since")
    peak_equity = journal.update_window_peak(session, base.value, limits.drawdown_window_sessions,
                                             since=since)
    journal.record_equity(base.value)

    if pending:
        log.warning("Pending orders detected — skipping these symbols this cycle: %s",
                    sorted(pending_symbols))

    summary: dict = {"analyzed": [], "orders": [], "stops": [], "rejected": [],
                     "pending_skipped": sorted(pending_symbols)}

    # ------------------------------------------------ 1. Data + indicators
    snaps: dict[str, TechnicalSnapshot] = {}
    sectors: dict[str, str] = {}
    for symbol in settings.universe:
        try:
            df = data.daily_bars(symbol, settings.history_days)
            snaps[symbol] = indicators.snapshot(symbol, df)
            sector = (data.fundamentals(symbol) or {}).get("sector")
            if sector:
                sectors[symbol] = str(sector)
        except Exception as exc:
            log.warning("Skipping %s — data error: %s", symbol, exc)

    # The context carries a shadow portfolio (positions, gross, sectors) that
    # every approved order updates, so caps hold across the whole cycle, not
    # just per decision. Stopped-out positions stay counted until the next
    # cycle's broker snapshot — conservative, since their sells may not fill.
    #
    # B2: the shadow must also start from what EARLIER cycles committed and
    # the snapshot does not show — pending orders (valued at the last close)
    # and the turnover already booked for the session they execute in.
    booked = journal.booked_turnover(session)
    prices = {s: sn.close for s, sn in snaps.items()}
    for s, pos in account.positions.items():
        if s not in prices and pos.qty:
            prices[s] = abs(pos.market_value / pos.qty)
    ctx = risk.CycleContext(peak_equity=peak_equity, sectors=sectors, current_value=base.value)
    ctx.seed(account, pending=pending, prices=prices, booked=booked)
    summary["session"] = session
    summary["turnover_seeded"] = round(ctx.turnover_used, 2)
    if ctx.unvalued:
        log.warning("Pending orders that cannot be valued: %s — new buys halted this cycle.",
                    sorted(set(ctx.unvalued)))

    # ------------------------------------------ 2. Deterministic stops FIRST
    entries = journal.entry_prices()
    unchecked: list[str] = []
    stop_verdicts = risk.check_stops(account, snaps, limits, entries,
                                     stops=journal.stop_prices(), unchecked=unchecked)
    summary["stops_unchecked"] = unchecked
    for verdict in stop_verdicts:
        if verdict.symbol in pending_symbols:
            log.warning("Stop hit on %s but an order is already pending — not doubling.",
                        verdict.symbol)
            continue
        order_id = broker.submit_order(verdict.symbol, verdict.qty, "sell")
        ctx.mark_traded(verdict.symbol)
        if broker.sends_orders:
            journal.book_turnover(session, order_id,
                                  verdict.qty * prices.get(verdict.symbol, float("inf")))
            journal.clear_entry(verdict.symbol)
        journal.record(JournalEntry(symbol=verdict.symbol, verdict=verdict, executed=True,
                                    note=f"stop-loss order {order_id}"))
        summary["stops"].append(verdict.symbol)

    # ------------------------------------------------ 3. Candidate selection
    held_symbols = [s for s in account.positions if s in snaps and s not in summary["stops"]]
    screened = indicators.screen(list(snaps.values()), settings.candidates_per_cycle)
    # B5: a symbol stopped out this cycle must not come back through the
    # screen — it has a sell in flight; any further order would be a second one.
    candidates = [s for s in dict.fromkeys(held_symbols + [s.symbol for s in screened])
                  if s not in pending_symbols and s not in summary["stops"]]

    # --------------------------------------- 4. Analysis (parallel, LLM-bound)
    # SQLite (journal, RAG) is main-thread only — prefetch lessons and canon
    # excerpts here before fanning out.
    from .rag.retrieve import Retriever

    retriever = Retriever(settings.rag_db_path)
    if retriever.available:
        log.info("Canon RAG active — theses will cite sources.")
    lessons_by_symbol = {s: journal.lessons(s) for s in candidates}
    canon_by_symbol: dict[str, dict[str, list[str]]] = {}
    for s in candidates:
        queries = _canon_queries(snaps[s], sectors.get(s))
        canon_by_symbol[s] = {
            role: retriever.excerpts(q, top_k=3) for role, q in queries.items()
        } if retriever.available else {}

    results: dict[str, tuple] = {}
    failures: dict[str, Exception] = {}

    def _safe(symbol: str):
        held = account.positions.get(symbol)
        try:
            results[symbol] = _analyze_symbol(
                settings, data, llm, snaps[symbol],
                held.qty if held else 0.0,
                lessons_by_symbol.get(symbol, []),
                canon_by_symbol.get(symbol, {}),
            )
        except Exception as exc:  # noqa: BLE001
            failures[symbol] = exc

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(_safe, candidates))

    for symbol, exc in failures.items():
        log.error("Agent failure on %s — skipping symbol: %s", symbol, exc)
        journal.record(JournalEntry(symbol=symbol, note=f"agent failure: {exc}"))

    # -------------------------- 5. Risk gate + orders (STRICTLY sequential,
    # in candidate order, so turnover/sector budgets are deterministic)
    for symbol in candidates:
        if symbol not in results:
            continue
        _, fundamental, technical, decision = results[symbol]
        snap = snaps[symbol]

        # B1: the gate must judge exactly the symbol that would be ordered.
        # A case/whitespace variant of the same ticker is canonicalised; any
        # other symbol the LLM writes is rejected, never ordered.
        if decision.symbol != symbol:
            if decision.symbol.strip().upper() == symbol.upper():
                decision = decision.model_copy(update={"symbol": symbol})
            else:
                reason = (f"symbol mismatch: decision names {decision.symbol!r}, "
                          f"pipeline is on {symbol!r} — rejected")
                log.warning("%s", reason)
                summary["analyzed"].append(symbol)
                summary["rejected"].append(f"{symbol}: {reason}")
                journal.record(JournalEntry(
                    symbol=symbol, fundamental_report=fundamental,
                    technical_report=technical, decision=decision, note=reason,
                ))
                continue

        verdict = risk.evaluate(decision, snap, account, limits, ctx)
        if verdict.symbol != symbol:  # defence in depth: never order what was not judged
            raise RuntimeError(f"risk verdict for {verdict.symbol!r} used for order on {symbol!r}")
        entry = JournalEntry(
            symbol=symbol, fundamental_report=fundamental, technical_report=technical,
            decision=decision, verdict=verdict,
        )
        summary["analyzed"].append(symbol)

        if verdict.approved:
            side = "buy" if verdict.action == "BUY" else "sell"
            order_id = broker.submit_order(symbol, verdict.qty, side)
            if broker.sends_orders:
                journal.book_turnover(session, order_id, verdict.qty * snap.close)
            entry.executed = True
            entry.note = f"order {order_id}"
            if not broker.sends_orders:
                pass  # dry run: never touch the real journal's entries or stops
            elif side == "buy":
                # Provisional entry price (last close); reconciled to the real
                # fill from the broker at the start of the next cycle.
                journal.set_entry_price(symbol, snap.close)
                if verdict.stop_price is not None:
                    in_force = journal.set_stop(symbol, verdict.stop_price)
                    if in_force != verdict.stop_price:
                        entry.note += (f"; frozen stop kept at {in_force} (tighten-only, "
                                       f"this lot's own stop {verdict.stop_price})")
            else:
                journal.clear_entry(symbol)
            summary["orders"].append(f"{side.upper()} {symbol} x{verdict.qty}")
        else:
            summary["rejected"].append(f"{symbol}: {verdict.reasons[-1] if verdict.reasons else 'n/a'}")

        journal.record(entry)

    log.info("Cycle done: %s", summary)
    return summary
