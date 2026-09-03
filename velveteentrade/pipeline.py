"""The daily cycle: data → analysts → executive → risk gate → broker → journal.

Structure: analysis (LLM-bound) runs in parallel threads; everything that
touches money — the risk gate and order submission — runs strictly
sequentially, in screen-rank order, so budgets are deterministic.
"""
from __future__ import annotations

import logging
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
    """One full decision cycle. Returns a summary dict for logging/notification."""
    account = broker.account()

    # ------------------------------------------- 0. Reconcile with the broker
    # Broker fills are the source of truth for entry prices (the price recorded
    # at order time was the prior close, not the actual fill).
    journal.reconcile_entries(account.positions)

    # Capital base: if the user's profile states an investment amount, the
    # engine sizes against THAT, not the (paper) account's full equity — a
    # $100k demo account trading like a $200 one, which is the point.
    from . import profile as profile_mod

    prof = profile_mod.load(settings.profile_path)
    if prof and prof.capital:
        effective_equity = min(account.equity, float(prof.capital))
        log.info("Capital base from profile: sizing against $%.2f (account equity $%.2f)",
                 effective_equity, account.equity)
        account = AccountState(equity=effective_equity, cash=account.cash,
                               positions=account.positions)
    peak_equity = journal.update_peak_equity(account.equity)
    journal.record_equity(account.equity)

    # Pending (unfilled) orders — e.g. queued while the market is closed — are
    # exposure the account() snapshot doesn't show. A symbol with an order in
    # flight is untouchable this cycle: no re-analysis, no second order.
    pending = broker.open_orders()
    pending_symbols = {o.symbol for o in pending}
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

    ctx = risk.CycleContext(peak_equity=peak_equity, sectors=sectors)
    ctx.seed_sector_exposure(account)

    # ------------------------------------------ 2. Deterministic stops FIRST
    entries = journal.entry_prices()
    for verdict in risk.check_stops(account, snaps, settings.risk, entries):
        if verdict.symbol in pending_symbols:
            log.warning("Stop hit on %s but an order is already pending — not doubling.",
                        verdict.symbol)
            continue
        order_id = broker.submit_order(verdict.symbol, verdict.qty, "sell")
        journal.clear_entry(verdict.symbol)
        journal.record(JournalEntry(symbol=verdict.symbol, verdict=verdict, executed=True,
                                    note=f"stop-loss order {order_id}"))
        summary["stops"].append(verdict.symbol)

    # ------------------------------------------------ 3. Candidate selection
    held_symbols = [s for s in account.positions if s in snaps and s not in summary["stops"]]
    screened = indicators.screen(list(snaps.values()), settings.candidates_per_cycle)
    candidates = [s for s in dict.fromkeys(held_symbols + [s.symbol for s in screened])
                  if s not in pending_symbols]

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

        verdict = risk.evaluate(decision, snap, account, settings.risk, ctx)
        entry = JournalEntry(
            symbol=symbol, fundamental_report=fundamental, technical_report=technical,
            decision=decision, verdict=verdict,
        )
        summary["analyzed"].append(symbol)

        if verdict.approved:
            side = "buy" if verdict.action == "BUY" else "sell"
            order_id = broker.submit_order(symbol, verdict.qty, side)
            entry.executed = True
            entry.note = f"order {order_id}"
            if side == "buy":
                # Provisional entry price (last close); reconciled to the real
                # fill from the broker at the start of the next cycle.
                journal.set_entry_price(symbol, snap.close)
            else:
                journal.clear_entry(symbol)
            summary["orders"].append(f"{side.upper()} {symbol} x{verdict.qty}")
        else:
            summary["rejected"].append(f"{symbol}: {verdict.reasons[-1] if verdict.reasons else 'n/a'}")

        journal.record(entry)

    log.info("Cycle done: %s", summary)
    return summary
