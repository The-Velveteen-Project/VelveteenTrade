"""Command-line interface.

  python -m velveteentrade doctor          # check credentials, data, LLM
  python -m velveteentrade run --dry-run   # full cycle, no orders sent
  python -m velveteentrade run             # full cycle, orders to paper account
  python -m velveteentrade positions       # current account snapshot
  python -m velveteentrade journal         # last decisions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .broker import AlpacaBroker, DryRunBroker
from .config import load_settings
from .data import MarketData
from .llm import StructuredLLM
from .memory import Journal
from .pipeline import run_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("velveteentrade")


def _broker(settings):
    if not settings.alpaca_key or not settings.alpaca_secret:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    return AlpacaBroker(settings.alpaca_key, settings.alpaca_secret, paper=settings.alpaca_paper)


def cmd_doctor(settings) -> None:
    ok = True
    print(f"Config: {len(settings.universe)} symbols in universe, provider={settings.models.provider} "
          f"(analyst={settings.models.analyst}, executive={settings.models.executive}), "
          f"paper={settings.alpaca_paper}")
    try:
        broker = _broker(settings)
        acct = broker.account()
        print(f"[OK] Alpaca account — equity ${acct.equity:,.2f}, cash ${acct.cash:,.2f}, "
              f"{len(acct.positions)} positions")
    except SystemExit:
        raise
    except Exception as exc:
        ok = False
        print(f"[FAIL] Alpaca trading API: {exc}")
    try:
        data = MarketData(settings.alpaca_key, settings.alpaca_secret, settings.cache_dir)
        df = data.daily_bars("SPY", 30)
        print(f"[OK] Market data — SPY: {len(df)} daily bars, last close {df['close'].iloc[-1]:.2f}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] Market data: {exc}")
    try:
        from pydantic import BaseModel

        class Ping(BaseModel):
            message: str

        llm = StructuredLLM(settings.models.provider)
        pong = llm.complete(settings.models.analyst, "Reply with a one-word message.", "ping", Ping)
        print(f"[OK] LLM ({settings.models.provider}/{settings.models.analyst}) — replied: {pong.message!r}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] LLM: {exc}")
    print("\nAll checks passed — ready to run." if ok else "\nFix the failures above before running.")
    sys.exit(0 if ok else 1)


def cmd_advisor(settings) -> None:
    from .advisor import run_onboarding

    llm = StructuredLLM(settings.models.provider)
    run_onboarding(settings, llm)


def cmd_profile(settings) -> None:
    from . import profile as profile_mod

    prof = profile_mod.load(settings.profile_path)
    if prof is None:
        print("Sin perfil. Corre: python -m velveteentrade advisor")
        return
    limits = profile_mod.limits_for(prof)
    eff = profile_mod.effective_tolerance(prof)
    print(f"Perfil: {prof.name} | tolerancia declarada {prof.tolerance}/5 → efectiva {eff}/5")
    print(f"Horizonte: {prof.horizon_years} años | experiencia: {prof.experience} | "
          f"fondo de emergencia: {'sí' if prof.has_emergency_fund else 'NO'}")
    print(f"Núcleo ETF: {profile_mod.etf_core_weight(prof):.0%}")
    print(f"Límites activos: pos máx {limits.max_position_pct:.0%}, {limits.max_positions} posiciones, "
          f"sector {limits.max_sector_exposure_pct:.0%}, riesgo/op {limits.per_trade_risk_pct:.1%}, "
          f"freno drawdown {limits.max_drawdown_halt_pct:.0%}, convicción mín {limits.min_conviction}")


def cmd_run(settings, dry_run: bool) -> None:
    if not settings.alpaca_paper and not dry_run:
        confirm = input("Account is LIVE (not paper). Type 'LIVE' to continue: ")
        if confirm.strip() != "LIVE":
            sys.exit("Aborted.")
    broker = _broker(settings)
    if dry_run:
        broker = DryRunBroker(broker)
    data = MarketData(settings.alpaca_key, settings.alpaca_secret, settings.cache_dir)
    llm = StructuredLLM(settings.models.provider)
    journal = Journal(settings.db_path)
    summary = run_cycle(settings, broker, data, llm, journal)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run and isinstance(broker, DryRunBroker):
        print("\n[DRY RUN] intended orders:", broker.intended_orders or "none")
    if not dry_run:
        from .bot import format_cycle_summary, notify

        notify(journal, format_cycle_summary(summary, journal))


def cmd_positions(settings) -> None:
    acct = _broker(settings).account()
    print(f"Equity: ${acct.equity:,.2f} | Cash: ${acct.cash:,.2f}")
    for p in acct.positions.values():
        print(f"  {p.symbol}: {p.qty} @ {p.avg_entry:.2f} (value ${p.market_value:,.2f})")


def cmd_orders(settings, cancel: str | None) -> None:
    broker = _broker(settings)
    orders = broker.open_orders()
    if not orders:
        print("Sin órdenes pendientes.")
        return
    for o in orders:
        print(f"  {o.id}  {o.side.upper():<4} {o.symbol:<6} x{o.qty}")
    if cancel:
        targets = [o for o in orders
                   if cancel == "all" or o.id == cancel or o.symbol == cancel.upper()]
        if not targets:
            print(f"Nada que cancelar para '{cancel}'.")
            return
        for o in targets:
            broker.cancel_order(o.id)
            print(f"Cancelada: {o.side.upper()} {o.symbol} x{o.qty} ({o.id})")


def cmd_flatten(settings) -> None:
    """Cancel every open order and close every position — a clean slate."""
    broker = _broker(settings)
    orders = broker.open_orders()
    acct = broker.account()
    if not orders and not acct.positions:
        print("Nada que limpiar: sin órdenes ni posiciones.")
        return
    print(f"Se cancelarán {len(orders)} órdenes y se cerrarán {len(acct.positions)} posiciones.")
    if input("Escribe SI para confirmar: ").strip().upper() != "SI":
        print("Cancelado.")
        return
    for o in orders:
        broker.cancel_order(o.id)
        print(f"  cancelada: {o.side.upper()} {o.symbol} x{o.qty:g}")
    for p in acct.positions.values():
        broker.submit_order(p.symbol, p.qty, "sell")
        print(f"  vendiendo: {p.symbol} x{p.qty:g}")
    Journal(settings.db_path)  # entries reconcile themselves next cycle
    print("Cuenta en limpio (las ventas se ejecutan cuando abra el mercado).")


def cmd_bot(settings) -> None:
    from .bot import run_bot

    run_bot(settings)


def cmd_report(settings) -> None:
    from pathlib import Path

    from . import profile as profile_mod
    from .config import ROOT
    from .report import write_report

    prof = profile_mod.load(settings.profile_path)
    out = write_report(Journal(settings.db_path),
                       prof.capital if prof else None,
                       Path(ROOT) / "docs" / "index.html")
    print(f"Track record generado: {out}")
    print("Publícalo con GitHub Pages: Settings → Pages → branch main, carpeta /docs.")
    print("Luego cada actualización es: report → git add docs → commit → push.")


def cmd_journal(settings) -> None:
    journal = Journal(settings.db_path)
    for row in journal.recent():
        d = row.get("decision") or {}
        v = row.get("verdict") or {}
        print(f"{row['ts'][:19]} {row['symbol']:<6} {d.get('action', '-'):<4} "
              f"conv={d.get('conviction', '-')} approved={v.get('approved', '-')} "
              f"executed={row['executed']}")


def cmd_rag(settings, action: str, query: str | None) -> None:
    from .rag import ingest as rag_ingest
    from .rag.corpus import CORPUS
    from .rag.store import RagStore

    if action == "download":
        rag_ingest.download(settings.corpus_dir)
    elif action == "ingest":
        store = RagStore(settings.rag_db_path)
        rag_ingest.ingest(settings.corpus_dir, store)
    elif action == "status":
        store = RagStore(settings.rag_db_path)
        done = store.doc_ids()
        for doc in CORPUS:
            mark = "✓" if doc.id in done else ("○ manual" if not doc.auto else "○")
            print(f"  [{mark}] {doc.id:<14} {doc.title}")
        print(f"Chunks: {store.count()}")
    elif action == "search":
        if not query:
            sys.exit("Usage: rag search \"your question\"")
        from .rag.retrieve import Retriever

        r = Retriever(settings.rag_db_path)
        if not r.available:
            sys.exit("RAG store empty — run 'rag download' then 'rag ingest' first.")
        for hit in r.search(query, top_k=4):
            print(f"\n--- {hit.citation} (score {hit.score:.3f})\n{hit.text[:600]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="velveteentrade")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    run_p = sub.add_parser("run")
    run_p.add_argument("--dry-run", action="store_true")
    sub.add_parser("positions")
    sub.add_parser("journal")
    rag_p = sub.add_parser("rag")
    rag_p.add_argument("action", choices=["download", "ingest", "status", "search"])
    rag_p.add_argument("query", nargs="?", default=None)
    sub.add_parser("advisor")
    sub.add_parser("profile")
    orders_p = sub.add_parser("orders")
    orders_p.add_argument("--cancel", default=None,
                          help="order id, symbol, or 'all' — cancels matching open orders")
    sub.add_parser("flatten")
    sub.add_parser("bot")
    sub.add_parser("report")
    args = parser.parse_args()

    settings = load_settings()
    from . import profile as _profile_mod

    _profile_mod.apply_profile(settings)  # personalized limits, if a profile exists

    if args.cmd == "doctor":
        cmd_doctor(settings)
    elif args.cmd == "run":
        cmd_run(settings, args.dry_run)
    elif args.cmd == "positions":
        cmd_positions(settings)
    elif args.cmd == "journal":
        cmd_journal(settings)
    elif args.cmd == "rag":
        cmd_rag(settings, args.action, args.query)
    elif args.cmd == "advisor":
        cmd_advisor(settings)
    elif args.cmd == "profile":
        cmd_profile(settings)
    elif args.cmd == "orders":
        cmd_orders(settings, args.cancel)
    elif args.cmd == "flatten":
        cmd_flatten(settings)
    elif args.cmd == "bot":
        cmd_bot(settings)
    elif args.cmd == "report":
        cmd_report(settings)


if __name__ == "__main__":
    main()
