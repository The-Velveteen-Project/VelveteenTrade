"""Telegram interface: the Advisor conversation and the daily reports, in chat.

Run alongside (or instead of) the CLI:

    python -m velveteentrade bot

Commands:
    /start   — begin (or restart) the advisor interview
    /plan    — re-explain your saved plan
    /estado  — account, positions, pending orders in plain language
    /ciclo   — run one trading cycle now and report it
    /ayuda   — command list

Design: this file is a thin transport. All intelligence lives in advisor.py,
pipeline.py and profile.py — the same functions the CLI uses. The bot only
answers the chat that owns the .env token's deployment (first /start binds it).
"""
from __future__ import annotations

import logging
import os

from . import profile as profile_mod
from .advisor import explain_plan, extract_profile, interview
from .broker import AlpacaBroker
from .config import Settings
from .data import MarketData
from .llm import StructuredLLM
from .memory import Journal
from .pipeline import run_cycle

log = logging.getLogger(__name__)

AYUDA = (
    "Comandos:\n"
    "/start — entrevista del asesor (crea o rehace tu perfil)\n"
    "/plan — tu plan explicado\n"
    "/estado — cuenta, posiciones y órdenes\n"
    "/informe — informe del día: P&L, posiciones y decisiones\n"
    "/ciclo — correr un ciclo de trading ahora\n"
    "/ayuda — esta lista"
)


def format_informe(broker, journal: Journal, capital: float | None) -> str:
    """Deterministic daily report — real numbers, no hallucinated brackets."""
    from datetime import date

    acct = broker.account()
    hist = journal.equity_history()
    today = date.today().isoformat()
    lines = [f"📋 Informe — {today}"]

    if hist:
        eq0, eq_now = hist[0][1], hist[-1][1]
        pnl = eq_now - eq0
        pct = (pnl / eq0 * 100) if eq0 else 0.0
        lines.append(f"Capital de trabajo: {_fmt_money(eq_now)} "
                     f"({'+' if pnl >= 0 else ''}{_fmt_money(pnl)}, {pct:+.2f}% desde el inicio)")
    elif capital:
        lines.append(f"Capital de trabajo: {_fmt_money(capital)} (sin ciclos registrados aún)")

    if acct.positions:
        lines.append("\nPosiciones:")
        for p in acct.positions.values():
            pnl_p = p.market_value - p.qty * p.avg_entry
            lines.append(f"  {p.symbol}: {p.qty:g} @ {_fmt_money(p.avg_entry)} → "
                         f"{_fmt_money(p.market_value)} ({'+' if pnl_p >= 0 else ''}{_fmt_money(pnl_p)})")
    else:
        lines.append("\nSin posiciones abiertas.")

    pending = broker.open_orders()
    if pending:
        lines.append("Órdenes en cola: " +
                     ", ".join(f"{o.side.upper()} {o.symbol} x{o.qty:g}" for o in pending))

    todays = [r for r in journal.recent(60) if r["ts"][:10] == today]
    if todays:
        lines.append("\nDecisiones de hoy:")
        for r in todays:
            d = r.get("decision") or {}
            v = r.get("verdict") or {}
            mark = "✅" if r.get("executed") else ("✋" if v and not v.get("approved") else "·")
            extra = f" conv {d.get('conviction')}/5" if d.get("conviction") else ""
            lines.append(f"  {mark} {r['symbol']} {d.get('action', v.get('action', ''))}{extra}")
            if r.get("executed") and d.get("thesis"):
                first_sentence = d["thesis"].split(". ")[0][:180]
                lines.append(f"     {first_sentence}.")
    else:
        lines.append("\nHoy no hubo ciclo todavía (corre a las 21:30 UTC, o /ciclo).")

    lines.append("\nRecuerda: paper trading, fase de evidencia. Sin promesas — solo historial.")
    return "\n".join(lines)


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def format_estado(broker, journal: Journal, capital: float | None) -> str:
    acct = broker.account()
    lines = [f"Equity de la cuenta demo: {_fmt_money(acct.equity)}"]
    if capital:
        lines.append(f"Tu capital de trabajo: {_fmt_money(capital)} (el sistema opera solo con esto)")
    if acct.positions:
        lines.append("\nPosiciones:")
        for p in acct.positions.values():
            pnl = p.market_value - p.qty * p.avg_entry
            lines.append(f"  {p.symbol}: {p.qty:g} @ {_fmt_money(p.avg_entry)} → "
                         f"{_fmt_money(p.market_value)} ({'+' if pnl >= 0 else ''}{_fmt_money(pnl)})")
    else:
        lines.append("Sin posiciones abiertas.")
    pending = broker.open_orders()
    if pending:
        lines.append("\nÓrdenes pendientes:")
        lines.extend(f"  {o.side.upper()} {o.symbol} x{o.qty:g}" for o in pending)
    return "\n".join(lines)


def format_cycle_summary(summary: dict, journal: Journal) -> str:
    lines = ["Ciclo completado."]
    if summary.get("stops"):
        lines.append("🛑 Stops ejecutados: " + ", ".join(summary["stops"]))
    if summary.get("orders"):
        lines.append("\nÓrdenes:")
        recent = {r["symbol"]: r for r in journal.recent(30)}
        for order in summary["orders"]:
            lines.append(f"  ✅ {order}")
            sym = order.split()[1]
            d = (recent.get(sym) or {}).get("decision") or {}
            if d:
                lines.append(f"     Tesis: {d.get('thesis', '')}")
                lines.append(f"     Invalidación: {d.get('invalidation', '')}")
                cites = d.get("canon_citations") or []
                if cites:
                    lines.append(f"     Evidencia: {', '.join(cites)}")
    else:
        lines.append("Sin órdenes nuevas — el sistema no encontró nada que superara sus reglas. "
                     "No operar también es una decisión.")
    if summary.get("rejected"):
        lines.append("\nDescartes del gate de riesgo:")
        lines.extend(f"  ✋ {r}" for r in summary["rejected"][:6])
    if summary.get("pending_skipped"):
        lines.append("\nCon orden en vuelo (no se tocan): " + ", ".join(summary["pending_skipped"]))
    return "\n".join(lines)


def run_bot(settings: Settings) -> None:
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import (Application, CommandHandler, ContextTypes,
                              MessageHandler, filters)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env")

    llm = StructuredLLM(settings.models.provider)
    journal = Journal(settings.db_path)
    interviews: dict[int, list[dict]] = {}       # chat_id -> transcript in progress
    pending_finalize: dict[int, list[dict]] = {}  # completed transcripts awaiting plan
    chats: dict[int, list[dict]] = {}             # chat_id -> general-chat history

    def _broker():
        return AlpacaBroker(settings.alpaca_key, settings.alpaca_secret,
                            paper=settings.alpaca_paper)

    async def _send_long(bot, chat_id: int, text: str) -> None:
        """Telegram caps messages at 4096 chars — split on paragraph boundaries."""
        chunk, limit = "", 3800
        for para in text.split("\n\n"):
            if len(chunk) + len(para) + 2 > limit:
                if chunk:
                    await bot.send_message(chat_id, chunk)
                chunk = para[:limit]
            else:
                chunk = f"{chunk}\n\n{para}" if chunk else para
        if chunk:
            await bot.send_message(chat_id, chunk)

    def _guarded(handler):
        """No command may ever fail silently — the error reaches the user."""
        async def wrapped(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                await handler(update, ctx)
            except Exception as exc:  # noqa: BLE001
                log.exception("Handler %s failed", handler.__name__)
                try:
                    await update.message.reply_text(
                        f"Algo falló ({type(exc).__name__}: {str(exc)[:150]}). "
                        "Inténtalo de nuevo o revisa la terminal.")
                except Exception:
                    pass
        return wrapped

    async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        journal.set_float("telegram_chat_id", float(chat_id))
        interviews[chat_id] = []
        reply, _ = interview(llm, settings.models.executive, interviews[chat_id], "Hola")
        await update.message.reply_text(
            "Soy tu asesor de VelveteenTrade. Vamos a conocernos para armar tu plan.\n\n" + reply)

    async def _finalize(chat_id: int, transcript: list[dict],
                        ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Extract profile + save + explain plan. Failure keeps the transcript for retry —
        the user's interview must never be lost to a transient LLM error."""
        import asyncio

        try:
            await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            prof = await asyncio.to_thread(extract_profile, llm,
                                           settings.models.executive, transcript)
            profile_mod.save(prof, settings.profile_path)
            await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            plan = await asyncio.to_thread(explain_plan, llm, settings.models.executive,
                                           prof, settings)
            pending_finalize.pop(chat_id, None)
            await _send_long(ctx.bot, chat_id, plan)
            await ctx.bot.send_message(
                chat_id,
                "Perfil guardado ✅ — el sistema opera con estos límites desde el próximo "
                "ciclo. /estado cuando quieras, o pregúntame lo que sea por aquí.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Finalize failed")
            pending_finalize[chat_id] = transcript
            await ctx.bot.send_message(
                chat_id,
                f"Tuve un problema generando tu plan ({type(exc).__name__}). Tu entrevista "
                "está a salvo — escribe cualquier mensaje y lo reintento.")

    async def _general_chat(chat_id: int, text: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        import asyncio

        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        history = chats.setdefault(chat_id, [])
        prof = profile_mod.load(settings.profile_path)
        context_note = (f"[Contexto: el usuario tiene perfil guardado — {prof.name}, "
                        f"tolerancia efectiva {profile_mod.effective_tolerance(prof)}/5, "
                        f"capital ${prof.capital or 'n/d'}]"
                        if prof else "[Contexto: el usuario aún no tiene perfil — sugiérele /start]")
        # The chat must see the REAL system state — an advisor that improvises
        # numbers or claims "no trades today" blindly is worse than no advisor.
        try:
            snapshot = await asyncio.to_thread(
                format_informe, _broker(), Journal(settings.db_path),
                prof.capital if prof else None)
            context_note += ("\n[Estado real del sistema ahora mismo — usa ESTOS datos, "
                             "no inventes cifras ni pidas comandos:]\n" + snapshot)
        except Exception as exc:  # noqa: BLE001
            log.warning("Chat context snapshot failed: %s", exc)
            context_note += "\n[No pude leer el estado del sistema ahora — dilo si te lo preguntan.]"
        history.append({"role": "user", "content": text})
        from .advisor import CHAT_SYSTEM

        reply = await asyncio.to_thread(
            llm.chat, settings.models.executive,
            CHAT_SYSTEM + "\n" + context_note, history[-12:],
        )
        history.append({"role": "assistant", "content": reply})
        del history[:-12]
        await ctx.bot.send_message(chat_id, reply)

    async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        if chat_id in pending_finalize:
            await self_retry(chat_id, ctx)
            return
        if chat_id not in interviews:
            await _general_chat(chat_id, update.message.text, ctx)
            return
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        reply, done = interview(llm, settings.models.executive,
                                interviews[chat_id], update.message.text)
        await update.message.reply_text(reply)
        if done:
            transcript = interviews.pop(chat_id)
            await _finalize(chat_id, transcript, ctx)

    async def self_retry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        transcript = pending_finalize.get(chat_id)
        if transcript:
            await _finalize(chat_id, transcript, ctx)

    async def plan(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        import asyncio

        prof = profile_mod.load(settings.profile_path)
        if prof is None:
            await update.message.reply_text("Aún no tienes perfil — /start para crearlo.")
            return
        chat_id = update.effective_chat.id
        await update.message.reply_text("Preparando tu plan — dame un minuto ⏳")
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        text = await asyncio.to_thread(explain_plan, llm, settings.models.executive,
                                       prof, settings)
        await _send_long(ctx.bot, chat_id, text)

    async def estado(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        prof = profile_mod.load(settings.profile_path)
        await update.message.reply_text(
            format_estado(_broker(), journal, prof.capital if prof else None))

    async def informe(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        import asyncio

        prof = profile_mod.load(settings.profile_path)
        text = await asyncio.to_thread(
            format_informe, _broker(), Journal(settings.db_path),
            prof.capital if prof else None)
        await _send_long(ctx.bot, update.effective_chat.id, text)

    def _cycle_in_thread():
        """SQLite connections are thread-bound: the cycle thread opens its own
        Journal instead of borrowing the bot's main-thread connection."""
        data = MarketData(settings.alpaca_key, settings.alpaca_secret, settings.cache_dir)
        j = Journal(settings.db_path)
        try:
            return run_cycle(settings, _broker(), data, llm, j)
        finally:
            j.close()

    async def ciclo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Corriendo el ciclo — un par de minutos…")
        import asyncio

        try:
            summary = await asyncio.to_thread(_cycle_in_thread)
            await _send_long(ctx.bot, update.effective_chat.id,
                             format_cycle_summary(summary, journal))
        except Exception as exc:  # noqa: BLE001
            log.exception("Cycle failed")
            await update.message.reply_text(f"El ciclo falló: {exc}")

    async def ayuda(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(AYUDA)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", _guarded(start)))
    app.add_handler(CommandHandler("plan", _guarded(plan)))
    app.add_handler(CommandHandler("estado", _guarded(estado)))
    app.add_handler(CommandHandler("informe", _guarded(informe)))
    app.add_handler(CommandHandler("ciclo", _guarded(ciclo)))
    app.add_handler(CommandHandler("ayuda", _guarded(ayuda)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _guarded(on_text)))

    # Daily cycle, weekdays 21:30 UTC (~30 min after US close) — the bot process
    # IS the scheduler, so one cloud service runs everything.
    async def scheduled_cycle(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        import asyncio

        chat_id = journal.get_float("telegram_chat_id")
        try:
            summary = await asyncio.to_thread(_cycle_in_thread)
            if chat_id:
                await ctx.bot.send_message(int(chat_id),
                                           "📊 Ciclo diario automático\n\n"
                                           + format_cycle_summary(summary, journal))
        except Exception as exc:  # noqa: BLE001
            log.exception("Scheduled cycle failed")
            if chat_id:
                await ctx.bot.send_message(int(chat_id), f"⚠️ El ciclo diario falló: {exc}")

    if app.job_queue is not None:
        import datetime as dt

        app.job_queue.run_daily(scheduled_cycle,
                                time=dt.time(21, 30, tzinfo=dt.timezone.utc),
                                days=(1, 2, 3, 4, 5))
        log.info("Ciclo diario programado: 21:30 UTC, lunes-viernes.")
    else:
        log.warning("JobQueue no disponible — instala python-telegram-bot[job-queue] "
                    "para el ciclo diario automático.")

    log.info("Bot listo — habla con él en Telegram.")
    # drop_pending_updates: a restart must not replay stale queued messages.
    app.run_polling(drop_pending_updates=True)


def notify(journal: Journal, text: str) -> None:
    """Push a message to the bound chat without the bot framework (used by CLI runs)."""
    import json
    import urllib.request

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = journal.get_float("telegram_chat_id")
    if not token or not chat_id:
        return
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": int(chat_id), "text": text[:4000]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as exc:  # notification failure must never break a cycle
        log.warning("Telegram notify failed: %s", exc)
