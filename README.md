# VelveteenTrade

**Multi-agent LLM swing-trading system with a deterministic risk core.**
A product of [The Velveteen Project](https://github.com/The-Velveteen-Project).

Three LLM agents — fundamental analyst, technical analyst, and an executive decision-maker — produce structured, auditable trade proposals. A deterministic risk engine sizes, gates, and can veto every one of them. Orders execute through a broker-agnostic adapter (Alpaca today; anything else is a subclass away).

> **Status: paper trading only.** This system trades a simulated account while it earns — or fails to earn — the evidence to do anything else. No performance claims are made, and none should be believed without months of live forward-testing net of costs.

## Design principles

1. **LLMs propose, code disposes.** Agents never compute indicators, never size positions, never touch the broker. Their output is a Pydantic-validated proposal; the risk gate (`risk.py`) is plain Python with hard limits no prompt can override.
2. **Each analyst sees only its own domain.** The fundamental agent gets fundamentals and headlines; the technical agent gets precomputed indicators; the executive sees only the two reports. Less context bleed, less bias, lower cost.
3. **Stops are deterministic and run first.** The stop-loss sweep executes before any LLM is consulted each cycle.
4. **Every decision is journaled** (SQLite): both reports, the decision, the risk verdict, and whether it executed — the audit trail and the raw material for post-trade reflection.
5. **Provider-agnostic LLM layer.** `LLM_PROVIDER=openai|anthropic` — switching is an env var, not a refactor.
6. **Broker-agnostic execution.** The XTB lesson: brokers kill APIs overnight. Nothing above `BrokerAdapter` knows which broker is underneath.

## Architecture

```
scheduler (daily, after close)
        │
        ▼
market data (Alpaca IEX, cached) ──► deterministic indicators + screen
        │                                       │
        ▼                                       ▼
fundamental agent ◄── fundamentals/news    technical agent
        └──────────────┬───────────────────────┘
                       ▼   two AnalystReports
               executive agent ──► ExecutiveDecision (proposal)
                       │
                       ▼
        ╔══════════════════════════════╗
        ║  RISK GATE (deterministic)   ║  position caps · conviction floor
        ║  sizes, approves, or vetoes  ║  drawdown halt · CVaR limit
        ╚══════════════╤═══════════════╝  turnover budget · ATR stops
                       ▼
               BrokerAdapter (Alpaca paper) ──► journal (SQLite)
```

## Quick start

```bash
git clone https://github.com/The-Velveteen-Project/VelveteenTrade.git
cd VelveteenTrade
pip install -r requirements.txt
cp .env.example .env         # fill in Alpaca paper keys + LLM key

python -m velveteentrade doctor        # verify credentials, data, LLM
python -m velveteentrade run --dry-run # full cycle, no orders sent
python -m velveteentrade run           # full cycle against the paper account
python -m velveteentrade positions     # account snapshot
python -m velveteentrade journal       # decision history
```

Strategy policy (universe, risk limits, models) lives in `config.yaml`. Secrets live in `.env` and are never committed.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the indicator math, every risk-gate rule, and an end-to-end pipeline dry run with a mock broker and mock LLM — no network or API keys needed.

## Scheduling

`.github/workflows/daily.yml` runs the cycle on weekdays after US market close. Add `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `OPENAI_API_KEY` as repository secrets to enable it, and commit `journal.db` storage or swap it for a hosted DB if you want history across runs (a VPS with cron is the better home once the system matures).

## Roadmap

- [x] Phase 0 — skeleton: broker adapter, data layer, indicators, risk gate, agents, journal
- [ ] Phase 1 — richer fundamentals (SEC EDGAR), event-triggered re-analysis, Telegram notifications
- [ ] Phase 2 — walk-forward backtest of the deterministic layer (≥20 bp round-trip costs)
- [ ] Phase 3 — 2–3 months of live paper trading; go/no-go criteria evaluated
- [ ] Phase 4 — real capital, small, only if the evidence supports it

## Disclaimer

Educational and research software. Nothing here is investment advice. Trading involves risk of loss; automated trading can amplify it. Do not run this against a live account without understanding every line of `risk.py`.

## Cloud deploy (Railway)

One service runs everything — the Telegram bot is also the scheduler (daily
cycle at 21:30 UTC, weekdays):

1. Push this repo to GitHub and create a Railway project from it (auto-deploys on every push).
2. Add a **volume** mounted at `/data` (journal + profile survive deploys).
3. Set service variables: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`,
   `OPENAI_API_KEY`, `LLM_PROVIDER=openai`, `TELEGRAM_BOT_TOKEN`, `DATA_DIR=/data`.
4. Commit `rag.db` (it ships with the image; it is derived, static data) — run
   `rag download` + `rag ingest` locally first.

The Next.js dashboard (future) deploys separately on Vercel and reads the same data
once it moves to a hosted DB.
