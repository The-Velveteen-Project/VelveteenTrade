"""Configuration: config.yaml (strategy/universe/risk) + .env (secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

# Mutable state (journal, profile, cache) lives in DATA_DIR so cloud deploys
# can mount a persistent volume (e.g. Railway: DATA_DIR=/data). Defaults to
# the repo dir for local use. The RAG db ships with the repo (static data).
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT)))
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RiskLimits:
    max_position_pct: float = 0.10        # max fraction of equity in one symbol
    max_positions: int = 8
    max_gross_exposure: float = 1.0       # 1.0 = no leverage, ever
    per_trade_risk_pct: float = 0.01      # equity fraction at risk per trade (ATR stop distance)
    stop_atr_mult: float = 2.0
    max_daily_turnover_pct: float = 0.25  # max fraction of equity traded per cycle
    max_drawdown_halt_pct: float = 0.10   # halt new BUYs beyond this peak-to-trough drawdown
    min_conviction: int = 3               # decisions below this are logged, not traded
    max_cvar_95_20d: float = -0.25        # reject symbols whose empirical 20d CVaR is worse than this
    max_sector_exposure_pct: float = 0.30 # max fraction of equity in one sector


@dataclass
class Models:
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    analyst: str = ""
    executive: str = ""

    def __post_init__(self) -> None:
        defaults = {
            "openai": ("gpt-5-mini", "gpt-5"),
            "anthropic": ("claude-haiku-4-5", "claude-sonnet-4-5"),
        }
        d = defaults.get(self.provider, defaults["openai"])
        self.analyst = self.analyst or os.getenv("LLM_ANALYST_MODEL", d[0])
        self.executive = self.executive or os.getenv("LLM_EXECUTIVE_MODEL", d[1])


@dataclass
class Settings:
    universe: list[str] = field(default_factory=list)
    candidates_per_cycle: int = 5   # how many screened symbols get full LLM analysis
    history_days: int = 320
    risk: RiskLimits = field(default_factory=RiskLimits)
    models: Models = field(default_factory=Models)
    alpaca_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    db_path: Path = DATA_DIR / "journal.db"
    cache_dir: Path = DATA_DIR / ".cache"
    rag_db_path: Path = ROOT / "rag.db"
    corpus_dir: Path = ROOT / "rag_corpus"
    profile_path: Path = DATA_DIR / "profile.yaml"


def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    risk = RiskLimits(**raw.get("risk", {}))
    models = Models(**raw.get("models", {}))
    return Settings(
        universe=raw.get("universe", []),
        candidates_per_cycle=raw.get("candidates_per_cycle", 5),
        history_days=raw.get("history_days", 320),
        risk=risk,
        models=models,
    )
