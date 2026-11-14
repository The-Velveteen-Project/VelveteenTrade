"""Configuration: config.yaml (strategy/universe/risk) + .env (secrets)."""
from __future__ import annotations

import math
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
    max_daily_turnover_pct: float = 0.25  # max fraction of equity traded per trading SESSION (all cycles; B2)
    max_drawdown_halt_pct: float = 0.10   # halt new BUYs beyond this peak-to-trough drawdown
    # R31: el pico del freno es el máximo de las últimas N sesiones (no el
    # histórico, que dejaba el freno puesto para siempre tras quedar en
    # efectivo). 126 ≈ medio año bursátil. Valor por defecto PROVISIONAL: lo
    # decide Carlos (config.yaml, risk.drawdown_window_sessions).
    drawdown_window_sessions: int = 126
    min_conviction: int = 3               # decisions below this are logged, not traded
    max_cvar_95_20d: float = -0.25        # reject symbols whose empirical 20d CVaR is worse than this
    max_sector_exposure_pct: float = 0.30 # max fraction of equity in one sector

    def __post_init__(self) -> None:
        check_window(self.drawdown_window_sessions)


# Cotas de la ventana del freno (R31; R17 m2 y m3).
# - Mínimo 2: con W = 1 el pico es el máximo de la sesión actual y, con un
#   ciclo diario, ninguna caída entre sesiones pone el freno (queda inerte).
#   W = 2 es el menor valor que compara contra la sesión anterior. Una cota
#   más alta (p. ej. 20) sería una decisión de política, no de corrección.
# - Máximo 2520 ≈ diez años bursátiles (252 sesiones por año). Por encima de
#   2**63 - 1 sqlite3 no liga el entero y cada ciclo lanzaba OverflowError
#   antes del barrido de stops; con la cota, el error sale al arrancar.
#   memory.py conserva session_peaks hasta este horizonte, así que cambiar W
#   dentro de [WINDOW_MIN, WINDOW_MAX] es reversible.
WINDOW_MIN = 2
WINDOW_MAX = 2520


def check_window(window: object) -> int:
    """La ventana del freno (R31) debe ser un entero en [WINDOW_MIN, WINDOW_MAX].
    Un valor inválido detiene el arranque: nunca se adivina un freno."""
    if (isinstance(window, bool) or not isinstance(window, int)
            or window < WINDOW_MIN or window > WINDOW_MAX):
        raise ValueError(
            f"drawdown_window_sessions debe ser un entero entre {WINDOW_MIN} y {WINDOW_MAX} "
            f"(recibido: {window!r})")
    return window


ALLOCATION_MODES = ("profile", "separate_account")
SATELLITE_PCT_MAX = 0.20   # §4 de la torre: el satélite nunca pasa del 20 % del total
# Cota de plausibilidad de total_capital: con el tope por posición más estricto
# de la tabla de perfiles (5 %), el satélite debe poder abrir una orden de al
# menos el mínimo del bróker (risk.MIN_ORDER_NOTIONAL = $1). Atrapa, por ejemplo,
# "10.000" escrito con punto de miles, que YAML lee como 10.0.
SATELLITE_MIN_POSITION_PCT = 0.05
SATELLITE_MIN_ORDER_USD = 1.0


@dataclass
class Allocation:
    """Reparto núcleo/satélite (R20).

    - "profile" (comportamiento anterior): el núcleo de ETF vive en ESTA cuenta
      y su peso sale de la tabla del perfil (`etf_core_weight`); el motor opera
      (1 - peso) x capital declarado.
    - "separate_account": el núcleo vive en OTRA cuenta, así que esta cuenta es
      100 % satélite. El motor opera `total_capital` x pct, con pct =
      `satellite_pct_paper` en paper y `satellite_pct_live` con dinero real.

    Un valor inválido detiene el arranque: nunca se adivina un presupuesto."""

    mode: str = "profile"
    total_capital: float | None = None
    satellite_pct_paper: float = 0.20
    satellite_pct_live: float = 0.10

    def __post_init__(self) -> None:
        if self.mode not in ALLOCATION_MODES:
            raise ValueError(f"allocation.mode debe ser uno de {ALLOCATION_MODES} (recibido: {self.mode!r})")
        for name in ("satellite_pct_paper", "satellite_pct_live"):
            pct = getattr(self, name)
            if isinstance(pct, bool) or not isinstance(pct, (int, float)) \
                    or not (0 < pct <= SATELLITE_PCT_MAX):
                raise ValueError(f"allocation.{name} debe estar en (0, {SATELLITE_PCT_MAX}] (recibido: {pct!r})")
        if self.mode == "separate_account":
            cap = self.total_capital
            if isinstance(cap, bool) or not isinstance(cap, (int, float)) \
                    or not math.isfinite(cap) or cap <= 0:
                raise ValueError("allocation.total_capital debe ser un número finito > 0 "
                                 f"con mode=separate_account (recibido: {cap!r})")
            pct = min(self.satellite_pct_paper, self.satellite_pct_live)
            if cap * pct * SATELLITE_MIN_POSITION_PCT < SATELLITE_MIN_ORDER_USD:
                piso = SATELLITE_MIN_ORDER_USD / (pct * SATELLITE_MIN_POSITION_PCT)
                raise ValueError(
                    f"allocation.total_capital = {cap!r} deja un satélite de ${cap * pct:,.2f}: "
                    f"ninguna orden llegaría al mínimo de ${SATELLITE_MIN_ORDER_USD:.0f}. "
                    f"Debe ser al menos {piso:,.0f}. ¿Escribiste los miles con punto "
                    "(10.000 se lee como 10.0)? Escríbelo sin separador: 10000.")

    def satellite_capital(self, paper: bool) -> float | None:
        """Presupuesto del satélite en USD, o None en modo "profile"."""
        if self.mode != "separate_account":
            return None
        pct = self.satellite_pct_paper if paper else self.satellite_pct_live
        return float(self.total_capital) * float(pct)


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
    allocation: Allocation = field(default_factory=Allocation)
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
    allocation = Allocation(**(raw.get("allocation") or {}))
    return Settings(
        universe=raw.get("universe", []),
        candidates_per_cycle=raw.get("candidates_per_cycle", 5),
        history_days=raw.get("history_days", 320),
        risk=risk,
        allocation=allocation,
        models=models,
    )
