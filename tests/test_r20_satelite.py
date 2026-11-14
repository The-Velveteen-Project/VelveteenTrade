"""R20 (14 nov 2026): reparto núcleo/satélite y ciclo único.

1. Con un perfil sintético de tolerancia efectiva 1 y un capital declarado
   pequeño ($150 en estos tests), el modo "profile" deja al motor un satélite de
   10 % x $150 = $15, y con $15 la puerta no aprueba ninguna compra (tope por
   posición < $1).
   Decisión del mantenedor (12 nov): el núcleo vive en otra cuenta, así que esta
   cuenta es 100 % satélite: 20 % del total en paper y 10 % con dinero real.
2. daily.yml corría un segundo ciclo, con journal vacío, junto al de Railway.
   Ningún workflow puede volver a correr el ciclo con órdenes.

Sin red, sin llaves y sin bróker real.
"""
import math
from pathlib import Path

import pytest
import yaml

from velveteentrade import capital, risk
from velveteentrade import profile as pm
from velveteentrade.broker import AccountState, MockBroker
from velveteentrade.config import ROOT, SATELLITE_PCT_MAX, Allocation, Settings, load_settings
from velveteentrade.memory import Journal
from velveteentrade.profile import RiskProfile
from velveteentrade.schemas import ExecutiveDecision, TechnicalSnapshot

SEP = dict(mode="separate_account", total_capital=10_000.0,
           satellite_pct_paper=0.20, satellite_pct_live=0.10)


def perfil_t1(**over) -> RiskProfile:
    """Perfil sintético de tolerancia efectiva 1: la declara y nada la sube."""
    kw = dict(name="U", tolerance=1, horizon_years=10, experience="INTERMEDIATE",
              max_drawdown_comfort=0.20, income_stability="VARIABLE",
              has_emergency_fund=True, has_high_interest_debt=False, capital=150.0)
    kw.update(over)
    return RiskProfile(**kw)


def journal(tmp_path, name="j.db") -> Journal:
    return Journal(tmp_path / name)


def buy(conv=5) -> ExecutiveDecision:
    return ExecutiveDecision(symbol="X", action="BUY", conviction=conv, thesis="t",
                             invalidation="i", horizon_days=10)


def snap(px: float) -> TechnicalSnapshot:
    return TechnicalSnapshot(symbol="X", close=px, atr_14=0.02 * px, hist_cvar_95_20d=-0.05)


# --------------------------------------------------------------- configuración
def test_repo_config_encodes_maintainer_decision():
    """config.yaml del repo: núcleo en otra cuenta, 20 % paper, 10 % real."""
    s = load_settings(ROOT / "config.yaml")
    a = s.allocation
    assert a.mode == "separate_account"
    assert a.satellite_pct_paper == pytest.approx(0.20)
    assert a.satellite_pct_live == pytest.approx(0.10)
    assert a.total_capital and a.total_capital > 0


def test_missing_allocation_keeps_old_behaviour(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"risk": {"max_position_pct": 0.1}}))
    assert load_settings(cfg).allocation.mode == "profile"
    assert Allocation().satellite_capital(True) is None


@pytest.mark.parametrize("over", [
    dict(mode="otro"),
    dict(satellite_pct_paper=0.0),
    dict(satellite_pct_paper=SATELLITE_PCT_MAX + 0.01),
    dict(satellite_pct_live=0.5),
    dict(satellite_pct_live=float("nan")),
    dict(satellite_pct_paper=True),
    dict(total_capital=None),
    dict(total_capital=0),
    dict(total_capital=-5),
    dict(total_capital=float("nan")),
    dict(total_capital=float("inf")),
    dict(total_capital="10000"),
])
def test_invalid_allocation_stops_startup(over):
    with pytest.raises(ValueError):
        Allocation(**{**SEP, **over})


def test_thousands_dot_in_total_capital_stops_startup(tmp_path):
    """`total_capital: 10.000` (punto de miles) es el float 10.0 para YAML: S = $2
    en paper y ninguna orden llega a $1. Antes arrancaba sin error."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("allocation:\n  mode: separate_account\n  total_capital: 10.000\n"
                   "  satellite_pct_paper: 0.20\n  satellite_pct_live: 0.10\n")
    with pytest.raises(ValueError, match="punto"):
        load_settings(cfg)


def test_satellite_floor_matches_risk_and_profile_constants():
    from velveteentrade import config as cfgmod, profile as pm, risk
    assert cfgmod.SATELLITE_MIN_ORDER_USD == risk.MIN_ORDER_NOTIONAL
    assert cfgmod.SATELLITE_MIN_POSITION_PCT == min(
        row["max_position_pct"] for row in pm._TOLERANCE_TABLE.values())
    Allocation(**{**SEP, "total_capital": 400.0})        # en el piso: S real = $40, 5 % = $2
    with pytest.raises(ValueError):                      # 200: tope de $1 exacto (endurecimiento R2)
        Allocation(**{**SEP, "total_capital": 200.0})


def test_invalid_allocation_in_yaml_stops_startup(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"allocation": {"mode": "separate_account"}}))
    with pytest.raises(ValueError):
        load_settings(cfg)


def test_satellite_capital_paper_and_live():
    a = Allocation(**SEP)
    assert a.satellite_capital(paper=True) == pytest.approx(2_000.0)
    assert a.satellite_capital(paper=False) == pytest.approx(1_000.0)


# ------------------------------------------------------- reparto (capital.py)
def test_profile_mode_imposes_90_10_on_tolerance_1(tmp_path):
    """Regresión documental del modo "profile" (R4): con tolerancia 1, núcleo 90 %
    y satélite del 10 % del capital declarado ($15 con el perfil sintético)."""
    prof = perfil_t1()
    assert pm.effective_tolerance(prof) == 1
    assert pm.etf_core_weight(prof) == pytest.approx(0.90)
    b = capital.resolve(journal(tmp_path), prof, 100_000.0)
    assert b.allocated == pytest.approx(15.0)
    assert b.core_weight == pytest.approx(0.90)


def test_profile_mode_satellite_cannot_buy_anything(tmp_path):
    """Con E = $15 y tope por posición de 5 %, ninguna compra llega al mínimo de $1."""
    prof = perfil_t1()
    lim = pm.limits_for(prof)
    E = capital.resolve(journal(tmp_path), prof, 100_000.0).sizing
    for px in (20, 50, 100, 300, 600):
        v = risk.evaluate(buy(), snap(px), AccountState(equity=E, cash=E), lim, risk.CycleContext())
        assert not v.approved


def test_separate_account_is_100pct_satellite(tmp_path):
    b = capital.resolve(journal(tmp_path), perfil_t1(), 100_000.0, allocation=Allocation(**SEP))
    assert b.core_weight == 0.0
    assert b.allocated == pytest.approx(2_000.0)
    assert b.sizing == pytest.approx(2_000.0)


def test_separate_account_live_is_10pct(tmp_path):
    b = capital.resolve(journal(tmp_path), perfil_t1(), 100_000.0,
                        allocation=Allocation(**SEP), paper=False)
    assert b.allocated == pytest.approx(1_000.0)


def test_separate_account_capped_by_account_equity(tmp_path):
    b = capital.resolve(journal(tmp_path), perfil_t1(), 1_500.0, allocation=Allocation(**SEP))
    assert b.allocated == pytest.approx(1_500.0)


def test_separate_account_without_profile(tmp_path):
    """Sin perfil, el presupuesto sigue siendo el de config, no toda la cuenta."""
    b = capital.resolve(journal(tmp_path), None, 100_000.0, allocation=Allocation(**SEP))
    assert b.allocated == pytest.approx(2_000.0)


def test_profile_changes_do_not_move_the_split(tmp_path):
    """En modo separate_account el perfil fija límites, no el reparto: sin re-base."""
    j = journal(tmp_path)
    a = Allocation(**SEP)
    first = capital.resolve(j, perfil_t1(), 100_000.0, allocation=a)
    again = capital.resolve(j, perfil_t1(capital=3_000.0, tolerance=5), 100_000.0, allocation=a)
    assert first.rebased and not again.rebased
    assert again.allocated == pytest.approx(2_000.0)


def test_switching_mode_or_paper_to_live_rebases(tmp_path):
    """B6c: cambiar de modo, o pasar de paper a real, abre una base nueva y vacía la ventana."""
    j = journal(tmp_path)
    capital.resolve(j, perfil_t1(), 100_000.0)                                   # 90/10, $15
    j.update_window_peak("2026-11-13", 15.0, 126)
    b1 = capital.resolve(j, perfil_t1(), 100_000.0, allocation=Allocation(**SEP))
    assert b1.rebased and b1.allocated == pytest.approx(2_000.0)
    assert capital.current_base(j)["fp"] == {"capital": 2_000.0, "core": 0.0, "paper": True}
    b2 = capital.resolve(j, perfil_t1(), 100_000.0, allocation=Allocation(**SEP), paper=False)
    assert b2.rebased and b2.allocated == pytest.approx(1_000.0)


def test_separate_account_gate_buys_within_satellite(tmp_path):
    """Con $2.000 y los límites del perfil (5 % por posición) la puerta sí compra,
    y nunca más que el tope por posición del satélite."""
    prof = perfil_t1()
    lim = pm.limits_for(prof)
    E = capital.resolve(journal(tmp_path), prof, 100_000.0, allocation=Allocation(**SEP)).sizing
    approved = 0
    for px in (20, 50, 100, 300, 600):
        v = risk.evaluate(buy(), snap(px), AccountState(equity=E, cash=E), lim, risk.CycleContext())
        if v.approved:
            approved += 1
            assert v.qty * px <= lim.max_position_pct * 2_000.0 + 1e-9
    assert approved == 5


def test_pipeline_uses_allocation(tmp_path):
    """Ciclo completo (mock): la base que registra el pipeline es la del satélite separado."""
    from velveteentrade.pipeline import run_cycle

    from test_pipeline_dry import EchoLLM, FakeData
    from velveteentrade.schemas import AnalystReport

    settings = Settings(universe=["AAA"], candidates_per_cycle=1,
                        db_path=tmp_path / "j.db", cache_dir=tmp_path,
                        allocation=Allocation(**SEP))
    settings.profile_path = tmp_path / "profile.yaml"
    pm.save(perfil_t1(), settings.profile_path)
    broker = MockBroker(equity=100_000)
    llm = EchoLLM(responses={
        "AnalystReport": AnalystReport(symbol="AAA", stance="BULLISH",
                                       summary="Strong.", data_quality="FULL"),
        "ExecutiveDecision": ExecutiveDecision(symbol="AAA", action="BUY", conviction=5,
                                               thesis="t", invalidation="i", horizon_days=15),
    })
    j = Journal(settings.db_path)
    run_cycle(settings, broker, FakeData(), llm, j)
    base = capital.current_base(j)
    assert base["allocated"] == pytest.approx(2_000.0)
    assert base["fp"]["core"] == 0.0
    assert broker.orders, "con $2.000 el satélite sí debe poder comprar"
    lim = pm.limits_for(perfil_t1())
    for sym, qty, side in broker.orders:
        assert side == "buy"
        close = float(FakeData().daily_bars(sym, settings.history_days)["close"].iloc[-1])
        assert qty * close <= lim.max_position_pct * 2_000.0 + 1e-6


# --------------------------------------------------------------- textos
def test_describe_split_texts():
    from velveteentrade.cli import describe_split

    s = Settings(allocation=Allocation(**SEP))
    s.alpaca_paper = True
    assert "100 % satélite" in describe_split(perfil_t1(), s)
    assert "$2,000.00" in describe_split(perfil_t1(), s)
    s.alpaca_paper = False
    assert "$1,000.00" in describe_split(perfil_t1(), s)
    old = Settings()
    assert describe_split(perfil_t1(), old).startswith("Núcleo ETF: 90%")


def test_estado_does_not_call_cash_a_core(tmp_path):
    from velveteentrade.bot import format_estado

    j = journal(tmp_path)
    capital.resolve(j, perfil_t1(), 100_000.0, allocation=Allocation(**SEP))
    wc = capital.working_capital(Allocation(**SEP), perfil_t1(), paper=True)
    txt = format_estado(MockBroker(equity=100_000), j, wc)
    assert "no lo opera" in txt and "núcleo de ETF" not in txt
    j2 = journal(tmp_path, "k.db")
    capital.resolve(j2, perfil_t1(), 100_000.0)
    assert "núcleo de ETF" in format_estado(MockBroker(equity=100_000), j2, 150.0)


def test_working_capital_matches_the_satellite_the_engine_operates(tmp_path):
    """Hallazgo 11 del endurecimiento: con separate_account, /estado, /informe,
    el chat y el track record mostraban el capital del perfil junto a un
    satélite de $2.000. La cifra de capital de trabajo es ahora la del motor."""
    from velveteentrade.bot import format_estado, format_informe
    from velveteentrade.report import generate

    a = Allocation(**SEP)
    wc = capital.working_capital(a, perfil_t1(), paper=True)
    assert wc == pytest.approx(2_000.0)
    assert capital.working_capital(a, None, paper=False) == pytest.approx(1_000.0)
    assert capital.working_capital(None, perfil_t1()) == 150.0            # modo profile
    assert capital.working_capital(Allocation(), perfil_t1()) == 150.0

    j = journal(tmp_path)
    antes = format_estado(MockBroker(equity=100_000), j, wc)               # sin base aún
    assert "$2,000.00" in antes and "$150.00" not in antes
    assert "Capital de trabajo: $2,000.00" in format_informe(MockBroker(equity=100_000), j, wc)
    assert "capital de trabajo $2,000.00" in generate(j, wc)
    b = capital.resolve(j, perfil_t1(), 100_000.0, allocation=a)
    despues = format_estado(MockBroker(equity=100_000), j, wc)
    assert b.allocated == pytest.approx(wc) and "$150.00" not in despues


# ------------------------------------------------------- ciclo único (daily.yml)
WORKFLOWS = ROOT / ".github" / "workflows"


def _steps():
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        doc = yaml.safe_load(wf.read_text()) or {}
        for job in (doc.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                yield wf.name, step


def test_no_workflow_runs_the_trading_cycle():
    assert not (WORKFLOWS / "daily.yml").exists()
    for name, step in _steps():
        cmd = str(step.get("run", ""))
        assert "velveteentrade run" not in cmd or "--dry-run" in cmd, name
        assert "velveteentrade bot" not in cmd, name


def test_no_workflow_reads_broker_or_llm_secrets():
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        text = wf.read_text()
        for secret in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "OPENAI_API_KEY",
                       "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"):
            assert f"secrets.{secret}" not in text, (wf.name, secret)


def test_run_refuses_on_github_actions(monkeypatch):
    from velveteentrade import cli

    def boom(*a, **k):
        raise AssertionError("no debe construir el bróker")

    monkeypatch.setattr(cli, "_broker", boom)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(Settings(), dry_run=False)
    assert "R20" in str(exc.value)


def test_paper_to_live_rebases_even_if_satellite_amount_matches(tmp_path):
    """Endurecimiento R3: la huella guarda si la cuenta es paper o real.

    Con total 10 000 en paper (20 %) y 20 000 en real (10 %) el satélite vale
    2 000 en los dos modos. Antes la huella solo guardaba ese producto, así que
    al pasar a real con equity 2 000 la base de la paper (equity 100 000)
    seguía en vigor: V = -96 000 y E = 0."""
    j = journal(tmp_path)
    b1 = capital.resolve(j, None, 100_000.0, allocation=Allocation(**SEP))
    assert b1.rebased and b1.allocated == pytest.approx(2_000.0)
    real = dict(SEP, total_capital=20_000.0)
    b2 = capital.resolve(j, None, 2_000.0, allocation=Allocation(**real), paper=False)
    assert b2.rebased
    assert b2.allocated == pytest.approx(2_000.0)
    assert b2.value == pytest.approx(2_000.0) and b2.sizing == pytest.approx(2_000.0)
    assert capital.current_base(j)["fp"]["paper"] is False


def test_total_capital_from_service_variable(tmp_path, monkeypatch):
    """Endurecimiento R3: con dinero real el total real no tiene que ir al
    config.yaml del repo público. SATELLITE_TOTAL_CAPITAL (variable de servicio
    de Railway) manda sobre config.yaml; sin ella rige config.yaml."""
    monkeypatch.delenv("SATELLITE_TOTAL_CAPITAL", raising=False)
    assert load_settings().allocation.total_capital == pytest.approx(10_000.0)
    monkeypatch.setenv("SATELLITE_TOTAL_CAPITAL", "55555")
    s = load_settings()
    assert s.allocation.total_capital == pytest.approx(55_555.0)
    assert s.allocation.satellite_capital(paper=False) == pytest.approx(5_555.5)
    for malo in ("20.000", "1,5", "abc", "-5", "0"):
        monkeypatch.setenv("SATELLITE_TOTAL_CAPITAL", malo)
        with pytest.raises(ValueError):
            load_settings()


def test_real_money_requires_the_service_variable(monkeypatch):
    """Endurecimiento R4: con ALPACA_PAPER=false y separate_account, sin
    SATELLITE_TOTAL_CAPITAL el arranque se detiene en vez de usar el 10000
    provisional de config.yaml. Antes, perder la variable un día re-basaba
    (S = 1000 en vez de 2000) y recuperarla re-basaba otra vez: el pico del
    freno partía de lo que hubiera y la caída acumulada se perdía."""
    monkeypatch.delenv("SATELLITE_TOTAL_CAPITAL", raising=False)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert yaml.safe_load((ROOT / "config.yaml").read_text())["allocation"]["mode"] == "separate_account"
    with pytest.raises(ValueError, match="SATELLITE_TOTAL_CAPITAL"):
        load_settings()
    monkeypatch.setenv("SATELLITE_TOTAL_CAPITAL", "20000")
    s = load_settings()
    assert s.allocation.satellite_capital(paper=False) == pytest.approx(2_000.0)
    monkeypatch.setenv("ALPACA_PAPER", "true")              # en paper rige config.yaml
    monkeypatch.delenv("SATELLITE_TOTAL_CAPITAL")
    assert load_settings().allocation.total_capital == pytest.approx(10_000.0)

