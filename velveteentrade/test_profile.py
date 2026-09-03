from velveteentrade import advisor
from velveteentrade.config import Settings
from velveteentrade.llm import MockLLM
from velveteentrade import profile as pm
from velveteentrade.profile import RiskProfile


def base_profile(**over):
    kwargs = dict(
        name="Ana", tolerance=4, horizon_years=10, experience="BASIC",
        max_drawdown_comfort=0.20, income_stability="STABLE",
        has_emergency_fund=True, has_high_interest_debt=False, capital=200, goals="jubilación",
    )
    kwargs.update(over)
    return RiskProfile(**kwargs)


def test_stated_tolerance_capped_by_circumstances():
    assert pm.effective_tolerance(base_profile()) == 4
    assert pm.effective_tolerance(base_profile(has_emergency_fund=False)) == 2
    assert pm.effective_tolerance(base_profile(horizon_years=1)) == 2
    assert pm.effective_tolerance(base_profile(income_stability="PRECARIOUS")) == 2
    assert pm.effective_tolerance(base_profile(max_drawdown_comfort=0.05)) == 2
    assert pm.effective_tolerance(base_profile(experience="NONE")) == 3
    # Caps only lower, never raise:
    assert pm.effective_tolerance(base_profile(tolerance=1)) == 1


def test_limits_scale_monotonically_with_tolerance():
    prev = None
    for t in range(1, 6):
        limits = pm._TOLERANCE_TABLE[t]
        if prev:
            assert limits["max_position_pct"] >= prev["max_position_pct"]
            assert limits["per_trade_risk_pct"] >= prev["per_trade_risk_pct"]
            assert limits["etf_core_weight"] <= prev["etf_core_weight"]
        prev = limits


def test_limits_for_conservative_profile_are_tight():
    limits = pm.limits_for(base_profile(tolerance=5, has_emergency_fund=False))
    assert limits.max_position_pct <= 0.06
    assert limits.min_conviction >= 4


def test_profile_roundtrip_and_apply(tmp_path):
    prof = base_profile()
    path = tmp_path / "profile.yaml"
    pm.save(prof, path)
    loaded = pm.load(path)
    assert loaded == prof

    settings = Settings()
    settings.profile_path = path
    pm.apply_profile(settings)
    assert settings.risk.max_position_pct == pm.limits_for(prof).max_position_pct


def test_apply_profile_noop_without_file(tmp_path):
    settings = Settings()
    settings.profile_path = tmp_path / "missing.yaml"
    before = settings.risk.max_position_pct
    pm.apply_profile(settings)
    assert settings.risk.max_position_pct == before


def test_interview_detects_completion_marker():
    llm = MockLLM(responses={"chat": "Gracias, ya te conozco. [PERFIL_COMPLETO]"})
    transcript = []
    reply, done = advisor.interview(llm, "m", transcript, "Hola")
    assert done and "[PERFIL_COMPLETO]" not in reply
    assert len(transcript) == 2


def test_marker_with_pending_question_does_not_finalize():
    llm = MockLLM(responses={
        "chat": "Imagina que inviertes 10.000 y ves 8.500. ¿Qué harías? [PERFIL_COMPLETO]"})
    reply, done = advisor.interview(llm, "m", [], "ok")
    assert not done                      # a pending question blocks completion
    assert "[PERFIL_COMPLETO]" not in reply


def test_extract_profile_from_transcript():
    llm = MockLLM(responses={"RiskProfile": base_profile()})
    prof = advisor.extract_profile(llm, "m", [{"role": "user", "content": "hola"}])
    assert prof.name == "Ana"


def test_high_interest_debt_forces_most_conservative():
    prof = base_profile(tolerance=5, has_high_interest_debt=True)
    assert pm.effective_tolerance(prof) == 1
    limits = pm.limits_for(prof)
    assert limits.max_position_pct == 0.05
    assert limits.min_conviction == 4
