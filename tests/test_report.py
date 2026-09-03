from velveteentrade.memory import Journal
from velveteentrade.report import generate, write_report
from velveteentrade.schemas import ExecutiveDecision, JournalEntry, RiskVerdict


def seeded_journal(tmp_path, points=3):
    j = Journal(tmp_path / "j.db")
    base = 200.0
    for i in range(points):
        j.record_equity(base + i * 1.5)
        # distinct timestamps (PRIMARY KEY on ts)
        import time
        time.sleep(0.002)
    j.record(JournalEntry(
        symbol="JPM",
        decision=ExecutiveDecision(symbol="JPM", action="BUY", conviction=4,
                                   thesis="Momentum alineado.", invalidation="Cierra bajo SMA50.",
                                   horizon_days=10, canon_citations=["tsmom#2"]),
        verdict=RiskVerdict(symbol="JPM", approved=True, action="BUY", qty=0.05,
                            reasons=["BUY approved"]),
        executed=True,
    ))
    j.record(JournalEntry(
        symbol="KO",
        decision=ExecutiveDecision(symbol="KO", action="BUY", conviction=2,
                                   thesis="Débil.", invalidation="n/a", horizon_days=5),
        verdict=RiskVerdict(symbol="KO", approved=False, action="BUY",
                            reasons=["conviction 2 < minimum 4"]),
    ))
    return j


def test_report_contains_tiles_chart_and_decisions(tmp_path):
    html_out = generate(seeded_journal(tmp_path), capital=200.0)
    assert "Track Record" in html_out
    assert "$203.00" in html_out              # last equity point
    assert "eq-svg" in html_out               # chart rendered (>=2 points)
    assert "tsmom#2" in html_out              # citation visible
    assert "vetada" in html_out               # rejected decision shown — losses/vetoes included
    assert "paper trading" in html_out

    # theme tokens present for light+dark
    assert "prefers-color-scheme: dark" in html_out


def test_report_empty_journal_degrades_gracefully(tmp_path):
    j = Journal(tmp_path / "j.db")
    html_out = generate(j, capital=None)
    assert "se dibuja a partir del segundo ciclo" in html_out
    assert "Aún sin decisiones" in html_out


def test_write_report_creates_file(tmp_path):
    out = write_report(seeded_journal(tmp_path), 200.0, tmp_path / "docs" / "index.html")
    assert out.exists() and out.stat().st_size > 2000
