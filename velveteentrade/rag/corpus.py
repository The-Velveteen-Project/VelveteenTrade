"""The canon manifest — v1 corpus, every entry legally accessible.

Commercial books (Murphy, Graham, Van Tharp...) are copyrighted and are NOT
ingested. The corpus below is public: papers the authors publish themselves,
public research libraries (AQR, NBER, Berkeley), and public-domain material.
`auto=False` entries are gated behind click-through (e.g. SSRN) — download
them manually into the corpus folder with the given filename.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    authors: str
    category: str  # technical | fundamental | risk | portfolio | behavior
    url: str
    filename: str
    auto: bool = True  # False → manual download required (gated)


CORPUS: list[Doc] = [
    # ------------------------------------------------------------- technical
    Doc("tsmom", "Time Series Momentum", "Moskowitz, Ooi, Pedersen (AQR)",
        "technical",
        "http://docs.lhpedersen.com/TimeSeriesMomentum.pdf",  # author-hosted; http-only server
        "time_series_momentum.pdf"),
    Doc("trends", "Trends Everywhere", "Babu, Levine, Ooi, Pedersen, Stamelos (AQR)",
        "technical",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3386035",
        "trends_everywhere.pdf", auto=False),  # SSRN click-through only
    Doc("century", "A Century of Evidence on Trend-Following Investing",
        "Hurst, Ooi, Pedersen (AQR)", "technical",
        "https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf",
        "century_trend_following.pdf"),
    Doc("lmw", "Foundations of Technical Analysis", "Lo, Mamaysky, Wang (NBER w7613)",
        "technical",
        "https://www.nber.org/system/files/working_papers/w7613/w7613.pdf",
        "foundations_technical_analysis.pdf"),
    Doc("dualmom", "Risk Premia Harvesting Through Dual Momentum", "Antonacci",
        "technical",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2042750",
        "dual_momentum.pdf", auto=False),
    # ----------------------------------------------------------- fundamental
    Doc("damodaran-val", "Valuation: Lecture Notes (all day seminar)", "Damodaran (NYU Stern)",
        "fundamental",
        "https://pages.stern.nyu.edu/~adamodar/pdfiles/country/valallday12.pdf",
        "damodaran_valuation_notes.pdf"),
    Doc("ff5", "A Five-Factor Asset Pricing Model", "Fama, French",
        "fundamental",
        "https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5f5a8f4912595c0d5f5af0f4/1599770442385/FF_Five_Factor.pdf",
        "fama_french_five_factor.pdf"),
    Doc("vme", "Value and Momentum Everywhere", "Asness, Moskowitz, Pedersen (AQR)",
        "fundamental",
        "http://docs.lhpedersen.com/ValMomEverywhere.pdf",  # author-hosted; http-only server
        "value_momentum_everywhere.pdf"),
    # ------------------------------------------------------------------ risk
    Doc("kelly", "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market",
        "Thorp", "risk",
        "https://www.edwardothorp.com/wp-content/uploads/2016/11/TheKellyCriterionAndTheStockMarket.pdf",
        "thorp_kelly_criterion.pdf"),
    Doc("voltarget", "The Impact of Volatility Targeting", "Harvey et al.",
        "risk",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538",
        "volatility_targeting.pdf", auto=False),
    # ------------------------------------------------------------- portfolio
    Doc("ifyoucan", "If You Can: How Millennials Can Get Rich Slowly", "Bernstein",
        "portfolio",
        "https://www.amazon.com/If-You-Can-Millennials-Slowly-ebook/dp/B00JCC5JKI",
        "bernstein_if_you_can.pdf", auto=False),  # no longer freely hosted — optional, buy/borrow
    Doc("sec-roadmap", "Saving and Investing: A Roadmap to Your Financial Security",
        "U.S. SEC (public domain)", "portfolio",
        "https://www.sec.gov/investor/pubs/sec-guide-to-savings-and-investing.pdf",
        "sec_saving_investing.pdf"),
    Doc("sharpe-active", "The Arithmetic of Active Management", "Sharpe",
        "portfolio",
        "https://web.stanford.edu/~wfsharpe/art/active/active.htm",
        "sharpe_arithmetic_active.html"),
    # -------------------------------------------------------------- behavior
    Doc("odean-hazard", "Trading is Hazardous to Your Wealth", "Barber, Odean",
        "behavior",
        "https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf",
        "odean_trading_hazardous.pdf"),
    Doc("odean-lose", "Just How Much Do Individual Investors Lose by Trading?",
        "Barber, Lee, Liu, Odean", "behavior",
        "https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/justhowmuchdoindividualinvestorslose_rfs_2009.pdf",
        "odean_how_much_lose.pdf"),
]


def by_id(doc_id: str) -> Doc:
    for d in CORPUS:
        if d.id == doc_id:
            return d
    raise KeyError(doc_id)
