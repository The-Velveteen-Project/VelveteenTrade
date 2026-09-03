import numpy as np

from velveteentrade import agents
from velveteentrade.rag.corpus import CORPUS
from velveteentrade.rag.ingest import chunk, clean
from velveteentrade.rag.retrieve import Retriever
from velveteentrade.rag.store import RagStore


def test_corpus_manifest_sane():
    ids = [d.id for d in CORPUS]
    assert len(ids) == len(set(ids))
    assert len(CORPUS) >= 12
    assert {d.category for d in CORPUS} == {"technical", "fundamental", "risk", "portfolio", "behavior"}
    for d in CORPUS:
        assert d.url.startswith("http") and d.filename


def test_chunking_respects_sizes():
    text = "\n\n".join(f"Paragraph {i}. " + ("Evidence sentence. " * 20) for i in range(60))
    chunks = chunk(clean(text))
    assert len(chunks) > 3
    assert all(len(c) <= 2400 + 400 for c in chunks)
    assert all(len(c) >= 200 for c in chunks)


def test_chunking_pdf_style_single_newlines():
    # pypdf output: single newlines, no blank lines — must NOT be dropped.
    text = "\n".join(f"Line {i} of the paper with enough words to matter here." for i in range(400))
    chunks = chunk(clean(text))
    assert len(chunks) >= 5
    assert sum(len(c) for c in chunks) > len(text) * 0.8


def test_chunking_one_giant_block():
    # A single huge block with no newlines at all must still be chunked, not lost.
    chunks = chunk("word " * 3000)
    assert len(chunks) >= 5


def test_store_roundtrip_and_search(tmp_path):
    store = RagStore(tmp_path / "rag.db")
    rng = np.random.default_rng(0)
    base = rng.normal(size=8).astype(np.float32)
    store.add("tsmom", "Time Series Momentum", "technical", 0, "momentum persists", base)
    store.add("kelly", "Kelly Criterion", "risk", 0, "bet sizing", rng.normal(size=8).astype(np.float32))
    store.commit()
    assert store.count() == 2
    hits = store.search(base, top_k=1)
    assert hits[0].doc_id == "tsmom"
    assert hits[0].citation == "Time Series Momentum [tsmom#0]"
    hits_risk = store.search(base, top_k=1, category="risk")
    assert hits_risk[0].doc_id == "kelly"


def test_retriever_graceful_without_db(tmp_path):
    r = Retriever(tmp_path / "missing.db")
    assert not r.available
    assert r.excerpts("anything") == []


def test_constitutions_load_into_system_prompts():
    for builder, marker in [
        (agents.technical_system, "Momentum"),
        (agents.fundamental_system, "cash flows"),
        (agents.executive_system, "Overtrading"),
    ]:
        prompt = builder()
        assert "constitution" in prompt.lower()
        assert marker in prompt
