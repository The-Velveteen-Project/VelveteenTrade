"""Download, extract, chunk, embed, and store the canon corpus.

Run on a machine with normal network access (your laptop):

    python -m velveteentrade rag download   # fetch the auto-downloadable docs
    python -m velveteentrade rag ingest     # extract + chunk + embed (OpenAI)
    python -m velveteentrade rag search "when should a trend follower exit"

Embeddings use OpenAI text-embedding-3-small regardless of LLM_PROVIDER
(Anthropic offers no embeddings API), so keep OPENAI_API_KEY set even after
migrating the agents to Claude. Cost for the full corpus: well under $0.10.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .corpus import CORPUS, Doc
from .store import RagStore

log = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 64
CHUNK_CHARS = 2400        # ~600 tokens
CHUNK_OVERLAP = 300       # ~75 tokens
MIN_CHUNK_CHARS = 200


# ------------------------------------------------------------------ download
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}


def _looks_valid(path: Path) -> bool:
    """A .pdf file must actually be a PDF; blocked downloads save HTML instead."""
    if path.suffix.lower() != ".pdf":
        return path.stat().st_size > 500
    with path.open("rb") as f:
        return f.read(1024).lstrip().startswith(b"%PDF")


def download(corpus_dir: Path) -> None:
    import urllib.request

    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc in CORPUS:
        dest = corpus_dir / doc.filename
        if dest.exists():
            if _looks_valid(dest):
                print(f"[skip] {doc.filename} (already present)")
                continue
            print(f"[redo] {doc.filename} was corrupt (HTML block page) — re-downloading")
            dest.unlink()
        if not doc.auto:
            print(f"[MANUAL] {doc.title}\n         gated URL: {doc.url}\n"
                  f"         download it in your browser and save as: {dest}")
            continue
        try:
            req = urllib.request.Request(doc.url, headers=_BROWSER_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if doc.filename.endswith(".pdf") and not data.lstrip().startswith(b"%PDF"):
                print(f"[FAIL] {doc.filename}: server returned a web page, not the PDF "
                      f"(bot protection).\n       Open {doc.url} in your browser and save as {dest}")
                continue
            dest.write_bytes(data)
            print(f"[ok]   {doc.filename} ({dest.stat().st_size // 1024} KB)")
        except Exception as exc:
            print(f"[FAIL] {doc.filename}: {exc}\n       get it manually from {doc.url}")


# ------------------------------------------------------------------- extract
def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raw = path.read_text(errors="ignore")
    if path.suffix.lower() in (".html", ".htm"):
        raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
    return raw


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str) -> list[str]:
    """Paragraph-aware sliding window, ~600 tokens with ~75 token overlap.

    PDF extraction usually yields single newlines with no blank lines between
    paragraphs, so fall back to line-level splitting when blank-line paragraphs
    are scarce — otherwise a whole document reads as one giant "paragraph".
    """
    pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(pieces) < max(3, len(text) // (CHUNK_CHARS * 2)):
        pieces = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    buf = ""
    for p in pieces:
        if len(buf) + len(p) + 1 <= CHUNK_CHARS:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        tail = buf[-CHUNK_OVERLAP:] if buf and CHUNK_OVERLAP else ""
        buf = f"{tail}\n{p}" if tail else p
        while len(buf) > CHUNK_CHARS:   # oversized single piece
            chunks.append(buf[:CHUNK_CHARS])
            buf = buf[CHUNK_CHARS - CHUNK_OVERLAP:]
    if len(buf) >= MIN_CHUNK_CHARS:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# --------------------------------------------------------------------- embed
def embed_texts(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t[:8000] for t in texts[i:i + EMBED_BATCH]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return np.asarray(out, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


# -------------------------------------------------------------------- ingest
def ingest(corpus_dir: Path, store: RagStore, only: list[str] | None = None) -> None:
    done = store.doc_ids()
    for doc in CORPUS:
        if only and doc.id not in only:
            continue
        if doc.id in done and not only:
            print(f"[skip] {doc.id} (already ingested)")
            continue
        path = corpus_dir / doc.filename
        if not path.exists():
            print(f"[miss] {doc.id}: {doc.filename} not downloaded — skipping")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                with path.open("rb") as f:
                    if not f.read(1024).lstrip().startswith(b"%PDF"):
                        print(f"[bad]  {doc.id}: {doc.filename} is not a real PDF (an HTML "
                              f"block page was saved instead). Delete it, download the real "
                              f"file in your browser from {doc.url}, and re-run ingest.")
                        continue
            text = clean(extract_text(path))
            pieces = chunk(text)
            if not pieces:
                print(f"[warn] {doc.id}: no extractable text — likely a scanned or corrupt "
                      f"file. Re-download from {doc.url}")
                continue
            embeddings = embed_texts(pieces)
            store.delete_doc(doc.id)
            for seq, (piece, emb) in enumerate(zip(pieces, embeddings)):
                store.add(doc.id, doc.title, doc.category, seq, piece, emb)
            store.commit()
            print(f"[ok]   {doc.id}: {len(pieces)} chunks")
        except Exception as exc:  # one bad document must never abort the whole ingest
            print(f"[FAIL] {doc.id}: {type(exc).__name__}: {exc} — skipping this document")
    print(f"Total chunks in store: {store.count()}")
