"""
vector_db/utils.py
==================
Shared helpers used across vector store modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def corpus_fingerprint(chunks) -> str:
    """
    Compute an MD5 hash that uniquely identifies a set of chunks.

    Used by local stores (FAISS, Chroma, LanceDB) to detect whether the
    corpus has changed since the last index run.
    Hashes each chunk's text content + source path.
    """
    hasher = hashlib.md5()
    for chunk in chunks:
        hasher.update(chunk.page_content.encode("utf-8", errors="replace"))
        src = str(chunk.metadata.get("source", ""))
        hasher.update(src.encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def save_fingerprint(fingerprint: str, path: str) -> None:
    """Persist a corpus fingerprint to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": fingerprint}))


def load_fingerprint(path: str) -> str | None:
    """Load a previously saved fingerprint; return None if not found."""
    try:
        return json.loads(Path(path).read_text()).get("fingerprint")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def corpus_changed(chunks, fingerprint_path: str) -> bool:
    """
    Return True if the corpus has changed since the last index run.

    Compares the current MD5 fingerprint against the one saved on disk.
    Returns True (needs re-index) when no fingerprint file exists.
    """
    current  = corpus_fingerprint(chunks)
    previous = load_fingerprint(fingerprint_path)
    changed  = current != previous
    if not changed:
        import logging
        logging.getLogger(__name__).info(
            "Corpus unchanged — loading existing index."
        )
    return changed
