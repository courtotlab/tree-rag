#!/usr/bin/env python3
"""Create a reproducible flat dense index for the public MultiHop-RAG corpus."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "input" / "corpus.json"
OUT_CHUNKS = HERE / "results" / "flat_chunks.jsonl"
OUT_EMBED = HERE / "results" / "flat_embeddings.npy"
MODEL = "nomic-embed-text"
OLLAMA = "http://localhost:11434"
WINDOW_WORDS = 300
STRIDE_WORDS = 240
BATCH = 48


def chunks_for(row: dict) -> list[dict]:
    words = re.findall(r"\S+", row.get("body") or "")
    pieces = []
    if not words:
        words = [row.get("title") or ""]
    for start in range(0, len(words), STRIDE_WORDS):
        body = " ".join(words[start : start + WINDOW_WORDS])
        if not body:
            continue
        pieces.append(
            {
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "source": row.get("source") or "",
                "category": row.get("category") or "",
                "published_at": row.get("published_at") or "",
                "text": body,
            }
        )
        if start + WINDOW_WORDS >= len(words):
            break
    return pieces


def embed_batch(texts: list[str]) -> np.ndarray:
    payload = json.dumps(
        {"model": MODEL, "input": texts, "truncate": True, "keep_alive": "30m"}
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA}/api/embed", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read())
    return np.asarray(data["embeddings"], dtype=np.float32)


def main() -> None:
    HERE.joinpath("results").mkdir(exist_ok=True)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rows = [piece for document in corpus for piece in chunks_for(document)]
    with OUT_CHUNKS.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    vectors = []
    started = time.time()
    for offset in range(0, len(rows), BATCH):
        batch = rows[offset : offset + BATCH]
        texts = [
            f"Title: {r['title']}\nSource: {r['source']}\nCategory: {r['category']}\n{r['text']}"
            for r in batch
        ]
        vectors.append(embed_batch(texts))
        print(f"embedded {min(offset + BATCH, len(rows))}/{len(rows)}", flush=True)
    matrix = np.vstack(vectors)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    np.save(OUT_EMBED, matrix)
    metadata = {
        "documents": len(corpus),
        "chunks": len(rows),
        "embedding_model": MODEL,
        "window_words": WINDOW_WORDS,
        "stride_words": STRIDE_WORDS,
        "embedding_seconds": round(time.time() - started, 2),
        "dimensions": int(matrix.shape[1]),
    }
    HERE.joinpath("results", "flat_index_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

