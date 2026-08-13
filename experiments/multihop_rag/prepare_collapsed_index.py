#!/usr/bin/env python3
"""Flatten the public summary tree and embed every node for a collapsed-tree baseline."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TREE = HERE / "tree_cache" / "corpus_tree.json"  # public MultiHop-RAG tree only
OUT_ROWS = HERE / "results" / "collapsed_nodes.jsonl"
OUT_EMBED = HERE / "results" / "collapsed_embeddings.npy"
MODEL = "nomic-embed-text"
OLLAMA = "http://localhost:11434"
BATCH = 48


def walk(node: dict) -> list[dict]:
    rows = []
    stack = [node]
    while stack:
        current = stack.pop()
        node_type = current.get("node_type") or "unknown"
        if node_type != "root":
            summary = current.get("summary") or ""
            content = current.get("content") or "" if node_type == "chunk" else ""
            text = "\n".join(x for x in (current.get("name") or "", summary, content) if x)
            if text.strip():
                rows.append(
                    {
                        "node_type": node_type,
                        "name": current.get("name") or "",
                        "path": current.get("path") or "",
                        "text": text,
                    }
                )
        stack.extend(reversed(current.get("children") or []))
    return rows


def embed(texts: list[str]) -> np.ndarray:
    payload = json.dumps(
        {"model": MODEL, "input": texts, "truncate": True, "keep_alive": "30m"}
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA}/api/embed", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return np.asarray(json.loads(response.read())["embeddings"], dtype=np.float32)


def main() -> None:
    if not TREE.exists():
        raise SystemExit(f"public tree does not exist: {TREE}")
    rows = walk(json.loads(TREE.read_text(encoding="utf-8")))
    with OUT_ROWS.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    vectors = []
    started = time.time()
    for offset in range(0, len(rows), BATCH):
        vectors.append(embed([r["text"] for r in rows[offset : offset + BATCH]]))
        print(f"embedded {min(offset + BATCH, len(rows))}/{len(rows)}", flush=True)
    matrix = np.vstack(vectors)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    np.save(OUT_EMBED, matrix)
    metadata = {
        "nodes": len(rows),
        "dimensions": int(matrix.shape[1]),
        "embedding_model": MODEL,
        "embedding_seconds": round(time.time() - started, 2),
    }
    (HERE / "results" / "collapsed_index_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

