#!/usr/bin/env python3
"""Generate answers from flat-hybrid, collapsed-tree, or oracle public context."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SAMPLE = HERE / "input" / "sample_200.json"
LOCAL_OLLAMA = "http://localhost:11434"
REMOTE_OLLAMA = "http://localhost:11528"
EMBED_MODEL = "nomic-embed-text"
ANSWER_MODEL = "gpt-oss:120b"
RRF_K = 60
CONTEXT_CHARS = 30_000


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.n = len(texts)
        self.k1 = k1
        self.b = b
        self.lengths = np.asarray([len(tokens(t)) for t in texts], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) or 1.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            for term, count in Counter(tokens(text)).items():
                postings[term].append((index, count))
        self.postings = postings

    def score(self, query: str) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float32)
        for term in set(tokens(query)):
            posting = self.postings.get(term, [])
            df = len(posting)
            if not df:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for index, tf in posting:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avgdl)
                result[index] += idf * tf * (self.k1 + 1) / denom
        return result


def post_json(url: str, body: dict, timeout: int = 600) -> dict:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:  # network/model transient
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"request failed after retries: {last}")


def query_embedding(query: str) -> np.ndarray:
    data = post_json(
        f"{LOCAL_OLLAMA}/api/embed",
        {"model": EMBED_MODEL, "input": query, "truncate": True, "keep_alive": "30m"},
    )
    vector = np.asarray(data["embeddings"][0], dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def rrf_order(dense: np.ndarray, lexical: np.ndarray, limit: int = 200) -> np.ndarray:
    n = len(dense)
    score = np.zeros(n, dtype=np.float32)
    for values in (dense, lexical):
        order = np.argsort(-values)[: min(limit, n)]
        score[order] += 1.0 / (RRF_K + np.arange(1, len(order) + 1))
    return np.argsort(-score)


def retrieve_flat(query: str, rows: list[dict], matrix: np.ndarray, bm25: BM25) -> list[dict]:
    vector = query_embedding(query)
    return [rows[i] for i in rrf_order(matrix @ vector, bm25.score(query))]


def retrieve_collapsed(query: str, rows: list[dict], matrix: np.ndarray) -> list[dict]:
    vector = query_embedding(query)
    return [rows[i] for i in np.argsort(-(matrix @ vector))]


def context_from_rows(rows: list[dict], max_docs: int = 10) -> tuple[str, list[str]]:
    blocks, seen, sources, used = [], set(), [], 0
    for row in rows:
        path_or_title = row.get("title") or row.get("path") or row.get("name") or "source"
        key = normalize_title(path_or_title)
        if key in seen:
            continue
        seen.add(key)
        text = row.get("text") or ""
        block = f"SOURCE: {path_or_title}\n{text}"
        if used + len(block) > CONTEXT_CHARS and blocks:
            break
        blocks.append(block)
        sources.append(path_or_title)
        used += len(block)
        if len(blocks) >= max_docs:
            break
    return "\n\n".join(blocks), sources


def answer(query: str, context: str) -> tuple[str, dict]:
    prompt = (
        "Answer the question using ONLY the supplied public news sources. "
        "Give the shortest complete answer (at most 100 words). If the sources do not "
        "contain enough information, answer exactly: Insufficient information.\n\n"
        f"QUESTION:\n{query}\n\nSOURCES:\n{context}\n\nANSWER:"
    )
    started = time.time()
    response = post_json(
        f"{REMOTE_OLLAMA}/api/chat",
        {
            "model": ANSWER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": 4096},
            "think": False,
            "stream": False,
            "keep_alive": "30m",
        },
        timeout=900,
    )
    text = (response.get("message", {}).get("content") or "").strip()
    usage = {
        "elapsed_sec": round(time.time() - started, 3),
        "input_tokens": response.get("prompt_eval_count") or 0,
        "output_tokens": response.get("eval_count") or 0,
        "llm_calls": 1,
    }
    return text, usage


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def recall_at(retrieved: list[str], evidence: list[dict], k: int) -> float:
    gold = {normalize_title(e.get("title") or "") for e in evidence}
    gold.discard("")
    if not gold:
        return 1.0
    got = {normalize_title(x) for x in retrieved[:k]}
    return len(gold & got) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("flat", "collapsed", "oracle"), required=True)
    args = parser.parse_args()
    questions = json.loads(SAMPLE.read_text(encoding="utf-8"))
    out = RESULTS / f"{args.mode}_answers.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {row["qid"] for row in existing}

    rows = matrix = bm25 = None
    if args.mode == "flat":
        rows = load_jsonl(RESULTS / "flat_chunks.jsonl")
        matrix = np.load(RESULTS / "flat_embeddings.npy")
        bm25 = BM25([r["text"] for r in rows])
    elif args.mode == "collapsed":
        rows = load_jsonl(RESULTS / "collapsed_nodes.jsonl")
        matrix = np.load(RESULTS / "collapsed_embeddings.npy")

    for position, question in enumerate(questions, 1):
        if question["qid"] in done:
            continue
        if args.mode == "flat":
            ranked = retrieve_flat(question["query"], rows, matrix, bm25)
            context, sources = context_from_rows(ranked)
        elif args.mode == "collapsed":
            ranked = retrieve_collapsed(question["query"], rows, matrix)
            context, sources = context_from_rows(ranked)
        else:
            facts = [e.get("fact") or "" for e in question.get("evidence_list") or []]
            context = "\n".join(f"GOLD EVIDENCE: {fact}" for fact in facts)
            sources = [e.get("title") or "" for e in question.get("evidence_list") or []]
        response, usage = answer(question["query"], context)
        row = {
            "qid": question["qid"],
            "answer": response,
            "files_read": sources,
            "question_type": question["question_type"],
            "recall_at_5": recall_at(sources, question.get("evidence_list") or [], 5),
            "recall_at_10": recall_at(sources, question.get("evidence_list") or [], 10),
            **usage,
        }
        existing.append(row)
        out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"{args.mode}: {position}/{len(questions)} complete", flush=True)

    if args.mode == "flat":
        (HERE / "qms_answers_public.json").write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    random.seed(20260806)
    main()

