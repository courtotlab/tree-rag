#!/usr/bin/env python3
"""Aggregate public-answer quality, retrieval recall, and cost."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def normalize(value: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", value.casefold())
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction: str, gold: str) -> float:
    pred, target = normalize(prediction).split(), normalize(gold).split()
    if not pred or not target:
        return float(pred == target)
    common = Counter(pred) & Counter(target)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def mean(rows: list[dict], key: str) -> float:
    values = [float(r.get(key) or 0) for r in rows]
    return float(np.mean(values)) if values else float("nan")


def load_answers(name: str) -> list[dict]:
    return json.loads((RESULTS / f"{name}_answers.json").read_text(encoding="utf-8"))


def tree_answers() -> list[dict]:
    report = json.loads((RESULTS / "treerag_report.json").read_text(encoding="utf-8"))
    output = []
    for row in report.get("results", []):
        output.append(
            {
                "qid": row.get("qid"),
                "answer": row.get("treerag_response") or row.get("tree_response") or row.get("response") or "",
                "elapsed_sec": row.get("seconds") or 0,
                "llm_calls": row.get("llm_calls") or 0,
                "input_tokens": row.get("in_tokens") or 0,
                "output_tokens": row.get("out_tokens") or 0,
                "semantic_score": row.get("treerag_accuracy"),
            }
        )
    return output


def summarize(name: str, rows: list[dict], gold: dict[str, dict]) -> dict:
    valid = [r for r in rows if r.get("qid") in gold]
    em = [float(normalize(r.get("answer") or "") == normalize(gold[r["qid"]]["answer"])) for r in valid]
    f1 = [token_f1(r.get("answer") or "", gold[r["qid"]]["answer"]) for r in valid]
    result = {
        "system": name,
        "n": len(valid),
        "exact_match": float(np.mean(em)),
        "token_f1": float(np.mean(f1)),
        "seconds": mean(valid, "elapsed_sec"),
        "calls": mean(valid, "llm_calls"),
    }
    if any(r.get("semantic_score") is not None for r in valid):
        result["semantic_score"] = mean(valid, "semantic_score")
    if any("recall_at_5" in r for r in valid):
        result["recall_at_5"] = mean(valid, "recall_at_5")
        result["recall_at_10"] = mean(valid, "recall_at_10")
    return result


def main() -> None:
    sample = json.loads((HERE / "input" / "sample_200.json").read_text(encoding="utf-8"))
    gold = {row["qid"]: row for row in sample}
    systems = {
        "TreeRAG": tree_answers(),
        "Flat hybrid": load_answers("flat"),
        "Collapsed tree": load_answers("collapsed"),
        "Oracle context": load_answers("oracle"),
    }
    summary = [summarize(name, rows, gold) for name, rows in systems.items()]
    (RESULTS / "public_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in summary:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

