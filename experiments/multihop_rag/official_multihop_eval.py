#!/usr/bin/env python3
"""Score frozen TreeQuest outputs with the official MultiHop-RAG metrics.

The metric implementations are vendored byte-for-byte from the pinned upstream
repository. This adapter supplies their expected in-memory inputs and writes an
aggregate-only, versioned output. It never modifies an existing result.

TreeQuest is an interactive reader rather than a fixed top-k retriever. Its
ordered retrieval list is defined as the first-visit sequence of public-corpus
chunk nodes recorded by ``read_file`` trace events: the passages actually
opened by the reader. Navigation-only folder events are not passages and are
excluded. Null queries are excluded by the official retrieval protocol and
retained by the official QA protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VENDOR = HERE / "vendor" / "multihop_rag"
DEFAULT_SAMPLE = HERE / "input" / "sample_200.json"
DEFAULT_REPORT = RESULTS / "treequest_public_frozen_v0_200_20260811.json"
DEFAULT_PUBLIC_TREE = HERE / "tree_cache" / "corpus_tree.json"
DEFAULT_COLLAPSED = RESULTS / "collapsed_answers.json"
DEFAULT_ORACLE = RESULTS / "oracle_answers.json"
DEFAULT_COLLAPSED_NODES = RESULTS / "collapsed_nodes.jsonl"
DEFAULT_OUTPUT = RESULTS / "treequest_official_multihop_eval_v2_20260813.json"
SCHEMA = "treequest.multihop-rag-official-eval.v2"
UPSTREAM_COMMIT = "cde8e844af14b3012f20158abc2854fe8458212a"
SEED = 20260813


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--treequest-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--public-tree", type=Path, default=DEFAULT_PUBLIC_TREE)
    parser.add_argument("--collapsed-answers", type=Path, default=DEFAULT_COLLAPSED)
    parser.add_argument("--oracle-answers", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--collapsed-nodes", type=Path, default=DEFAULT_COLLAPSED_NODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--randomization-samples", type=int, default=200_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def normalize_identity(value: str) -> str:
    import re

    return re.sub(r"\W+", " ", value.casefold()).strip()


def index_public_tree(tree: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stack = [tree]
    while stack:
        node = stack.pop()
        by_name[str(node.get("name") or "")].append(node)
        stack.extend(node.get("children") or [])
    return by_name


def official_retrieval_inputs(
    questions: list[dict[str, Any]],
    report_by_qid: dict[str, dict[str, Any]],
    tree_by_name: dict[str, list[dict[str, Any]]],
) -> tuple[list[list[str]], list[list[str]], dict[str, Any], dict[str, list[int]]]:
    retrieved_lists: list[list[str]] = []
    gold_lists: list[list[str]] = []
    lengths: list[int] = []
    visits = duplicate_visits = nonpassage_events = 0
    unresolved_events = ambiguous_chunk_events = 0
    by_type: dict[str, list[int]] = defaultdict(list)

    for question in questions:
        if question.get("question_type") == "null_query":
            continue
        qid = str(question["qid"])
        if qid not in report_by_qid:
            raise ValueError(f"missing TreeQuest result for qid={qid}")
        row = report_by_qid[qid]
        retrieved: list[str] = []
        seen: set[str] = set()
        for event in row.get("treerag_trace") or []:
            if event.get("event") != "read_file":
                continue
            matches = tree_by_name.get(str(event.get("at") or ""), [])
            chunks = [node for node in matches if node.get("node_type") == "chunk"]
            if not chunks:
                if matches:
                    nonpassage_events += 1
                else:
                    unresolved_events += 1
                continue
            if len(chunks) != 1:
                ambiguous_chunk_events += 1
                continue
            node = chunks[0]
            node_id = str(node["node_id"])
            if node_id in seen:
                duplicate_visits += 1
                continue
            seen.add(node_id)
            passage = str(node.get("content") or node.get("summary") or "")
            if not passage.strip():
                raise ValueError(f"empty public passage for node_id={node_id}")
            retrieved.append(passage)
            visits += 1

        gold = [
            str(item.get("fact") or "")
            for item in question.get("evidence_list") or []
            if str(item.get("fact") or "").strip()
        ]
        if not gold:
            raise ValueError(f"non-null qid={qid} has no gold facts")
        retrieved_lists.append(retrieved)
        gold_lists.append(gold)
        lengths.append(len(retrieved))
        by_type[str(question["question_type"])].append(len(retrieved_lists) - 1)

    if unresolved_events or ambiguous_chunk_events:
        raise ValueError(
            "retrieval reconstruction is not exact: "
            f"unresolved={unresolved_events}, ambiguous_chunks={ambiguous_chunk_events}"
        )
    coverage = {
        "non_null_questions": len(retrieved_lists),
        "ordered_unique_passage_visits": visits,
        "duplicate_passage_visits_removed": duplicate_visits,
        "navigation_only_nonpassage_events_excluded": nonpassage_events,
        "unresolved_passage_events": unresolved_events,
        "ambiguous_passage_events": ambiguous_chunk_events,
        "retrieved_passages_per_question": {
            "minimum": min(lengths),
            "median": median(lengths),
            "mean": mean(lengths),
            "maximum": max(lengths),
            "fewer_than_4": sum(value < 4 for value in lengths),
            "fewer_than_10": sum(value < 10 for value in lengths),
        },
    }
    return retrieved_lists, gold_lists, coverage, by_type


def score_retrieval(
    official_module: Any,
    retrieved: list[list[str]],
    gold: list[list[str]],
    by_type: dict[str, list[int]],
) -> dict[str, Any]:
    strata = {}
    for question_type, indices in sorted(by_type.items()):
        strata[question_type] = {
            "n": len(indices),
            "metrics": official_module.calculate_metrics(
                [retrieved[index] for index in indices],
                [gold[index] for index in indices],
            ),
        }
    return {
        "n": len(retrieved),
        "metrics": official_module.calculate_metrics(retrieved, gold),
        "by_question_type": strata,
    }


def stored_control_retrieval_inputs(
    questions: list[dict[str, Any]],
    answers_by_qid: dict[str, dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> tuple[list[list[str]], list[list[str]], dict[str, Any], dict[str, list[int]]]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        identity = row.get("title") or row.get("path") or row.get("name") or ""
        by_identity[normalize_identity(str(identity))].append(row)

    retrieved_lists: list[list[str]] = []
    gold_lists: list[list[str]] = []
    by_type: dict[str, list[int]] = defaultdict(list)
    unresolved = ambiguous = 0
    for question in questions:
        if question.get("question_type") == "null_query":
            continue
        row = answers_by_qid[str(question["qid"])]
        passages = []
        for identity in row.get("files_read") or []:
            matches = by_identity.get(normalize_identity(str(identity)), [])
            if not matches:
                unresolved += 1
                continue
            unique_texts = {str(match.get("text") or "") for match in matches}
            if len(unique_texts) != 1:
                ambiguous += 1
                continue
            passage = next(iter(unique_texts))
            if not passage.strip():
                raise ValueError("collapsed control contains an empty retrieved passage")
            passages.append(passage)
        retrieved_lists.append(passages)
        gold_lists.append(
            [str(item.get("fact") or "") for item in question.get("evidence_list") or []]
        )
        by_type[str(question["question_type"])].append(len(retrieved_lists) - 1)
    if unresolved or ambiguous:
        raise ValueError(
            f"control retrieval mapping failed: unresolved={unresolved}, ambiguous={ambiguous}"
        )
    return retrieved_lists, gold_lists, {
        "non_null_questions": len(retrieved_lists),
        "ordered_passages": sum(map(len, retrieved_lists)),
        "unresolved_passages": unresolved,
        "ambiguous_passages": ambiguous,
    }, by_type


def oracle_retrieval_inputs(
    questions: list[dict[str, Any]],
) -> tuple[list[list[str]], list[list[str]], dict[str, Any], dict[str, list[int]]]:
    retrieved_lists = []
    gold_lists = []
    by_type: dict[str, list[int]] = defaultdict(list)
    for question in questions:
        if question.get("question_type") == "null_query":
            continue
        facts = [
            str(item.get("fact") or "")
            for item in question.get("evidence_list") or []
        ]
        retrieved_lists.append(facts)
        gold_lists.append(facts)
        by_type[str(question["question_type"])].append(len(retrieved_lists) - 1)
    return retrieved_lists, gold_lists, {
        "non_null_questions": len(retrieved_lists),
        "ordered_gold_passages_supplied_to_generator": sum(map(len, retrieved_lists)),
    }, by_type


def score_qa(
    official_module: Any,
    questions: list[dict[str, Any]],
    report_by_qid: dict[str, dict[str, Any]],
    answer_field: str,
) -> dict[str, Any]:
    grouped: dict[str, tuple[list[str], list[str]]] = {"overall": ([], [])}
    for question in questions:
        qid = str(question["qid"])
        if qid not in report_by_qid:
            raise ValueError(f"missing result for qid={qid}")
        predicted = official_module.extract_answer(
            str(report_by_qid[qid].get(answer_field) or "")
        )
        gold = str(question.get("answer") or "")
        question_type = str(question["question_type"])
        grouped.setdefault(question_type, ([], []))
        grouped["overall"][0].append(predicted)
        grouped["overall"][1].append(gold)
        grouped[question_type][0].append(predicted)
        grouped[question_type][1].append(gold)

    output = {}
    for group, (predictions, gold_answers) in sorted(grouped.items()):
        precision, recall, f1, accuracy = official_module.calculate_metrics(
            predictions, gold_answers
        )
        output[group] = {
            "n": len(predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
    return output


def matched_judge_analysis(
    rows: list[dict[str, Any]], bootstrap_samples: int, randomization_samples: int
) -> dict[str, Any]:
    tree = []
    baseline = []
    for row in rows:
        left = row.get("treerag_accuracy")
        right = row.get("qms_accuracy")
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            tree.append(float(left))
            baseline.append(float(right))
    difference = np.asarray(tree) - np.asarray(baseline)
    rng = np.random.default_rng(SEED)

    bootstrap = []
    remaining = bootstrap_samples
    while remaining:
        size = min(2_000, remaining)
        indices = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap.extend(difference[indices].mean(axis=1).tolist())
        remaining -= size
    ci_low, ci_high = np.quantile(np.asarray(bootstrap), [0.025, 0.975])

    observed = abs(float(difference.mean()))
    exceedances = completed = 0
    while completed < randomization_samples:
        size = min(2_000, randomization_samples - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(difference)))
        permuted = np.abs((signs * difference).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += size
    p_value = (exceedances + 1) / (randomization_samples + 1)

    return {
        "metric": "shared_gpt_oss_120b_joint_judge_score_0_0.5_1",
        "n": len(difference),
        "treequest_mean": mean(tree),
        "flat_hybrid_mean": mean(baseline),
        "mean_paired_difference": float(difference.mean()),
        "paired_bootstrap_95_percent_ci": [float(ci_low), float(ci_high)],
        "paired_sign_flip_randomization_p_value_two_sided": p_value,
        "wins": int(np.count_nonzero(difference > 0)),
        "ties": int(np.count_nonzero(difference == 0)),
        "losses": int(np.count_nonzero(difference < 0)),
        "bootstrap_samples": bootstrap_samples,
        "randomization_samples": randomization_samples,
        "seed": SEED,
        "comparability": "within-study matched comparison; not an official MultiHop-RAG metric",
    }


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing result: {args.output}")
    if args.public_tree.resolve() != DEFAULT_PUBLIC_TREE.resolve():
        raise ValueError("only the fixed public MultiHop-RAG tree is permitted")

    retrieval_path = VENDOR / "retrieval_evaluate.py"
    qa_path = VENDOR / "qa_evaluate.py"
    retrieval_module = load_module("official_multihop_retrieval", retrieval_path)
    qa_module = load_module("official_multihop_qa", qa_path)
    questions = read_json(args.sample)
    report_payload = read_json(args.treequest_report)
    rows = report_payload["results"]
    report_by_qid = {str(row["qid"]): row for row in rows}
    collapsed_rows = read_json(args.collapsed_answers)
    oracle_rows = read_json(args.oracle_answers)
    collapsed_by_qid = {str(row["qid"]): row for row in collapsed_rows}
    oracle_by_qid = {str(row["qid"]): row for row in oracle_rows}
    if len(questions) != 200 or len(rows) != 200 or len(report_by_qid) != 200:
        raise ValueError("the frozen evaluation requires exactly 200 unique questions/results")
    if len(collapsed_by_qid) != 200 or len(oracle_by_qid) != 200:
        raise ValueError("both completed controls must contain 200 unique results")

    public_tree = read_json(args.public_tree)
    retrieved, gold, coverage, by_type = official_retrieval_inputs(
        questions, report_by_qid, index_public_tree(public_tree)
    )
    collapsed_retrieved, collapsed_gold, collapsed_coverage, collapsed_by_type = (
        stored_control_retrieval_inputs(
            questions, collapsed_by_qid, read_jsonl(args.collapsed_nodes)
        )
    )
    oracle_retrieved, oracle_gold, oracle_coverage, oracle_by_type = (
        oracle_retrieval_inputs(questions)
    )
    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "MultiHop-RAG",
            "sample": "balanced frozen 200-question sample, seed 20260806",
            "sample_counts": {
                "comparison_query": 50,
                "inference_query": 50,
                "temporal_query": 50,
                "null_query": 50,
            },
            "external_comparability": (
                "metric implementation is official; sample distribution differs from the "
                "full natural-prevalence dataset and must be disclosed"
            ),
        },
        "official_implementation": {
            "repository": "https://github.com/yixuantt/MultiHop-RAG",
            "commit": UPSTREAM_COMMIT,
            "retrieval_evaluate_sha256": sha256(retrieval_path),
            "qa_evaluate_sha256": sha256(qa_path),
        },
        "inputs": {
            "sample_sha256": sha256(args.sample),
            "treequest_report_sha256": sha256(args.treequest_report),
            "public_tree_sha256": sha256(args.public_tree),
            "collapsed_answers_sha256": sha256(args.collapsed_answers),
            "oracle_answers_sha256": sha256(args.oracle_answers),
            "collapsed_nodes_sha256": sha256(args.collapsed_nodes),
        },
        "retrieval_adapter": {
            "ordered_unit": "first-visit read_file trace event mapped to public chunk text",
            "deduplication": "public node_id, preserving first-visit order",
            "navigation_nodes": "excluded because folders are not retrieved passages",
            "null_queries": "excluded by official retrieval protocol",
            "coverage": coverage,
        },
        "official_retrieval": {
            "treequest": score_retrieval(retrieval_module, retrieved, gold, by_type),
            "flat_hybrid": {
                "status": "unavailable",
                "reason": (
                    "the stored answer artifact records document titles, but its flat index "
                    "contains multiple distinct passages per title; exact ranked passage "
                    "identity cannot be reconstructed without replaying retrieval"
                ),
            },
            "collapsed": {
                **score_retrieval(
                    retrieval_module,
                    collapsed_retrieved,
                    collapsed_gold,
                    collapsed_by_type,
                ),
                "mapping_coverage": collapsed_coverage,
            },
            "oracle": {
                **score_retrieval(
                    retrieval_module, oracle_retrieved, oracle_gold, oracle_by_type
                ),
                "mapping_coverage": oracle_coverage,
                "interpretation": "gold-context diagnostic, not a deployable retriever",
            },
        },
        "official_qa": {
            "treequest": score_qa(qa_module, questions, report_by_qid, "treerag_answer"),
            "flat_hybrid": score_qa(qa_module, questions, report_by_qid, "qms_answer"),
            "collapsed": score_qa(
                qa_module, questions, collapsed_by_qid, "answer"
            ),
            "oracle": score_qa(qa_module, questions, oracle_by_qid, "answer"),
        },
        "matched_joint_judge": matched_judge_analysis(
            rows, args.bootstrap_samples, args.randomization_samples
        ),
        "reporting_rules": [
            "Report official retrieval and QA metrics separately from the matched joint judge.",
            "Do not call the matched joint-judge score retrieval accuracy.",
            "Do not describe the balanced 200-question sample as the full MultiHop-RAG test set.",
            "Preserve the original frozen TreeQuest report as an immutable input.",
            "Treat oracle retrieval as a diagnostic ceiling, not a deployable system.",
            "Do not report unavailable flat-hybrid retrieval metrics as zero.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps({
        "official_retrieval": payload["official_retrieval"]["treequest"]["metrics"],
        "official_retrieval_collapsed": payload["official_retrieval"]["collapsed"]["metrics"],
        "official_retrieval_oracle": payload["official_retrieval"]["oracle"]["metrics"],
        "official_qa": {
            name: result["overall"] for name, result in payload["official_qa"].items()
        },
        "matched_joint_judge": payload["matched_joint_judge"],
    }, indent=2))


if __name__ == "__main__":
    main()
