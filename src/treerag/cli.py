"""Command-line entry point for one TreeRAG search."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from treerag import TreeRAGConfig, TreeRAGMode, treerag_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Navigate a TreeRAG corpus tree")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("quick", "thorough"), default="thorough")
    parser.add_argument("--tree", help="Path to corpus_tree.json")
    parser.add_argument("--ollama-url")
    parser.add_argument("--model")
    args = parser.parse_args()

    config = TreeRAGConfig.from_env().with_mode(TreeRAGMode(args.mode))
    updates = {}
    if args.tree:
        updates["tree_path"] = Path(args.tree)
    if args.ollama_url:
        updates["ollama_url"] = args.ollama_url
    if args.model:
        updates["model"] = args.model
    if updates:
        config = replace(config, **updates)
        config.validate()
    result = treerag_search(args.question, config)
    print(
        json.dumps(
            {
                "answer": result.answer,
                "sources": [item.source_path for item in result.evidence],
                "elapsed_seconds": result.elapsed_seconds,
                "llm_calls": result.llm_calls,
                "mode": result.mode.value,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
