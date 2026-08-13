#!/usr/bin/env python3
"""Convert the fixed MultiHop-RAG sample to the benchmark harness schema.

MultiHop-RAG is public ODC-BY data.  A single option containing the gold answer
lets the existing semantic judge grade open-ended answers without exposing the
retrieval model to multiple-choice distractors.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "input" / "sample_200.json"
DEST = HERE / "questions_public.json"


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    converted = [
        {
            "qid": row["qid"],
            "stem": row["query"],
            "options": {"A": row["answer"]},
            "answers": ["A"],
            "question_type": row["question_type"],
        }
        for row in records
    ]
    DEST.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print(f"wrote {len(converted)} public questions to {DEST}")


if __name__ == "__main__":
    main()

