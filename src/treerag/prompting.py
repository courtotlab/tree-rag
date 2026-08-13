"""Small prompt/citation compatibility layer used by the standalone package."""

from __future__ import annotations

import re
from typing import Any


DOMAIN = "organizational knowledge"
ORGANIZATION = "the organization"
DUNNO = "The requested information could not be found."


def _dedup(groups: list[list[Any]]) -> list[Any]:
    output = []
    for group in groups:
        for value in group:
            if value not in output:
                output.append(value)
    return output


def format_llm_references(
    answer: str, urls: list[str], names: list[str] | None = None
) -> str:
    """Resolve numeric citations and append a compact Markdown reference list."""
    names = names or urls
    pattern = re.compile(r"[\[【]([0-9,†L\-]+)[\]】]")
    citations = []
    for match in pattern.finditer(answer):
        ids = [group.split("†")[0] for group in match.group(1).split(",")]
        citations.append((match.group(0), ids))
    if not citations:
        return answer
    used = _dedup([ids for _, ids in citations])
    aliases = {old: str(new + 1) for new, old in enumerate(used)}
    fixed = answer
    for old, ids in citations:
        fixed = fixed.replace(old, f"[{','.join(aliases[i] for i in ids)}]")
    refs = "<br>\n".join(
        f"[{new + 1}] [{names[int(old) - 1]}]({urls[int(old) - 1]})"
        for new, old in enumerate(used)
    )
    return fixed + "\n#### References\n" + refs

