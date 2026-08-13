"""
FetchQuest - TreeQuest hierarchical agentic search
Copyright (C) 2025 Ontario Institute for Cancer Research

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Reconciling TreeQuest's citations with the ones vector search already renders.

The two modes cite differently by construction. Vector search hands the model integer
document indices and rewrites them into a numbered reference table via
``format_llm_references``. TreeQuest hands the model full source paths, because it has no
index to hand and the path IS the document's identity in the tree.

They reconcile cleanly in one direction: a source path is resolvable to the same document
row vector search draws its URL from, so TreeQuest's ``[path/to/file.docx]`` citations are
renumbered to ``[1]``, ``[2]`` in order of first appearance and then handed to the very
same ``format_llm_references`` renderer. Both modes therefore end with an identical
numbered, clickable reference table. Neither format is mangled: the model still emits what
it is best at, and the rewrite happens once, afterwards.
"""

import re
from pathlib import Path

from loguru import logger

from treequest.prompting import format_llm_references

_PATH_CITATION_RE = re.compile(r"\[([^\]\[]+)\]")


def _lookup_urls(paths: list[str]) -> dict[str, str]:
  """Optional URL hook for host applications.

  Args:
    paths: Source document paths cited by the answer.

  Returns:
    The standalone package has no database dependency, so paths render as plain-text
    references. Host applications may monkeypatch this function or post-process the
    returned references to attach their own document URLs.
  """
  return {}


def render_citations(answer: str, doc_ids: list[str]) -> str:
  """Rewrite path citations into the numbered reference table both modes share.

  Args:
    answer: The answer text, citing documents as ``[source/path.docx]``.
    doc_ids: The source paths supplied to the answer, in the order they were shown.

  Returns:
    The answer with numbered citations and a trailing reference section, matching what
    vector search renders. When the answer cites nothing recognisable, it is returned
    unchanged.
  """
  if not answer or not doc_ids:
    return answer

  known = set(doc_ids)
  ordered: list[str] = []

  def replace(match: re.Match[str]) -> str:
    token = match.group(1).strip()
    if token not in known:
      return match.group(0)
    if token not in ordered:
      ordered.append(token)
    return f"[{ordered.index(token) + 1}]"

  rewritten = _PATH_CITATION_RE.sub(replace, answer)
  if not ordered:
    return answer

  urls = _lookup_urls(ordered)
  names = [Path(path).stem or path for path in ordered]
  resolved = [urls.get(path, "") for path in ordered]
  if not any(resolved):
    # No URL for anything cited: render the reference list as plain names, which is still
    # consistent with the vector-search layout, just without links.
    lines = "<br>\n".join(f"[{i + 1}] {name}" for i, name in enumerate(names))
    return rewritten + "\n#### References\n" + lines
  resolved = [
    url or f"#{Path(path).stem}" for url, path in zip(resolved, ordered, strict=True)
  ]
  return format_llm_references(rewritten, resolved, names)


def source_names(doc_ids: list[str]) -> list[str]:
  """Render source paths as the document names the UI shows.

  Args:
    doc_ids: Source document paths.

  Returns:
    The display name of each path, matching how vector search names a hit.
  """
  return [Path(path).stem or path for path in doc_ids]
