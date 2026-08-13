"""
FetchQuest - TreeRAG hierarchical agentic search
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

Pure text helpers shared by the ranker, the judges and the answer assembly. All of them
are deterministic, allocation-light and free of network access: the lexical relevance
signal is a bag-of-words overlap on purpose, because embedding every chunk turned one
question into thousands of Ollama round-trips in the source benchmark.
"""

import json
import re

# Structural words carry no topical signal for a lexical overlap score.
_STOP = frozenset(
  "a an the of to in on for and or is are be as at by with from this that these those "
  "it its which what when how who whom where why any all each such into per via if "
  "then than also may must will shall should can could would about within between "
  "during under "
  "over".split()
)

# Format words are stripped from a document NAME before the name-match boost, so that
# "... Procedure.docx" does not match every question that contains the word procedure.
_NAME_STOP = frozenset(
  "procedure procedures plan plans document documents sop sops worksheet worksheets "
  "docx pdf doc form forms log logs record records file files".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_FENCE_RE = re.compile(r"^```(?:json)?|```$", flags=re.M)
_OBJECT_RE = re.compile(r"\{.*\}", flags=re.S)


def clip(text: str | None, limit: int) -> str:
  """Collapse whitespace and truncate to a character budget.

  Args:
    text: The text to clip; ``None`` is treated as empty.
    limit: Maximum number of characters to keep.

  Returns:
    The whitespace-normalised text, suffixed with an ellipsis when it was truncated.
  """
  collapsed = re.sub(r"\s+", " ", text or "").strip()
  if len(collapsed) <= limit:
    return collapsed
  return collapsed[:limit] + " …"


def full(text: str | None) -> str:
  """Normalise whitespace without ever truncating.

  Args:
    text: The text to normalise; ``None`` is treated as empty.

  Returns:
    The text with CRLF converted to LF, runs of spaces and tabs collapsed, and runs of
    three or more blank lines reduced to one blank line.
  """
  out = (text or "").replace("\r\n", "\n")
  out = re.sub(r"[ \t]+", " ", out)
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def tokens(text: str | None) -> list[str]:
  """Split text into lowercase content tokens.

  Args:
    text: The text to tokenise; ``None`` is treated as empty.

  Returns:
    The lowercase alphanumeric tokens longer than two characters that are not stop words,
    in document order and including repeats.
  """
  return [
    w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in _STOP
  ]


def stem(word: str) -> str:
  """Apply a deliberately tiny suffix stemmer.

  Maps checklist/checklists, bench/benches and clean/cleaning onto one another. It does
  not expand abbreviations - ``qc`` does not become ``quality control`` - so those stay
  misses, which is what keeps the name-match boost conservative.

  Args:
    word: A single lowercase token.

  Returns:
    The stemmed token, or the token unchanged when no suffix rule applies.
  """
  for suffix in ("ings", "ing", "ies", "es", "ed", "s"):
    if word.endswith(suffix) and len(word) - len(suffix) >= 3:
      return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
  return word


def stem_set(text: str | None) -> set[str]:
  """Stem every content token of a text.

  Args:
    text: The text to tokenise and stem; ``None`` is treated as empty.

  Returns:
    The set of distinct stemmed content tokens.
  """
  return {stem(w) for w in tokens(text)}


def name_stem_set(name: str | None) -> set[str]:
  """Stem a document or folder name, dropping structural format words first.

  Args:
    name: The candidate's name; ``None`` is treated as empty.

  Returns:
    The set of distinct stemmed, topically meaningful tokens of the name.
  """
  return {stem(w) for w in tokens(name) if w not in _NAME_STOP}


def strip_code_fence(raw: str | None) -> str:
  """Remove a Markdown code fence the model may have wrapped its JSON in.

  Args:
    raw: The raw model response; ``None`` is treated as empty.

  Returns:
    The response with leading and trailing fences removed and surrounding whitespace
    stripped.
  """
  return _FENCE_RE.sub("", (raw or "").strip()).strip()


def parse_json_object(raw: str | None) -> dict[str, object] | None:
  """Extract the first JSON object from a model response.

  The models used here reliably emit a single JSON object, sometimes preceded by a
  sentence or wrapped in a code fence. This finds that object and parses it strictly;
  it never falls back to evaluating Python literals.

  Args:
    raw: The raw model response; ``None`` is treated as empty.

  Returns:
    The decoded object, or ``None`` when the response holds no parseable JSON object.
  """
  match = _OBJECT_RE.search(strip_code_fence(raw))
  if match is None:
    return None
  try:
    decoded = json.loads(match.group(0))
  except json.JSONDecodeError:
    return None
  if not isinstance(decoded, dict):
    return None
  return decoded


def json_str(obj: dict[str, object], key: str) -> str:
  """Read a string field from a decoded model response.

  Args:
    obj: The decoded JSON object.
    key: The field to read.

  Returns:
    The field's value coerced to a stripped string, or the empty string when absent.
  """
  value = obj.get(key)
  if value is None:
    return ""
  return str(value).strip()


def json_str_list(obj: dict[str, object], key: str) -> list[str]:
  """Read a list-of-strings field from a decoded model response.

  Args:
    obj: The decoded JSON object.
    key: The field to read.

  Returns:
    The non-empty stripped string entries of the field, or an empty list when the field
    is absent or is not a list.
  """
  value = obj.get(key)
  if not isinstance(value, list):
    return []
  return [s for s in (str(v).strip() for v in value) if s]


def word_count(text: str | None) -> int:
  """Count whitespace-separated words.

  Args:
    text: The text to count; ``None`` is treated as empty.

  Returns:
    The number of whitespace-separated words.
  """
  return len((text or "").split())
