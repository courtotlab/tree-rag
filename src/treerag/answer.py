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

Writing the final answer from whatever the traversal collected.

The prompt is deliberately minimal - system prompt, the house instructions, the documents,
the question - because the instructions already state everything about format, quoting,
citations and the not-found fallback. Two mechanical safeguards remain, and they correct
malformed OUTPUT rather than the answer's content: a retry when the model exhausts its
budget reasoning and returns nothing, and a citation repair when it drops the document
references the instructions require.
"""

import re
from dataclasses import dataclass, field

from treerag.prompting import DUNNO
from treerag.context import SearchContext
from treerag.prompts import ANSWER_INSTRUCTIONS, system_prompt
from treerag.text import full, word_count
from treerag.types import TreeNode

_CITATION_RE = re.compile(r"\[[^\]\[]+\]")
_PATH_CITATION_RE = re.compile(r"\[([^\]\[]+)\]")
_ANY_CITATION_RE = re.compile(r"\[[^\]]+\]")
_ANSWER_HEADING_RE = re.compile(r"ANSWER\s*:\s*(.*)", flags=re.I | re.S)
_FINAL_CHANNEL_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>", flags=re.I)
_GLUED_FINAL_RE = re.compile(r"\bassistant\s*final\b\s*[:\-]?", flags=re.I)
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", flags=re.S | re.I)
_THINK_OPEN_RE = re.compile(r"<think>.*$", flags=re.S | re.I)
_STRAY_TAG_RE = re.compile(
  r"<\|?(?:channel|start|end|message|assistant|analysis|final)\|?>", flags=re.I
)
_TOOL_JSON_RE = re.compile(r'\{\s*"(action|tool|tool_call|function|name)"\s*:')
_TOOL_PROSE_RE = re.compile(
  r"(?i)\b(use|call|invoke|run|issue)\b[^.\n]{0,40}\b(search|retrieval)\b"
  r"[^.\n]{0,20}\b(tool|function)s?\b"
)
_SEARCHING_RE = re.compile(r"(?i)^\s*search(ing)?\s+for\b")
_PERSONA_RE = re.compile(
  r"(?i)\bas\s+chatgpt\b|\bwe\s+cannot\s+actually\s+run\b|\bsimulate\b.{0,30}\btool"
)
_SECTION_HEADING_RE = re.compile(r"(?i)^(question|documents?|rules?|--- )")
_STRIP_CHARS = " \"'\n`-:•–*#"


def _citation_ids(text: str) -> list[str]:
  """Return normalized bracketed document identifiers from an answer."""
  return [match.group(1).strip() for match in _PATH_CITATION_RE.finditer(text)]


def _has_allowed_citation(text: str, doc_ids: list[str]) -> bool:
  """Report whether an answer contains at least one supplied document identifier."""
  allowed = set(doc_ids)
  return any(token in allowed for token in _citation_ids(text))


def _only_allowed_citations(text: str, doc_ids: list[str]) -> bool:
  """Require every bracketed citation in a repair to name supplied evidence."""
  cited = _citation_ids(text)
  allowed = set(doc_ids)
  return bool(cited) and all(token in allowed for token in cited)


@dataclass(slots=True)
class AnswerAttempt:
  """Diagnostics for one answer-generation attempt.

  Attributes:
    number: 1-based attempt number.
    budget: Token budget the attempt was given.
    content_len: Length of the content the model returned.
    thinking_len: Length of the reasoning it produced.
    done_reason: Why generation stopped.
    reasoning_exhausted: Whether the budget was spent reasoning and content came back
      empty.
  """

  number: int
  budget: int
  content_len: int
  thinking_len: int
  done_reason: str
  reasoning_exhausted: bool


@dataclass(slots=True)
class AnswerDiagnostics:
  """Why the answer came out the way it did.

  Attributes:
    attempts: One record per generation attempt.
    dunno_with_evidence: The model returned the not-found string despite gate-approved
      evidence, which is far more likely mechanical than real.
    words: Word count of the final answer.
    citations_repaired: A citation-repair pass was accepted.
  """

  attempts: list[AnswerAttempt] = field(default_factory=list)
  dunno_with_evidence: bool = False
  words: int = 0
  citations_repaired: bool = False


@dataclass(slots=True)
class DraftedAnswer:
  """The finished answer and the document ids it may cite.

  Attributes:
    text: The answer text, with one citation per cited document.
    doc_ids: Source paths of the documents supplied to the answer, in the order shown.
    diagnostics: How the answer was produced.
  """

  text: str
  doc_ids: list[str]
  diagnostics: AnswerDiagnostics


def dedupe_citations(text: str) -> str:
  """Collapse repeated document citations, keeping each document's last occurrence.

  The instructions ask the model to cite each document once, grouping that document's
  information together with a single reference at the end. This is the safety net for when
  it repeats a reference after every sentence anyway.

  Args:
    text: The answer text.

  Returns:
    The answer with every repeated citation removed except its last occurrence, and the
    whitespace left behind tidied up.
  """
  if not text:
    return text
  matches = list(_CITATION_RE.finditer(text))
  if not matches:
    return text
  counts: dict[str, int] = {}
  last_position: dict[str, int] = {}
  for match in matches:
    token = match.group(0)
    counts[token] = counts.get(token, 0) + 1
    last_position[token] = match.start()
  drop: list[tuple[int, int]] = [
    (m.start(), m.end())
    for m in matches
    if counts[m.group(0)] > 1 and m.start() != last_position[m.group(0)]
  ]
  if not drop:
    return text
  out: list[str] = []
  previous = 0
  for start, end in sorted(drop):
    out.append(text[previous:start])
    previous = end
  out.append(text[previous:])
  cleaned = "".join(out)
  cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
  cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
  return cleaned.strip()


def _strip_reasoning(text: str) -> str:
  """Remove reasoning channels and stray markers from a raw generation.

  When a ``final`` channel exists the answer is ONLY what sits inside it; every other
  channel is reasoning and is dropped entirely. Stripping only the channel markers, and
  keeping the text between them, concatenated deliberation into the answer.

  Args:
    text: The raw generation.

  Returns:
    The answer text with reasoning removed.
  """
  out = text or ""
  parts = _FINAL_CHANNEL_RE.split(out)
  if len(parts) > 1:
    out = re.split(r"<\|", parts[-1])[0]
  else:
    glued = _GLUED_FINAL_RE.split(out)
    if len(glued) > 1:
      out = glued[-1]
  out = _THINK_CLOSED_RE.sub("", out)
  out = _THINK_OPEN_RE.sub("", out)
  out = _STRAY_TAG_RE.sub("", out)
  return out.strip()


def _is_bad_answer(text: str) -> bool:
  """Report whether a generation can never be shown to the user as an answer.

  Args:
    text: The extracted answer text.

  Returns:
    True when the text is empty, an attempted tool call, or visible deliberation about
    using tools. Reasoning is never returned as an answer.
  """
  if not text:
    return True
  if _TOOL_JSON_RE.search(text):
    return True
  if _TOOL_PROSE_RE.search(text):
    return True
  if _SEARCHING_RE.search(text):
    return True
  return bool(_PERSONA_RE.search(text))


def _extract(raw: str) -> str:
  """Pull the answer out of a raw generation.

  Args:
    raw: The raw generation.

  Returns:
    The answer text, tolerating an echoed ``ANSWER:`` heading and falling back to the last
    substantive paragraph.
  """
  clean = _strip_reasoning(raw)
  match = _ANSWER_HEADING_RE.search(clean)
  candidate = (match.group(1) if match else clean).strip(_STRIP_CHARS)
  if candidate:
    return candidate
  paragraphs = [p.strip(_STRIP_CHARS) for p in re.split(r"\n\s*\n", clean) if p.strip()]
  paragraphs = [p for p in paragraphs if not _SECTION_HEADING_RE.match(p)]
  return paragraphs[-1] if paragraphs else clean.strip(_STRIP_CHARS)


def _group_documents(
  evidence: list[TreeNode], max_evidence: int
) -> tuple[str, list[str]]:
  """Group evidence pieces by source document for the answer prompt.

  Args:
    evidence: The evidence collected by the traversal.
    max_evidence: Cap on the number of pieces to include.

  Returns:
    A pair of the rendered document block and the document ids in the order shown.
  """
  groups: dict[str, list[str]] = {}
  order: list[str] = []
  for item in evidence[:max_evidence]:
    key = item.source_file() or item.path or item.name
    if key not in groups:
      groups[key] = []
      order.append(key)
    groups[key].append(full(item.content or item.summary))
  rendered = (
    "\n\n".join(
      f"--- START DOCUMENT [doc_id: {key}] ---\n"
      + "\n".join(groups[key])
      + "\n--- END DOCUMENT ---"
      for key in order
    )
    or "(no documents were retrieved)"
  )
  return rendered, order


def write_answer(
  ctx: SearchContext, question: str, evidence: list[TreeNode]
) -> DraftedAnswer:
  """Write the final answer from the collected evidence, in the house format.

  Generous token budgets, and one escalating retry, guard the failure this most often
  hits: the model reasons over sizeable evidence, exhausts its whole budget in the
  reasoning channel, and returns empty content - which would otherwise collapse into a
  not-found answer even though retrieval was good and the gate had just certified it.

  Args:
    ctx: The search context.
    question: The question being answered.
    evidence: The evidence collected by the traversal.

  Returns:
    The drafted answer, its document ids and the diagnostics.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  docs, doc_ids = _group_documents(evidence, ctx.config.max_evidence)
  header = f"{system_prompt()}\n\n{ANSWER_INSTRUCTIONS}\n\n"
  prompt = f"{header}DOCUMENTS:\n{docs}\n\nQUESTION: {question}\n\nANSWER:\n"
  retry_prompt = (
    f"{header}"
    "NOTE: the search and retrieval steps have ALREADY been completed — the resulting "
    "documents are provided in full below. Do NOT attempt to call, simulate, or "
    "describe any tool. Output ONLY the answer text, with no headings, no preamble, "
    "and no reasoning.\n\n"
    f"DOCUMENTS:\n{docs}\n\nQUESTION: {question}\n\nANSWER:"
  )

  diagnostics = AnswerDiagnostics()
  answer = ""
  for number, (attempt_prompt, budget) in enumerate(
    ((prompt, 4096), (retry_prompt, 6144)), start=1
  ):
    response = ctx.llm_full(attempt_prompt, num_predict=budget, thinking_fallback=False)
    diagnostics.attempts.append(
      AnswerAttempt(
        number=number,
        budget=budget,
        content_len=response.content_len,
        thinking_len=response.thinking_len,
        done_reason=response.done_reason,
        reasoning_exhausted=response.reasoning_exhausted(),
      )
    )
    answer = _extract(response.text)
    good = not _is_bad_answer(answer) and not (evidence and answer.strip() == DUNNO)
    if good:
      break
    if evidence and answer.strip() == DUNNO:
      diagnostics.dunno_with_evidence = True

  if _is_bad_answer(answer):
    answer = DUNNO

  if evidence and answer and answer.strip() != DUNNO:
    diagnostics.words = word_count(answer)

  # Citation repair: the instructions require a [doc_id], but the model sometimes drops
  # them entirely. Re-emit the SAME answer with citations added, one per document.
  if (
    evidence
    and answer
    and answer.strip() != DUNNO
    and not _has_allowed_citation(answer, doc_ids)
  ):
    allowed = ", ".join("[" + d + "]" for d in doc_ids)
    cite_prompt = (
      f"{system_prompt()}\n\n"
      "Rewrite the answer below so that the information from each document ends with a "
      "single citation to that document, formatted exactly as [doc_id], using ONLY "
      "these "
      f"document ids: {allowed}. Cite each document at most ONCE — group the information "
      "from the same document together and place its single [doc_id] at the end of "
      "that group. Do NOT repeat a [doc_id] after every sentence. Keep the wording and "
      "facts the same; only add the [doc_id] citations. Output ONLY the rewritten "
      "answer, no headings.\n\n"
      f"DOCUMENTS:\n{docs}\n\n"
      f"QUESTION: {question}\n\nANSWER TO CITE:\n{answer}\n\nCITED ANSWER:"
    )
    raw = ctx.llm(cite_prompt, num_predict=1024, thinking_fallback=False)
    cited = _strip_reasoning(raw).strip(_STRIP_CHARS)
    if (
      cited
      and cited.strip() != DUNNO
      and _only_allowed_citations(cited, doc_ids)
      and not _is_bad_answer(cited)
    ):
      answer = cited
      diagnostics.citations_repaired = True

  return DraftedAnswer(
    text=dedupe_citations(answer), doc_ids=doc_ids, diagnostics=diagnostics
  )
