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

Ranking a node's children for relevance to the question.

Scores order the descent winner and seed the teleport frontier. Three signals reach the
ranker for each candidate: the LLM summary (a lossy gist), a ``text:`` excerpt of the
candidate's ACTUAL words, and for a folder a ``contains:`` line of the names inside it.
Scoring happens in small fixed batches so the JSON response is never truncated - the
all-0.5 failure mode came from exactly that truncation, not from the model's judgement.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from typing import TypeAlias

from treerag.context import SearchContext
from treerag.state import FrontierItem
from treerag.text import (
  clip,
  full,
  json_str,
  name_stem_set,
  parse_json_object,
  stem_set,
  strip_code_fence,
  tokens,
)
from treerag.types import TreeNode

#: One scored candidate: the node, its score, and the ranker's reasoning sentence.
ScoredChild: TypeAlias = tuple[TreeNode, float, str]

_SCORES_REGION_RE = re.compile(r'"scores"\s*:\s*\{(.*)', flags=re.S)
_SCORE_PAIR_RE = re.compile(r'"(\d+)"\s*:\s*(0?\.\d+|[01](?:\.0+)?)')
_REASONING_RE = re.compile(r'"reasoning"\s*:\s*"([^"]*)"')
_CHOICE_RE = re.compile(r'"?choice"?\s*[:=]\s*(-?\d+)')

#: Recorded against candidates past the fan-out cap, so a trace shows why they scored 0.
_OVERFLOW_REASON = "deferred: node fan-out exceeded the ranking cap"


def _cosine(a: list[float], b: list[float]) -> float:
  """Compute cosine similarity between two vectors.

  Args:
    a: The first vector.
    b: The second vector.

  Returns:
    The cosine similarity, or 0.0 when either vector is degenerate.
  """
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  denominator = norm_a * norm_b
  if denominator == 0.0:
    return 0.0
  return sum(x * y for x, y in zip(a, b, strict=False)) / denominator


def _name_order_key(
  ctx: SearchContext, query: str, population: int
) -> Callable[[str], tuple[int, float]]:
  """Build the sort key that floats query-relevant names to the front of a clipped list.

  Lexical overlap is the primary signal: deterministic, free, and it catches the exact
  filename matches an embedding can miss. Embedding similarity is secondary, for synonyms
  - but each one is a network round-trip, and ordering a several-hundred-name listing one
  embedding at a time costs minutes before a single ranking call is made. That was the
  difference between a query taking two minutes and taking two hours. So embeddings are
  used only while the list is small enough to afford them; past that the ordering is
  lexical alone.

  Args:
    ctx: The search context.
    query: The question being answered.
    population: How many names are about to be ordered.

  Returns:
    A sort key mapping a name to its (lexical overlap, embedding similarity) pair.
  """
  query_tokens = set(tokens(query))
  affordable = population <= ctx.config.max_embed_per_decision
  query_vector = ctx.embed(query) if (affordable and query_tokens) else None

  def key(name: str) -> tuple[int, float]:
    lexical = sum(1 for w in set(tokens(name)) if w in query_tokens)
    similarity = 0.0
    if query_vector is not None:
      name_vector = ctx.embed(name)
      if name_vector is not None:
        similarity = _cosine(query_vector, name_vector)
    return (lexical, similarity)

  return key


def contains_line(ctx: SearchContext, query: str, node: TreeNode) -> str:
  """Build the ``contains:`` line listing an interior candidate's child names.

  A folder's own summary is lossy - a generic one-line gist can hide the exactly-right
  file sitting inside it - so the ranker is also shown the names of what the folder
  holds, which for SOPs and worksheets are usually dead giveaways. When a folder is
  too wide for every name to fit the budget, names are ordered so query-relevant ones
  survive the clip: lexical overlap first (deterministic and free), embedding similarity
  second (for synonyms, when the endpoint is up), natural order last.

  Args:
    ctx: The search context.
    query: The question being answered.
    node: The candidate whose children should be listed.

  Returns:
    The formatted line, or the empty string for leaves and for nodes whose children are
    all chunks, which carry no routing signal.
  """
  kids = list(node.children)
  if not kids:
    return ""
  if all(kid.is_leaf() for kid in kids):
    return ""
  budget = ctx.config.contains_chars
  names = [str(kid.name or "?") for kid in kids]
  if sum(len(n) + 2 for n in names) > budget:
    names = sorted(names, key=_name_order_key(ctx, query, len(names)), reverse=True)
  shown: list[str] = []
  used = 0
  for name in names:
    if used + len(name) + 2 > budget:
      break
    shown.append(name)
    used += len(name) + 2
  extra = len(kids) - len(shown)
  listing = "; ".join(shown) + (f"; (+{extra} more)" if extra > 0 else "")
  plural = "s" if len(kids) != 1 else ""
  return f"\n      contains ({len(kids)} item{plural}): {listing}"


def _iter_chunks_bounded(node: TreeNode, cap: int) -> Iterator[TreeNode]:
  """Walk a node's chunks, stopping after a fixed number.

  Args:
    node: The subtree to walk.
    cap: Maximum number of chunks to yield.

  Yields:
    Up to ``cap`` leaf chunks, in document order.
  """
  stack: list[TreeNode] = [node]
  seen = 0
  while stack and seen < cap:
    current = stack.pop()
    if current.is_leaf():
      yield current
      seen += 1
    else:
      stack.extend(reversed(current.children))


def content_preview(
  ctx: SearchContext, node: TreeNode, limit: int, query: str
) -> tuple[str, bool]:
  """Build a candidate's ``text:`` excerpt from its actual words.

  For a single-document node whose text overflows the budget the excerpt is drawn from the
  chunks with the highest lexical overlap with the question, restored to document order so
  it reads naturally. A section that states the answer past its preamble is therefore
  previewed on the answer rather than on the preamble. Folders short-circuit: their "text"
  would just be one document's intro, so they route on contained names instead.

  Relevance here is lexical on purpose. Embedding each chunk turned one question into
  thousands of round-trips in the source benchmark; a preview is only a routing hint, and
  a bag-of-words overlap floats the answer-bearing paragraph perfectly well.

  Args:
    ctx: The search context.
    node: The candidate to preview.
    limit: Character budget for the excerpt.
    query: The question being answered.

  Returns:
    A pair of the excerpt and a flag that is True when the node spans several documents,
    in which case no excerpt is built.
  """
  cap = ctx.config.preview_scan_cap
  if node.is_leaf():
    return clip(full(node.content or node.summary), limit), False

  chunks: list[tuple[int, str]] = []
  sources: set[str] = set()
  for position, chunk in enumerate(_iter_chunks_bounded(node, cap)):
    source = chunk.source_file()
    if source:
      sources.add(source)
      if len(sources) > 1:
        return "", True
    text = full(chunk.content or "")
    if text:
      chunks.append((position, text))
  if not chunks:
    return "", False

  picked = chunks
  overflows = sum(len(t) + 2 for _, t in chunks) > limit
  if ctx.config.preview_relevance and query and overflows and len(chunks) > 1:
    query_tokens = set(tokens(query))
    if query_tokens:
      scored = [
        (sum(1 for w in tokens(text) if w in query_tokens), position, text)
        for position, text in chunks
      ]
      scored.sort(key=lambda item: item[0], reverse=True)
      selected: list[tuple[int, str]] = []
      used = 0
      for overlap, position, text in scored:
        if used >= limit:
          break
        if overlap == 0 and used > 0:
          continue
        selected.append((position, text))
        used += len(text) + 2
      if selected:
        picked = sorted(selected, key=lambda item: item[0])

  buffer: list[str] = []
  used = 0
  for _position, text in picked:
    if used >= limit:
      break
    buffer.append(text)
    used += len(text) + 2
  return clip("\n".join(buffer), limit), False


def score_entry(ctx: SearchContext, position: int, node: TreeNode, query: str) -> str:
  """Render one candidate's block for the ranker.

  Args:
    ctx: The search context.
    position: The candidate's index within its batch.
    node: The candidate.
    query: The question being answered.

  Returns:
    The candidate's name, summary, text excerpt and contains-line, as the ranker sees it.
  """
  preview_budget = ctx.config.score_preview
  if node.is_leaf():
    preview, _ = content_preview(ctx, node, preview_budget, query)
    return f"[{position}] {node.name} — {preview}"
  preview, multi = content_preview(ctx, node, preview_budget, query)
  summary = node.summary or ""
  line = f"[{position}] {node.name}" + (f" — {summary}" if summary else "")
  if not multi and preview:
    line += f"\n      text: {preview}"
  line += contains_line(ctx, query, node)
  return line


def parse_scores(raw: str | None, count: int) -> dict[int, float]:
  """Extract a score per candidate index from the ranker's response.

  Args:
    raw: The raw model response.
    count: How many candidates were scored, so out-of-range indices can be dropped.

  Returns:
    A mapping of candidate index to score in ``[0, 1]``, holding only the indices the
    model actually scored.
  """
  out: dict[int, float] = {}
  decoded = parse_json_object(raw)
  if decoded is not None:
    scores_field = decoded.get("scores", decoded)
    if isinstance(scores_field, dict):
      for key, value in scores_field.items():
        key_text = str(key).strip()
        if not key_text.isdigit():
          continue
        try:
          out[int(key_text)] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
          continue
    elif isinstance(scores_field, list):
      for position, value in enumerate(scores_field):
        try:
          out[position] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
          continue
  if not out:
    text = strip_code_fence(raw)
    region_match = _SCORES_REGION_RE.search(text)
    region = region_match.group(1) if region_match else text
    for key, value in _SCORE_PAIR_RE.findall(region):
      index = int(key)
      if 0 <= index < count:
        out[index] = max(0.0, min(1.0, float(value)))
  return {i: v for i, v in out.items() if 0 <= i < count}


def _reasoning_of(raw: str | None) -> str:
  """Pull the ranker's reasoning sentence out of its response.

  Args:
    raw: The raw model response.

  Returns:
    The reasoning sentence, or the empty string when the response carries none.
  """
  match = _REASONING_RE.search(raw or "")
  return match.group(1).strip() if match else ""


def name_match_boost(ctx: SearchContext, query_stems: set[str], node: TreeNode) -> float:
  """Compute the small deterministic nudge a name match earns a file or folder.

  A recurring failure was the correct file ranking second behind a decoy, with the tell
  always a NAME the model under-weighted. The nudge is capped small so it only reorders
  near-ties: a clear content winner is never overturned.

  Args:
    ctx: The search context.
    query_stems: The question's stemmed content tokens.
    node: The candidate.

  Returns:
    The additive nudge, and 0.0 for sections and paragraphs, whose names are
    auto-generated, and for an empty intersection.
  """
  if node.node_type not in ("folder", "file", "document"):
    return 0.0
  hits = len(query_stems & name_stem_set(node.name))
  if hits <= 0:
    return 0.0
  return min(ctx.config.name_boost_max, ctx.config.name_boost_step * hits)


def _pick_best_index(
  ctx: SearchContext, query: str, candidates: list[TreeNode], memory: list[str]
) -> int:
  """Ask the model which single candidate is most likely to hold the answer.

  Used only as a tie-break for candidates the ranker refused to score, so those never
  collapse onto one flat default value.

  Args:
    ctx: The search context.
    query: The question being answered.
    candidates: The candidates to choose between.
    memory: Working-memory facts gathered so far.

  Returns:
    The index of the chosen candidate, defaulting to 0 when the response is unusable.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  if len(candidates) == 1:
    return 0
  lines = "\n".join(score_entry(ctx, i, node, query) for i, node in enumerate(candidates))
  prompt = (
    "Which ONE of these sections is MOST likely to contain information answering the "
    f"question?\n\nQUESTION: {query}\n\nSECTIONS:\n{lines}\n\n"
    'Reply ONLY json: {"choice":<index>}'
  )
  raw = ctx.llm(prompt, num_predict=128)
  match = _CHOICE_RE.search(raw or "")
  if match:
    choice = int(match.group(1))
    if 0 <= choice < len(candidates):
      return choice
  return 0


def _score_prompt(
  query: str, memory: list[str], lines: str, count: int, strict: bool
) -> str:
  """Build the scoring prompt for one batch of candidates.

  Args:
    query: The question being answered.
    memory: Working-memory facts gathered so far.
    lines: The rendered candidate blocks.
    count: How many candidates are in the batch.
    strict: Add an explicit instruction to score every index, used on the retry.

  Returns:
    The complete prompt.
  """
  memory_text = "\n".join("- " + m for m in memory) or "(empty)"
  keys = ", ".join(f'"{i}":<0-1>' for i in range(count))
  note = (
    ""
    if not strict
    else f"\nIMPORTANT: score EVERY index 0..{count - 1}. "
    f"Output all {count} scores, no omissions.\n"
  )
  return (
    "You are navigating a document tree to answer a question. Rate how likely EACH "
    "entry below is to contain information that helps answer the question.\nEach entry "
    "may show up to three things: a SUMMARY (a lossy gist), a 'text:' excerpt (the "
    "section's ACTUAL words), and a 'contains:' line (names of the files/subsections "
    "inside). Weight them accordingly:\n- 'text:' is the strongest signal. If the "
    "actual words address the question, score HIGH even when the name and summary do "
    "not obviously match — a summary often drops the exact sentence that answers, and "
    "a section's name may use different words than the question.\n- 'contains:' names "
    "are a first-class routing signal for folders: a folder whose listed contents "
    "match what is asked should score HIGH even if its summary reads generic.\n- Match "
    "on MEANING, not shared keywords. Domain-equivalent wordings count: an "
    "'unsatisfactory', 'ungraded', or 'unacceptable' result IS a 'failed' one; "
    "'corrective action' / 'follow-up' is 'what to do' after a problem; 'frequency' "
    "answers 'how often'. Judge whether the content addresses the question's "
    "INTENT.\n- Do not demand a particular FORM of answer. A question asking WHEN / "
    "WHY / HOW / UNDER WHAT CIRCUMSTANCES an event happens is answered not only by an "
    "explicit rule sentence but equally by the process, trigger, or form that "
    "INITIATES the event ('done as part of Y', 'requires initiating Z'), and by a "
    "record DOCUMENTING an actual instance of it (the circumstances in the record show "
    "when it happens). Score such content HIGH.\n- The question's IDENTIFYING words "
    "matter as much as its attribute words. When it asks a fact about one SPECIFIC "
    "named thing (a particular instrument, unit, system, model, site or document), an "
    "entry stating the SAME KIND of fact for a DIFFERENT thing of that kind is a "
    "decoy: score it LOW however exactly its wording matches the asked attribute, and "
    "score HIGH entries identifiably about the named thing itself (or the branch "
    "covering it), even when their wording matches the attribute less well.\n\n"
    f"QUESTION: {query}\n\n"
    f"WORKING MEMORY (facts gathered so far):\n{memory_text}\n\n"
    f"CANDIDATES:\n{lines}\n\n"
    "Give ONE short sentence of reasoning FIRST, then score each section from 0.0 "
    "(irrelevant) to 1.0 (almost certainly contains the answer). Use the full range; "
    "differentiate them — do NOT give everything the same middling number." + note + "\n"
    "Reply with ONLY json:\n"
    '{"reasoning":"<1 short sentence>", "scores":{' + keys + "}}"
  )


def _score_small_batch(
  ctx: SearchContext, query: str, candidates: list[TreeNode], memory: list[str]
) -> list[ScoredChild]:
  """Score one small batch of candidates, guaranteeing a real score for each.

  Retries once with an explicit instruction, then breaks any remaining tie by asking the
  model to pick the best of the unscored leftovers, so nothing ever falls back to a flat
  default value.

  Args:
    ctx: The search context.
    query: The question being answered.
    candidates: The batch to score.
    memory: Working-memory facts gathered so far.

  Returns:
    One triple per candidate, in the batch's own order.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  count = len(candidates)
  lines = "\n".join(score_entry(ctx, i, node, query) for i, node in enumerate(candidates))
  budget = max(256, 140 * count)

  raw = ctx.llm(
    _score_prompt(query, memory, lines, count, strict=False), num_predict=budget
  )
  scores = parse_scores(raw, count)
  reasoning = _reasoning_of(raw)
  missing = [i for i in range(count) if i not in scores]
  if missing:
    raw_retry = ctx.llm(
      _score_prompt(query, memory, lines, count, strict=True), num_predict=budget
    )
    for index, value in parse_scores(raw_retry, count).items():
      scores.setdefault(index, value)
    if not reasoning:
      reasoning = _reasoning_of(raw_retry)
    missing = [i for i in range(count) if i not in scores]
  if missing:
    base = min(scores.values(), default=0.5)
    remaining = list(missing)
    value = max(0.0, base - 0.01)
    while remaining:
      best = _pick_best_index(ctx, query, [candidates[i] for i in remaining], memory)
      chosen = remaining.pop(best if 0 <= best < len(remaining) else 0)
      scores[chosen] = max(0.0, value)
      value = max(0.0, value - 0.02)
  return [(candidates[i], scores[i], reasoning) for i in range(count)]


def shortlist_by_fanout(
  ctx: SearchContext, query: str, candidates: list[TreeNode]
) -> tuple[list[TreeNode], list[TreeNode]]:
  """Split a node's children into the ones worth scoring and the rest.

  Ranking costs one LLM call per ``score_batch`` candidates, so its price is linear in
  fan-out. On this corpus the median node has 2-4 children and nothing needs splitting,
  but the tail runs to 2,661 - which is 533 LLM calls for a single decision, and the
  reason a query can run for hours. Past the cap, candidates are ordered the same way
  :func:`contains_line` already orders an over-long name list - lexical overlap with the
  question first, then embedding similarity when the endpoint is up, then natural order -
  and the remainder is deferred rather than dropped.

  Args:
    ctx: The search context.
    query: The question being answered.
    candidates: The children of the node being ranked.

  Returns:
    A pair of the candidates to score and the overflow to defer. The overflow is empty
    whenever the node fits under the cap, which is the overwhelming majority of nodes.
  """
  cap = ctx.config.max_rank_candidates
  if len(candidates) <= cap:
    return list(candidates), []

  key = _name_order_key(ctx, query, len(candidates))
  ordered = sorted(candidates, key=lambda node: key(node.name), reverse=True)
  return ordered[:cap], ordered[cap:]


def rank_children(
  ctx: SearchContext, query: str, candidates: list[TreeNode], memory: list[str]
) -> list[ScoredChild]:
  """Score the candidates and return them best first.

  Scoring runs in fixed batches so each JSON response stays small and complete. After
  scoring, the name-match boost nudges file and folder candidates whose name shares
  distinctive terms with the question - but only within a proximity window of the top raw
  score, so a low-scored name decoy cannot leap a confident winner.

  A node whose fan-out exceeds ``max_rank_candidates`` is shortlisted first; the overflow
  is returned at score 0.0 so the caller's frontier push routes it to the sub-floor
  reserve tier, where it stays reachable once the active frontier is exhausted. Recall is
  preserved, and only the pathological tail of wide nodes is affected.

  Args:
    ctx: The search context.
    query: The question being answered.
    candidates: The children to score.
    memory: Working-memory facts gathered so far.

  Returns:
    The scored candidates sorted from most to least relevant, followed by any deferred
    overflow at score 0.0.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
    SearchBudgetError: If the search runs past its ceiling mid-ranking.
  """
  pool, overflow = shortlist_by_fanout(ctx, query, candidates)
  if not pool:
    return [(node, 0.0, _OVERFLOW_REASON) for node in overflow]
  batch_size = ctx.config.score_batch
  results: list[ScoredChild] = []
  for start in range(0, len(pool), batch_size):
    results.extend(
      _score_small_batch(ctx, query, pool[start : start + batch_size], memory)
    )
  deferred: list[ScoredChild] = [(node, 0.0, _OVERFLOW_REASON) for node in overflow]

  if ctx.config.name_boost_max > 0 and results:
    query_stems = stem_set(query)
    top_raw = max(score for _, score, _ in results)
    adjusted: list[tuple[TreeNode, float, str, float, float]] = []
    for node, score, reason in results:
      boost = (
        name_match_boost(ctx, query_stems, node)
        if score >= top_raw - ctx.config.name_boost_window
        else 0.0
      )
      adjusted.append((node, min(1.0, score + boost), reason, boost, score))
    adjusted.sort(key=lambda item: (item[1], item[3], item[4]), reverse=True)
    return [(node, score, reason) for node, score, reason, _, _ in adjusted] + deferred

  results.sort(key=lambda item: item[1], reverse=True)
  return results + deferred


def alternative_entry(
  ctx: SearchContext, position: int, entry_node: TreeNode, score: float, query: str
) -> str:
  """Render one alternative's block for the contrast and residual choosers.

  Shows the same signals the ranker routes on - score, position in the tree, summary, the
  candidate's own words, and for a folder the names it contains - so the choice is made on
  content rather than on score alone.

  Args:
    ctx: The search context.
    position: The candidate's index in the shortlist.
    entry_node: The candidate node.
    score: Its frontier score.
    query: The question being answered.

  Returns:
    The rendered block.
  """
  line = f"[{position}] (score {score:.2f}) {entry_node.name}"
  parent = ctx.index.parent_of(entry_node)
  if parent is not None and parent.node_type != "root":
    line += f"   [inside: {clip(parent.name, 40)}]"
  summary = clip(entry_node.summary or "", 200)
  if summary:
    line += "\n      summary: " + summary
  preview, multi = content_preview(ctx, entry_node, ctx.config.alt_preview, query)
  if preview and not multi:
    line += "\n      text: " + preview
  contains = contains_line(ctx, query, entry_node)
  if contains:
    line += "\n      " + clip(contains, 300)
  return line


def parse_choice(raw: str | None) -> tuple[int | None, str]:
  """Read a ``{"reasoning": ..., "choice": n}`` response.

  Args:
    raw: The raw model response.

  Returns:
    A pair of the chosen index - ``None`` when the response carried none - and the
    reasoning sentence.
  """
  decoded = parse_json_object(raw)
  reasoning = ""
  choice: int | None = None
  if decoded is not None:
    reasoning = json_str(decoded, "reasoning")
    value = decoded.get("choice")
    if isinstance(value, bool):
      value = None
    if isinstance(value, (int, float, str)):
      try:
        choice = int(value)
      except (TypeError, ValueError):
        choice = None
  if choice is None:
    match = _CHOICE_RE.search(strip_code_fence(raw))
    if match:
      choice = int(match.group(1))
  return choice, reasoning


def diverse_shortlist(
  ctx: SearchContext,
  entries: list[FrontierItem],
  limit: int,
) -> list[FrontierItem]:
  """Build a score-ordered shortlist in which no one top-level region dominates.

  A frontier score is not comparable across the tree: a candidate inside the committed
  region was scored while the ranker was there, reading its text previews with a working
  memory full of that region's account, whereas a top-level sibling was scored cold on its
  name alone. So a plain top-K by score is always K siblings of what was just read, and an
  agent asked to pick the candidate that would CHANGE its answer can only ever pick more
  of the same. Capping each region's slots is what lets the widened eligibility bands mean
  anything. Recall is untouched: no entry leaves the frontier, and the list is topped back
  up in score order, so with fewer regions than slots this returns the old top-K exactly.

  Args:
    ctx: The search context.
    entries: The frontier entries to choose from.
    limit: Maximum shortlist length.

  Returns:
    The shortlist, score-ordered.
  """
  quota = ctx.config.shortlist_region_quota
  ranked = sorted(entries, key=lambda e: e.score, reverse=True)
  out: list[FrontierItem] = []
  used: Counter[str] = Counter()
  for entry in ranked:
    if len(out) >= limit:
      break
    region = ctx.index.region_of(entry.node.node_id)
    if used[region] >= quota:
      continue
    out.append(entry)
    used[region] += 1
  if len(out) < limit:
    chosen = {id(entry) for entry in out}
    for entry in ranked:
      if len(out) >= limit:
        break
      if id(entry) not in chosen:
        out.append(entry)
  return sorted(out, key=lambda e: e.score, reverse=True)
