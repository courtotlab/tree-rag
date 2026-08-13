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

Every agentic decision the traversal makes, other than ranking children.

Each of these is one LLM call with a strict JSON reply, and each carries a default that
matters: an unparseable take/reject defaults to TAKE, because the stop gate will verify
sufficiency anyway, whereas an unparseable below-the-bar read defaults to SKIP, because
admitting the weakest material in a document with no decision made at all is how a
neighbouring-subject passage displaces an answer already in hand.
"""

import re
from dataclasses import dataclass

from treequest.prompting import DUNNO
from treequest.context import SearchContext
from treequest.events import (
  GranularityStep,
  RankedCandidate,
  SweepBreadthEvict,
  SweepDeferred,
  SweepFloorDismissed,
  SweepRead,
  SweepRecord,
  SweepSelection,
  SweepTriaged,
)
from treequest.prompts import (
  CONTRAST_TAG,
  POLARITY_CRITERIA,
  PROVENANCE_CRITERIA,
  RECENCY_CRITERIA,
  RESIDUAL_TAG,
  STRICT_CRITERIA,
  UNREAD_SECTIONS_NOTE,
  contrast_directive,
  residual_directive,
)
from treequest.ranking import (
  alternative_entry,
  parse_choice,
  rank_children,
)
from treequest.state import (
  AgentState,
  FrontierItem,
  add_memory,
  evidence_sources,
)
from treequest.text import (
  clip,
  full,
  json_str,
  json_str_list,
  parse_json_object,
  strip_code_fence,
)
from treequest.tree import whole_unit
from treequest.types import TreeNode

_SCOPE_RE = re.compile(r'"?scope"?\s*[:=]\s*"?(self|wider)"?', re.I)
_DECISION_RE = re.compile(r'"?decision"?\s*[:=]\s*"?(take|skip)"?', re.I)
_SUFFICIENT_RE = re.compile(r'"?sufficient"?\s*[:=]\s*(true|false)', re.I)


@dataclass(frozen=True, slots=True)
class FileVerdict:
  """The agent's take/reject decision about a document it navigated to.

  Attributes:
    decision: ``take`` or ``reject``.
    reasoning: The one-sentence justification, produced before the decision.
    remember: An answer-bearing fact worth keeping in working memory.
  """

  decision: str
  reasoning: str
  remember: str


@dataclass(frozen=True, slots=True)
class SufficiencyVerdict:
  """The sufficiency gate's verdict on the evidence in hand.

  Attributes:
    sufficient: Whether the evidence would produce a real answer rather than the
      not-found string.
    reasoning: The gate's one-sentence verdict, with the missing parts appended.
    missing: The parts of the question the evidence does not yet settle.
    basis: For a definition question, ``subject``, ``incidental`` or ``none``.
  """

  sufficient: bool
  reasoning: str
  missing: tuple[str, ...]
  basis: str


# ---------------------------------------------------------------------------
# reading a reached document
# ---------------------------------------------------------------------------


def judge_file(
  ctx: SearchContext,
  query: str,
  node: TreeNode,
  memory: list[str],
  *,
  full_content: bool = False,
) -> FileVerdict:
  """Decide whether a reached document contributes real answer-bearing content.

  This is a topical keep/skip, not a stop decision: "is what I have ENOUGH to answer?" is
  handled separately by the sufficiency gate, at the moment the agent tries to stop.

  The prompt carries two opposed guards. The answer-form guard stops the judge silently
  substituting a stricter question that demands a particular SHAPE of answer, because
  corpora state such answers as the process that performs an event or as a record of an
  instance of it. The subject-identity guard stops the opposite error, taking a document
  that states the right KIND of fact about the wrong entity.

  Args:
    ctx: The search context.
    query: The question being answered.
    node: The document or passage that was reached.
    memory: Working memory, read but not modified.
    full_content: Judge on the assembled document text rather than the node's own text,
      used in synthesis mode where a document node's content is empty and its summary is
      a lossy gist.

  Returns:
    The verdict. An unparseable response defaults to ``take``, because the stop gate will
    verify sufficiency anyway.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  if full_content and node.is_file():
    body = clip(full(whole_unit(node).content or node.summary), 3000)
  else:
    body = clip(full(node.content or node.summary), 3000)
  memory_text = "\n".join("- " + m for m in memory) or "(empty)"
  prompt = (
    "You have navigated to a specific document and can now read it. Decide, based ONLY "
    "on its content, whether it contributes real answer-bearing information toward the "
    "question.\n\n"
    f"QUESTION: {query}\n\n"
    f"WORKING MEMORY (facts gathered so far):\n{memory_text}\n\n"
    f"DOCUMENT: {node.name}\nCONTENT:\n{body}\n\n"
    "Judge the content against the question AS ASKED — do not silently substitute a "
    "stricter question that demands a specific FORM of answer (an explicit rule, "
    "timing, or number). For a question about when / why / how / under what "
    "circumstances something happens, ALL of these are real answer-bearing content:\n- "
    "a TRIGGER, CONDITION, or INITIATING PROCESS: the asked-about event is performed "
    "or initiated through some process or form ('X is changed by initiating Y', 'as "
    "part of Z') — that process IS the answer to when/how/under-what-circumstances it "
    "happens;\n- a cross-reference naming WHICH procedure, form, or record governs or "
    "performs the event;\n- a concrete RECORD or completed form documenting an actual "
    "INSTANCE of the event — the circumstances recorded in it are evidence of when and "
    "why the event occurs.\n\nSUBJECT IDENTITY — the opposite trap: do not accept a "
    "document merely because it states the RIGHT KIND of fact for the WRONG thing. "
    "When the question asks a fact about one specific named subject, first ask WHOSE "
    "fact this document states — corpora routinely record the same attribute for MANY "
    "same-kind entities (other units, instruments, models, sites, versions):\n- if the "
    "document identifiably concerns a DIFFERENT entity of the same kind, 'reject' — "
    "the identical attribute of a sibling entity is NOT the answer, however exactly it "
    "matches the asked attribute — and put WHICH entity it does describe in "
    "'remember', so the search can tell the siblings apart;\n- if it states the asked "
    "fact without identifying which entity the value belongs to, you may 'take' it, "
    "but say in 'remember' that the value is not yet tied to the asked "
    "subject.\n\nGive ONE sentence of reasoning FIRST, then decide:\n- 'take' if it "
    "holds real answer-bearing content toward the question — even a partial piece (a "
    "definition, an example, one required fact) counts.\n- 'reject' if it contributes "
    "nothing an answer could be built on (boilerplate, off-topic, or merely mentions "
    "the right words without actual information). Being in the right file or using the "
    "right terms is NOT enough.\nBefore deciding 'reject', re-read your reasoning "
    "sentence: if it says the document 'only' states some fact ABOUT the asked-about "
    "event (what initiates it, which process handles it, an instance of it happening), "
    "then that fact IS answer-bearing — decide 'take' and put it in 'remember'.\nReply "
    "ONLY json:\n"
    '{"reasoning":"<1 sentence, before deciding>", "remember":"<a useful answer-bearing '
    'fact to keep, else empty>", "decision":"take|reject"}'
  )
  raw = ctx.llm(prompt, num_predict=384)
  decoded = parse_json_object(raw) or {}
  decision = json_str(decoded, "decision").lower()
  if decision not in ("take", "reject"):
    decision = "take"
  return FileVerdict(
    decision=decision,
    reasoning=json_str(decoded, "reasoning"),
    remember=json_str(decoded, "remember"),
  )


# ---------------------------------------------------------------------------
# granularity: how much surrounding context to keep
# ---------------------------------------------------------------------------


def scope_choice(
  ctx: SearchContext,
  query: str,
  passage: TreeNode,
  section: TreeNode,
  memory: list[str],
) -> tuple[str, str]:
  """Ask whether the answer lives in a passage alone or depends on its wider section.

  Args:
    ctx: The search context.
    query: The question being answered.
    passage: The single best-matching part.
    section: The larger section the passage sits inside.
    memory: Working memory, read but not modified.

  Returns:
    A pair of the scope - ``self`` to keep the passage, ``wider`` to climb to the section
    - and the one-sentence reasoning. Defaults to ``self``.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  passage_text = clip(full(passage.content or passage.summary), 1600)
  section_text = clip(full(whole_unit(section).content), 3500)
  memory_text = "\n".join("- " + m for m in memory) or "(empty)"
  prompt = (
    "You have found a passage relevant to the question. Decide the right amount of "
    "surrounding context to keep as evidence.\n\n"
    f"QUESTION: {query}\n\n"
    f"WORKING MEMORY (facts already gathered):\n{memory_text}\n\n"
    f"PASSAGE (the single best-matching part):\n{passage_text}\n\n"
    f"SURROUNDING SECTION (the passage is one part of this larger section):\n"
    f"{section_text}\n\n"
    "Decide:\n- 'self'  : the PASSAGE on its own fully contains the answer; the rest "
    "of the section adds nothing the answer needs.\n- 'wider' : the answer DEPENDS ON "
    "the surrounding section — a required detail, qualifier, definition, version, "
    "location, or reference the passage omits sits elsewhere in this section. Keep the "
    "whole section so that detail is not lost.\nPrefer 'self' unless the section "
    "clearly carries a needed detail the passage is missing.\n"
    'Reply ONLY json: {"reasoning":"<1 sentence>", "scope":"self|wider"}'
  )
  raw = ctx.llm(prompt, num_predict=320)
  decoded = parse_json_object(raw)
  scope = "self"
  reason = ""
  if decoded is not None:
    scope = json_str(decoded, "scope").lower() or "self"
    reason = json_str(decoded, "reasoning")
  if scope not in ("self", "wider"):
    match = _SCOPE_RE.search(strip_code_fence(raw))
    scope = match.group(1).lower() if match else "self"
  return scope, reason


def granularity_unit(
  ctx: SearchContext, query: str, leaf: TreeNode, memory: list[str]
) -> tuple[TreeNode, bool, list[GranularityStep]]:
  """Climb from a taken passage to the right granularity.

  Keeps escalating to the parent while the agent says the answer depends on the wider
  section, capped at the document boundary so it never grabs a sibling document. This is
  what stops an answer losing a required phrase that sat a few sentences from the single
  highest-scoring passage.

  Args:
    ctx: The search context.
    query: The question being answered.
    leaf: The passage that was taken.
    memory: Working memory, read but not modified.

  Returns:
    A triple of the node to take, a flag that is True when the climb escalated, and the
    escalation decisions in order.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  node = leaf
  escalations: list[GranularityStep] = []
  while True:
    parent = ctx.index.parent_of(node)
    if (
      parent is None or parent.node_type == "root" or not ctx.index.single_source(parent)
    ):
      break
    scope, reason = scope_choice(ctx, query, node, parent, memory)
    escalations.append(
      GranularityStep(
        from_node=node.name, to_node=parent.name, scope=scope, reasoning=reason
      )
    )
    if scope != "wider":
      break
    node = parent
    if node.is_file():
      break
  return node, node.node_id != leaf.node_id, escalations


# ---------------------------------------------------------------------------
# intra-file triage sweep
# ---------------------------------------------------------------------------


def triage_select(
  ctx: SearchContext,
  query: str,
  candidates: list[tuple[TreeNode, float]],
  evidence: list[TreeNode],
  anchor: float,
) -> list[tuple[TreeNode, float]]:
  """Make one agentic pass over every remaining same-file candidate.

  ``anchor`` is the score the search committed to for the evidence already kept. The
  prompt asks the agent to select sections scoring roughly as high as that evidence - but
  the number was never shown to it, so its only scale was the remaining sections compared
  to each other, and relative to a residue the residue's own top always looks high.
  Showing the anchor lets it conclude that NOTHING here belongs.

  Args:
    ctx: The search context.
    query: The question being answered.
    candidates: The remaining same-file candidates with their scores, best first.
    evidence: The evidence already kept.
    anchor: The reference score the selection is judged against.

  Returns:
    The candidates the agent wants read in full.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  have = (
    clip("\n\n".join(full(e.content or e.summary) for e in evidence), 2200)
    or "(nothing yet)"
  )
  selected: list[tuple[TreeNode, float]] = []
  batch_size = ctx.config.triage_batch
  top_kept = max((score for _, score in candidates), default=0.0)
  for start in range(0, len(candidates), batch_size):
    batch = candidates[start : start + batch_size]
    lines = "\n".join(
      f"[{i}] ({score:.2f}) {node.name} — {clip(full(node.content or node.summary), 110)}"
      for i, (node, score) in enumerate(batch)
    )
    reference = (
      f"(For reference: the evidence already kept was ranked {anchor:.2f}, and the "
      f"highest remaining score here is {top_kept:.2f}.) Judge the remaining sections "
      "against the evidence you kept, NOT against each other — the strong sections of "
      "this document are already taken, so what is left may be nothing but its weakest "
      "material, and the top of that residue is not thereby relevant. Select "
      "generously when sections score close to the kept evidence; select NOTHING when "
      "they all sit far below it and the question is already answered.\n"
      if anchor
      else f"(For reference, the highest remaining score is {top_kept:.2f}.) Select "
      "generously when high-scoring sections cluster; select little only when the "
      "remainder is clearly unrelated.\n"
    )
    prompt = (
      "You just took evidence from a document you deliberately navigated to. Below is "
      "everything ELSE in that SAME document, each with the relevance score it earned "
      "during navigation and its first line. Your job now is to FINISH this document: "
      "select every section that plausibly belongs to the answer before the search "
      "moves on to other documents.\n\n"
      f"QUESTION: {query}\n\n"
      f"EVIDENCE ALREADY KEPT:\n{have}\n\n"
      f"REMAINING SECTIONS OF THIS DOCUMENT (score — heading — first line):\n{lines}\n\n"
      "How to choose:\n- Every section here is inside a document already judged "
      "relevant, and each score is how relevant navigation found it. Sections scoring "
      "roughly as high as the evidence you just kept are very likely part of the SAME "
      "answer — ESPECIALLY when the answer is a multi-step procedure, an enumerated "
      "list, or a set of components / requirements / criteria that runs across several "
      "consecutive sections. When a cluster of high-scoring sections sits together, "
      "that is the signature of an answer spanning the whole section: select them ALL, "
      "not just the first one you already took.\n- Reading further WITHIN this one "
      "document is cheap; the limited-time budget you are protecting applies to OTHER "
      "documents, which are handled separately. So err toward inclusion here rather "
      "than leaving relevant sections behind.\n- Skip a section ONLY when it is "
      "clearly off-topic, boilerplate, or fully redundant with evidence already kept. "
      "A near-floor score with an unrelated heading is a skip.\n- A section about a "
      "NEIGHBOURING subject is not part of this answer. Documents routinely cover "
      "several sibling subjects (different assays, programmes, instruments, "
      "processes); a section stating the asked fact — or stating that it does not "
      "apply — for a DIFFERENT one of them does not answer the question and must not "
      "be selected, however similar it reads. Select it only if it is about the "
      "subject "
      "the question actually names.\n"
      + reference
      + 'Reply ONLY json: {"reasoning":"<1 sentence>", "read":[<indices>]}'
    )
    raw = ctx.llm(prompt, num_predict=max(220, 12 * len(batch)))
    decoded = parse_json_object(raw)
    indices: list[int] = []
    if decoded is not None:
      raw_read = decoded.get("read")
      if isinstance(raw_read, list):
        for value in raw_read:
          if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
          try:
            index = int(value)
          except (TypeError, ValueError):
            continue
          if 0 <= index < len(batch):
            indices.append(index)
    for index in sorted(set(indices)):
      selected.append(batch[index])
  return selected


def read_selection(
  ctx: SearchContext,
  query: str,
  node: TreeNode,
  evidence: list[TreeNode],
  *,
  fallback: bool = False,
  below_bar: bool = False,
) -> tuple[str, str, str]:
  """Read one selected section as a whole unit and decide take or skip.

  ``fallback`` marks a piece the navigator itself ranked BELOW the bar and deferred, now
  being read only because the sufficiency gate called the kept evidence incomplete.
  ``below_bar`` marks any other piece ranked materially below the evidence already kept.
  Both are the same tier and earn admission the same stricter way: material still on the
  floor is, by the navigator's own ranking, the weakest in the document, and the piece
  most likely to mention the question's subject without governing it. They differ only in
  what an unparsed response defaults to - a below-bar override that was never justified
  must not happen at all, whereas a triage selection already has two signals saying it is
  relevant.

  Args:
    ctx: The search context.
    query: The question being answered.
    node: The section to read.
    evidence: The evidence already kept.
    fallback: Mark this as a deferred piece recovered after an insufficiency verdict.
    below_bar: Mark this as ranked materially below the evidence already kept.

  Returns:
    A triple of the decision (``take`` or ``skip``), what it adds when taken, and the
    reasoning.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  unit = node if node.is_leaf() else whole_unit(node)
  body = clip(full(unit.content or unit.summary), 3000)
  have = (
    clip("\n\n".join(full(e.content or e.summary) for e in evidence), 2400)
    or "(nothing yet)"
  )
  tier = ""
  if fallback or below_bar:
    tier = (
      "\nIMPORTANT — this section was ranked BELOW the evidence already kept"
      + (
        ", and is being read only because that evidence was judged incomplete."
        if fallback
        else ", which already states an answer to the question."
      )
      + " It earns a place only by supplying what is actually missing. Two traps:\n"
      "- Mentioning the question's subject is NOT answering about it. If it names the "
      "subject near the topic but does not state the asked fact, 'skip'.\n- If it "
      "CONTRADICTS evidence already kept (e.g. asserts the general rule does not "
      "apply, or that something is handled differently), take it ONLY if it is "
      "unmistakably about the SAME subject the question asks about and genuinely "
      "states an exception that overrides. If it concerns a neighbouring subject, a "
      "different programme, or a special case, 'skip' — a lower-ranked passage must "
      "not overturn a direct statement of the answer already in hand, and a fact "
      "stated about a NEIGHBOURING subject is not a fact about the one asked for.\n"
    )
  prompt = (
    "You selected this section of the document as worth reading. Having read it, "
    "decide whether it ADDS distinct information the answer needs — a required detail, "
    "frequency, qualifier, definition, version, location, rule, OR a further step / "
    "component / criterion of the same procedure or list the question is asking for. A "
    "distinct item of an enumerated answer counts as adding information even if the "
    "kept evidence already has other items. Only 'skip' true repeats or genuinely "
    "off-topic text.\n" + tier + "\n"
    f"QUESTION: {query}\n\n"
    f"EVIDENCE ALREADY KEPT:\n{have}\n\n"
    f"SECTION ({node.name}):\n{body}\n\n"
    'Reply ONLY json: {"reasoning":"<1 sentence>", "decision":"take|skip", '
    '"adds":"<if take: what it adds, <=12 words>"}'
  )
  # A generous budget on purpose: the body runs to 3000 characters and the kept evidence
  # to 2400, and "decision" is the LAST key in the schema, so a tight budget exhausts
  # itself in the reasoning channel and truncation destroys the decision first.
  raw = ctx.llm(prompt, num_predict=2048)
  decoded = parse_json_object(raw)
  decision = ""
  adds = ""
  why = ""
  parsed = False
  if decoded is not None:
    decision = json_str(decoded, "decision").lower()
    adds = json_str(decoded, "adds")
    why = json_str(decoded, "reasoning")
    parsed = decision in ("take", "skip")
  if not parsed:
    match = _DECISION_RE.search(strip_code_fence(raw))
    if match:
      decision = match.group(1).lower()
      parsed = True
  if not parsed:
    decision = "skip" if fallback else "take"
    why = (
      "[unparsed model response; not admitted — a below-bar section must be "
      "affirmatively justified to override kept evidence]"
      if fallback
      else "[unparsed model response; kept because the ranker scored this section "
      "relevant]"
    )
  return decision, adds, why


def intra_file_sweep(
  ctx: SearchContext,
  query: str,
  file_node: TreeNode,
  state: AgentState,
  *,
  breadth: bool = False,
  anchor: float = 0.0,
) -> list[SweepRecord]:
  """Finish the current document before the global frontier is consulted again.

  One triage pass shows the agent every remaining same-file candidate - heading, the
  descent score it already earned, and a snippet - and only its selections are read, each
  as a whole unit. The top cluster is read automatically, because the agentic opt-in
  reliably returned "nothing" even on a dense cluster of high-scoring paragraphs and
  dropped most of an enumerated answer. Candidates neither auto-read nor picked are marked
  visited so nothing teleports back into this file mid-search, but they are DEFERRED
  rather than destroyed, so the sufficiency gate can still pull the best of them back.

  Args:
    ctx: The search context.
    query: The question being answered.
    file_node: The document to finish.
    state: The traversal state, modified in place.
    breadth: Run the evict-only sweep synthesis mode uses instead of deep reading.
    anchor: The relevance score the line that reached this document committed to.

  Returns:
    The sweep's records, in the order they occurred.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  file_id = file_node.node_id
  records: list[SweepRecord] = []

  local: list[FrontierItem] = []
  kept: list[FrontierItem] = []
  for entry in state.frontier:
    (local if ctx.index.is_under(entry.node.node_id, file_id) else kept).append(entry)
  state.frontier[:] = kept
  kept_reserve: list[FrontierItem] = []
  for entry in state.reserve:
    (local if ctx.index.is_under(entry.node.node_id, file_id) else kept_reserve).append(
      entry
    )
  state.reserve[:] = kept_reserve

  if breadth:
    # Synthesis already captured this document's relevant unit as one whole piece. Deep
    # reading the rest is exactly the drilling this mode exists to avoid, so the sweep is
    # evict-only here: it reads nothing and makes no LLM call.
    for entry in local:
      state.visited.add(entry.node.node_id)
    for child in file_node.children:
      state.visited.add(child.node_id)
    record = SweepBreadthEvict(node=file_node.name, evicted_same_file=len(local))
    records.append(record)
    return records

  candidates: list[tuple[TreeNode, float]] = [(e.node, e.score) for e in local]
  queued = {node.node_id for node, _ in candidates}
  fresh = [
    c
    for c in file_node.children
    if c.node_id not in state.visited
    and c.node_id not in state.seen
    and c.node_id not in queued
  ]
  if fresh:
    for node, score, _ in rank_children(ctx, query, fresh, state.memory):
      candidates.append((node, score))
      queued.add(node.node_id)
  candidates = [
    (n, s)
    for n, s in candidates
    if n.node_id not in state.visited and n.node_id not in state.seen
  ]
  if not candidates:
    return records

  floor = ctx.config.noise_floor
  low = [(n, s) for n, s in candidates if s < floor]
  for node, _ in low:
    state.visited.add(node.node_id)
  candidates = [(n, s) for n, s in candidates if s >= floor]
  if low:
    records.append(SweepFloorDismissed(count=len(low)))
  if not candidates:
    return records

  candidates.sort(key=lambda item: item[1], reverse=True)
  records.append(
    SweepTriaged(
      candidates=tuple(
        RankedCandidate(node=n.name, score=round(s, 3)) for n, s in candidates
      )
    )
  )

  # The cluster band is anchored on the score the search actually committed to, not on the
  # top of the residue. Anchored on the residue it re-normalises: after a 0.95 take, a
  # floor of 0.49/0.47/0.45 leftovers all sit within the delta of 0.49 and get auto-read
  # as though co-equal. max() keeps it monotone, so a genuine high cluster is unaffected.
  top = candidates[0][1] if candidates else 0.0
  reference = max(top, anchor or 0.0)
  delta = ctx.config.cluster_delta
  high = ctx.config.cluster_high
  auto = [(n, s) for n, s in candidates if s >= reference - delta or s >= high]
  auto_ids = {n.node_id for n, _ in auto}
  ask = [(n, s) for n, s in candidates if n.node_id not in auto_ids]
  selected = triage_select(ctx, query, ask, state.evidence, reference) if ask else []
  read_list = auto + list(selected)
  records.append(
    SweepSelection(
      auto_selected=tuple(
        RankedCandidate(node=n.name, score=round(s, 3)) for n, s in auto
      ),
      selected=tuple(
        RankedCandidate(node=n.name, score=round(s, 3)) for n, s in selected
      ),
      anchor=round(reference, 3),
    )
  )

  read_ids = {n.node_id for n, _ in read_list}
  deferred_count = 0
  for node, score in candidates:
    if node.node_id in read_ids:
      continue
    state.visited.add(node.node_id)
    if score >= floor:
      state.deferred.append((node, score))
      deferred_count += 1
  if deferred_count:
    records.append(SweepDeferred(count=deferred_count))

  for node, score in sorted(read_list, key=lambda item: item[1], reverse=True):
    if len(state.evidence) >= ctx.config.max_evidence:
      break
    if node.node_id in state.visited or node.node_id in state.seen:
      continue
    state.visited.add(node.node_id)
    # A candidate that is NOT co-equal to the committed score is the same tier as a
    # deferred one, so it gets the subject-scope guard rather than the purely additive
    # criterion. Only meaningful once there IS an account it could displace.
    below = bool(state.evidence) and score < reference - delta
    decision, adds, why = read_selection(
      ctx, query, node, state.evidence, below_bar=below
    )
    is_auto = node.node_id in auto_ids
    if decision == "take" and state.collect(node, whole=not node.is_leaf()):
      read_record = SweepRead(
        node=node.name,
        score=round(score, 3),
        decision="take",
        adds=adds,
        reasoning="",
        auto=is_auto,
      )
    else:
      read_record = SweepRead(
        node=node.name,
        score=round(score, 3),
        decision="skip",
        adds="",
        reasoning=why,
        auto=is_auto,
      )
    records.append(read_record)
  return records


# ---------------------------------------------------------------------------
# the sufficiency gate
# ---------------------------------------------------------------------------


def evidence_would_answer(
  ctx: SearchContext,
  query: str,
  evidence: list[TreeNode],
  *,
  unread: list[str] | None = None,
  polarity: bool = False,
  definitional: bool = False,
  recency: bool = False,
) -> SufficiencyVerdict:
  """Decide whether the evidence in hand would actually produce an answer.

  This is the only thing between a premature "could not be found" and continuing to look.
  It fires only when the agent is about to answer, and the search remains bounded by the
  step and file caps, so the not-found string is emitted only once every avenue is spent.

  A model that names a missing part but still ticks sufficient is contradicting itself, so
  the missing list is believed and the search continues; the same applies to a definition
  question whose reported basis is entirely incidental.

  Args:
    ctx: The search context.
    query: The question being answered.
    evidence: The evidence collected so far.
    unread: Headings of unread same-document sections to show the gate.
    polarity: Apply the universal/existence criteria instead of the strict-match ones.
    definitional: Also demand provenance for a "what is X" question.
    recency: Also demand the most recent entry of a tracked series.

  Returns:
    The gate's verdict.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  if not evidence:
    return SufficiencyVerdict(False, "no evidence yet", (), "")

  groups: dict[str, list[str]] = {}
  order: list[str] = []
  for item in evidence[: ctx.config.max_evidence]:
    key = item.source_file() or item.path or item.name
    if key not in groups:
      groups[key] = []
      order.append(key)
    groups[key].append(full(item.content or item.summary))
  docs = "\n\n".join(f"[{k}]\n" + "\n".join(groups[k]) for k in order)

  criteria = POLARITY_CRITERIA if polarity else STRICT_CRITERIA
  wants_basis = definitional and not polarity
  if wants_basis:
    criteria += PROVENANCE_CRITERIA
  if recency and not polarity:
    criteria += RECENCY_CRITERIA

  unread_block = ""
  if unread:
    unread_block = (
      "SECTIONS OF THE SAME DOCUMENT YOU HAVE NOT BEEN GIVEN:\n"
      + "\n".join(f"  - {name}" for name in unread)
      + UNREAD_SECTIONS_NOTE
    )

  prompt = (
    "Decide whether the evidence below is SUFFICIENT to actually answer the question. "
    "Simulate answering using ONLY this evidence: if you can state a real, substantive "
    f"answer, say yes; if you would have to reply '{DUNNO}' or could only give a "
    "vague/partial non-answer, say no.\n"
    + criteria
    + f"\nQUESTION: {query}\n\nEVIDENCE:\n{docs}\n\n"
    + unread_block
    + "Give ONE sentence of reasoning FIRST (state which parts of the question the "
    "evidence covers and which, if any, are still missing), then answer. Reply ONLY "
    "json:\n"
    '{"reasoning":"<1 sentence>", '
    + ('"basis":"subject|incidental|none", ' if wants_basis else "")
    + '"missing":["<part still unanswered>"], "sufficient":true|false}'
  )
  raw = ctx.llm(prompt, num_predict=2048)
  decoded = parse_json_object(raw)
  if decoded is None:
    match = _SUFFICIENT_RE.search(strip_code_fence(raw))
    return SufficiencyVerdict(
      bool(match and match.group(1).lower() == "true"), "", (), ""
    )

  sufficient = bool(decoded.get("sufficient", False))
  missing = json_str_list(decoded, "missing")
  if sufficient and missing:
    sufficient = False
  basis = json_str(decoded, "basis").lower()
  if definitional and basis == "incidental":
    sufficient = False
    if not missing:
      missing = [
        "a document whose own subject is this entity: what it is and what it is for"
      ]
  reasoning = json_str(decoded, "reasoning")
  if missing:
    reasoning += f"  [still missing: {'; '.join(missing)}]"
  return SufficiencyVerdict(sufficient, reasoning, tuple(missing), basis)


# ---------------------------------------------------------------------------
# decomposition, steering directives and the retrieval hand-off
# ---------------------------------------------------------------------------


def decompose_parts(ctx: SearchContext, query: str) -> tuple[list[str], str]:
  """Name the distinct sub-questions a complete answer must settle, and the subject.

  One LLM call. Most stems ask exactly one thing and return one part, which makes every
  compound mechanism downstream a no-op; a stem that genuinely asks several things returns
  one part per ask. The subject rides along on the same call, so this is cost-neutral.

  Args:
    ctx: The search context.
    query: The question being answered.

  Returns:
    A pair of the sub-questions, capped at ``config.parts_max``, and the question's one
    specific subject - empty when it asks about a category or process in general.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  raw = ctx.llm(
    "Break the question into the DISTINCT sub-questions a complete answer must settle "
    "— one entry per separate thing being asked. MOST questions ask exactly ONE thing; "
    "return a single entry then. Split ONLY where the stem really does ask separate "
    "things, each with its own answer (usually joined by 'and' / 'or' / commas). Never "
    "split one ask into restatements of itself.\nAlso name the question's SPECIFIC "
    "SUBJECT: the ONE particular, individuated thing (a named instrument, unit, "
    "system, assay, site, document, version) whose fact is being asked, with every "
    "identifying qualifier the stem gives (its name, model, location, "
    'the system it serves). Leave it "" when the question asks about a '
    "category, process, policy or rule in general rather than one specific "
    "instance that could be confused with same-kind siblings.\n\n"
    f"QUESTION: {query}\n\n"
    'Reply ONLY json: {"parts":["<sub-question, <=12 words>", ...], '
    '"subject":"<the one specific thing asked about, else empty>"}  '
    f"(at most {ctx.config.parts_max} parts)",
    num_predict=320,
  )
  decoded = parse_json_object(raw)
  if decoded is None:
    return [], ""
  parts: list[str] = []
  for value in json_str_list(decoded, "parts"):
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    if cleaned and cleaned not in parts:
      parts.append(cleaned)
  subject = re.sub(r"\s+", " ", json_str(decoded, "subject")).strip(" .")
  return parts[: ctx.config.parts_max], subject


def set_residual(memory: list[str], missing: tuple[str, ...], parts_max: int) -> None:
  """Pin, or clear, the note re-aiming the search at what the gate says is still open.

  The gate already reports which parts the evidence does not settle, and that list used to
  be read only for its truthiness. Pinning it into working memory - which the ranker and
  every evidence judge already read - means a candidate is scored on whether it settles
  what is still OPEN, rather than on similarity to a stem that is mostly answered.

  Clears only its own tagged line, so any other standing directive survives.

  Args:
    memory: Working memory, modified in place.
    missing: The unsettled parts; empty clears the note.
    parts_max: How many parts to name.
  """
  memory[:] = [m for m in memory if not m.startswith(RESIDUAL_TAG)]
  if not missing:
    return
  rendered = "; ".join(
    f"({i}) {clip(part, 120)}" for i, part in enumerate(missing[:parts_max], 1)
  )
  memory.insert(0, residual_directive(rendered))


def set_contrast(memory: list[str], evidence: list[TreeNode] | None) -> None:
  """Pin, or clear, the directive that re-aims ranking during a contrast check.

  Args:
    memory: Working memory, modified in place.
    evidence: The evidence forming the account being challenged; ``None`` or empty clears
      the directive.
  """
  memory[:] = [m for m in memory if not m.startswith(CONTRAST_TAG)]
  if not evidence:
    return
  sources = evidence_sources(evidence)
  memory.insert(0, contrast_directive("; ".join(clip(s, 60) for s in sources[:4])))


def select_by_part_coverage(
  ctx: SearchContext, query: str, parts: list[str], evidence: list[TreeNode]
) -> tuple[list[TreeNode], dict[int, tuple[int, ...]] | None]:
  """Rebuild the evidence bundle part by part, each part leading with what settles it.

  On a compound question every part is settled in a different document at a different
  step, and the bundle handed over is whatever accumulated, in collection order, with the
  piece settling the LAST part at the bottom because it was found last. The answer,
  written once to a word limit, then states the parts it meets first and drops the rest -
  and retrieval is marked down for a part it did retrieve. This is evidence SELECTION, the
  same job a top-k does; the answer prompt and its single call are untouched.

  Conservative by construction: pieces are dropped only when the agent placed at least one
  against EVERY part, and any parse failure hands the bundle back untouched.

  Args:
    ctx: The search context.
    query: The question being answered.
    parts: The sub-questions a complete answer must settle.
    evidence: The evidence collected so far.

  Returns:
    A pair of the rebuilt evidence and the per-part index map, or the original evidence
    and ``None`` when the bundle was handed back untouched.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  lines = "\n".join(
    "[%d] (%s) %s"
    % (
      i,
      clip(str(e.source_file() or e.path or e.name), 60),
      clip(full(e.content or e.summary), 300),
    )
    for i, e in enumerate(evidence)
  )
  part_list = "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
  raw = ctx.llm(
    "The search is finished. Below are the PARTS of the question and every piece of "
    "evidence that was collected for it. Say which pieces settle which part.\n\n"
    f"QUESTION: {query}\n\nPARTS:\n{part_list}\n\nEVIDENCE PIECES:\n{lines}\n\n"
    f"For each part, list the piece indices that STATE its answer, best first, at most "
    f"{ctx.config.part_keep} per part. A piece that merely mentions the topic without "
    "stating the answer is NOT a match — leave it out. One piece may serve several "
    "parts. Leave a part's list empty only if nothing here settles it.\n"
    'Reply ONLY json: {"coverage":{"1":[<indices>], ...}}',
    num_predict=512,
  )
  decoded = parse_json_object(raw)
  if decoded is None:
    return evidence, None
  coverage = decoded.get("coverage")
  if not isinstance(coverage, dict):
    return evidence, None

  picked: list[int] = []
  used: set[int] = set()
  per_part: dict[int, tuple[int, ...]] = {}
  for part_number in range(1, len(parts) + 1):
    got: list[int] = []
    raw_indices = coverage.get(str(part_number))
    if not isinstance(raw_indices, list):
      raw_indices = []
    for value in raw_indices[: ctx.config.part_keep]:
      if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        continue
      try:
        index = int(value)
      except (TypeError, ValueError):
        continue
      if not 0 <= index < len(evidence):
        continue
      got.append(index)
      if index not in used:
        used.add(index)
        picked.append(index)
    per_part[part_number] = tuple(got)
  if not picked:
    return evidence, None
  if not all(per_part[key] for key in per_part):
    picked += [i for i in range(len(evidence)) if i not in used]
  return [evidence[i] for i in picked], per_part


# ---------------------------------------------------------------------------
# agentic teleport choices
# ---------------------------------------------------------------------------


def choose_alternative(
  ctx: SearchContext,
  query: str,
  evidence: list[TreeNode],
  candidates: list[FrontierItem],
) -> tuple[FrontierItem | None, str]:
  """Pick the unexplored document most likely to CONTRADICT or SHARPEN the held account.

  Choosing by raw score spends this budget confirming the hypothesis it exists to
  challenge: score is produced by the same ranker reading the same question under the
  framing that produced the first commitment, so the highest-scoring unexplored candidate
  is almost always a sibling of the committed account. Every eligible alternative is
  therefore shortlisted, and the agent chooses on their own words.

  Args:
    ctx: The search context.
    query: The question being answered.
    evidence: The evidence forming the account being challenged.
    candidates: The shortlisted alternatives.

  Returns:
    A pair of the chosen alternative - ``None`` when the agent judged that every candidate
    would merely repeat the account in hand - and its reasoning.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  sources = evidence_sources(evidence)
  account = (
    clip("\n\n".join(full(e.content or e.summary) for e in evidence), 2000) or "(nothing)"
  )
  lines = "\n".join(
    alternative_entry(ctx, i, entry.node, entry.score, query)
    for i, entry in enumerate(candidates)
  )
  prompt = (
    "You are about to answer the question using only the evidence below. Before you "
    "commit, you get ONE look at an unexplored document. Spend it on whichever would "
    "most likely CONTRADICT or SHARPEN what you have.\n\nWhy this matters: the "
    "evidence you hold may not be the answer but a plausible NEIGHBOUR of it — the "
    "general category the question falls under, or a record, form or instance of that "
    "category. Such material reads like a complete answer, while the document that "
    "actually governs the SPECIFIC subject the question names states the fact more "
    "precisely, and sometimes differently. You cannot detect that from the evidence "
    "itself; only a different document can show it.\n\n"
    f"QUESTION: {query}\n\n"
    f"EVIDENCE YOU HOLD (from: {', '.join(clip(s, 60) for s in sources)}):\n{account}\n\n"
    f"UNEXPLORED CANDIDATES:\n{lines}\n\n"
    "Choose the ONE candidate most likely to change or sharpen your answer:\n- Prefer "
    "whatever is most likely to be ABOUT the exact subject or process the question "
    "names, and to state the asked fact for it directly.\n- Do NOT choose another "
    "document of the SAME KIND as one you already read — a further record, form, "
    "template, or another copy of the same procedure. More of the same account cannot "
    "change that account, however high its score. A LOWER-scored candidate that "
    "addresses the named subject head-on is a better use of this one look than a "
    "high-scored sibling of what you already have.\n- AGREEMENT IS NOT CONTRAST: a "
    "candidate whose shown text states the SAME value or fact you already hold cannot "
    "contradict or sharpen it — reading it would merely corroborate, and same-kind "
    "documents about sibling entities routinely repeat the same figures, so agreement "
    "proves nothing about the subject the question names. If your evidence's tie to "
    "that exact subject is unproven, spend the look on the candidate most likely to be "
    "about the NAMED subject itself — often from a part of the tree not yet searched — "
    "rather than on any restatement, however topical its wording.\n- Judge on each "
    "candidate's own words and contents where shown, NOT on its score. The scores were "
    "produced by the same reading of the question that led you to the evidence above, "
    "so they favour more of the same.\nAnswer -1 ONLY if every candidate listed would "
    "merely repeat the account you already hold.\n"
    'Reply ONLY json: {"reasoning":"<1 short sentence>", "choice":<index or -1>}'
  )
  raw = ctx.llm(prompt, num_predict=512)
  choice, why = parse_choice(raw)
  if choice is None:
    # A missing decision must not become a stop: fall back to the top alternative.
    return candidates[
      0
    ], "[unparsed contrast response; checking the top-scored alternative]"
  if 0 <= choice < len(candidates):
    return candidates[choice], why
  return None, why or "no listed candidate could give a different account"


def choose_by_residual(
  ctx: SearchContext,
  query: str,
  missing: tuple[str, ...],
  evidence: list[TreeNode],
  candidates: list[FrontierItem],
) -> tuple[FrontierItem, str]:
  """Pick the frontier entry most likely to settle what the gate says is still missing.

  Every frontier score was produced before any evidence existed, so it ranks a candidate
  on how much it RESEMBLES THE QUESTION - and the candidates resembling it most are the
  same kind of document whose partial account is already in hand. Letting the agent choose
  against the named gap is what stops each teleport landing on another sibling of what was
  just read.

  Args:
    ctx: The search context.
    query: The question being answered.
    missing: The parts the gate says are unsettled.
    evidence: The evidence collected so far.
    candidates: The shortlisted frontier entries.

  Returns:
    A pair of the chosen entry and its reasoning. Falls back to the top-scored entry when
    the model returns nothing usable, so a missing decision degrades to score priority
    rather than to a bad jump.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  sources = evidence_sources(evidence)
  have = (
    clip("\n\n".join(full(e.content or e.summary) for e in evidence), 1600)
    or "(nothing yet)"
  )
  gap = "; ".join(
    f"({i}) {clip(part, 120)}"
    for i, part in enumerate(missing[: ctx.config.parts_max], 1)
  )
  lines = "\n".join(
    alternative_entry(ctx, i, entry.node, entry.score, query)
    for i, entry in enumerate(candidates)
  )
  prompt = (
    "You are navigating a document tree and have just judged the evidence you hold "
    "INCOMPLETE. Choose the ONE unexplored candidate to go to next.\n\n"
    f"QUESTION: {query}\n\n"
    f"STILL MISSING (your own verdict on the evidence below): {gap}\n\n"
    f"DOCUMENTS ALREADY READ: {', '.join(clip(s, 60) for s in sources) or '(none)'}\n"
    f"WHAT THEY GAVE YOU:\n{have}\n\n"
    f"UNEXPLORED CANDIDATES:\n{lines}\n\n"
    "Choose on ONE criterion: which candidate is most likely to STATE the missing part "
    "above.\n- Judge on each candidate's own words, its contents, and what KIND of "
    "document it is — NOT on its score. The scores were produced by reading the WHOLE "
    "question before you had any evidence, so they rank candidates by how much they "
    "resemble the question, and the ones that resemble it most are the same kind of "
    "document that already gave you the part you HAVE. Going to another one returns "
    "the same partial account and leaves the same gap open.\n- The parts you already "
    "hold are settled; re-confirming them is worth nothing. A LOWER-scored candidate "
    "that plausibly states the MISSING part is a better jump than a high-scored "
    "sibling of what you have already read.\n- Prefer whatever would state the missing "
    "fact directly and specifically for the subject the question names — the document, "
    "table or specification that GOVERNS it — over one that discusses the topic "
    "generally around it. General overviews restate what you have; specific criteria, "
    "thresholds, values and conditions are what is missing.\n"
    'Reply ONLY json: {"reasoning":"<1 short sentence>", "choice":<index>}'
  )
  raw = ctx.llm(prompt, num_predict=512)
  choice, why = parse_choice(raw)
  if choice is None or not 0 <= choice < len(candidates):
    return candidates[0], "[unparsed residual choice; falling back to score-priority]"
  return candidates[choice], why


def remember_fact(memory: list[str], fact: str) -> None:
  """Add a fact the file judge asked to keep into working memory.

  Args:
    memory: Working memory, modified in place.
    fact: The fact to keep; ignored when empty.
  """
  if fact:
    add_memory(memory, fact)
