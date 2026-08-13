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

The agent's navigation trace, as a discriminated union of typed events rather than the
source notebook's untyped dicts. The agent already emitted exactly this information; only
its shape changed. :func:`render_event` turns an event into the one user-legible line the
reasoning panel shows, so the UI never renders raw JSON.

These events carry document and section NAMES, because naming what it is reading is the
point of a reasoning trace. They are for the user's screen only: nothing here is written
to a log file, a test fixture or an error message.
"""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from treerag.text import clip

# ---------------------------------------------------------------------------
# shared value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedCandidate:
  """One scored candidate in a ranking or frontier snapshot.

  Attributes:
    node: The candidate's display name.
    score: Its relevance score in ``[0, 1]``.
  """

  node: str
  score: float


@dataclass(frozen=True, slots=True)
class FrontierEntry:
  """One entry of a teleport frontier snapshot.

  Attributes:
    node: The candidate's display name.
    score: The relevance score it was pushed with.
    origin: Name of the node whose ranking produced it.
  """

  node: str
  score: float
  origin: str


@dataclass(frozen=True, slots=True)
class GranularityStep:
  """One escalation decision of the granularity climb.

  Attributes:
    from_node: The node currently held as evidence.
    to_node: Its parent, the wider unit being considered.
    scope: ``self`` to keep the passage, ``wider`` to climb to the parent.
    reasoning: The agent's one-sentence justification.
  """

  from_node: str
  to_node: str
  scope: str
  reasoning: str


# ---------------------------------------------------------------------------
# intra-file sweep records (nested inside a read_file event)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepTriaged:
  """The full candidate list a triage pass was shown.

  Attributes:
    candidates: Every remaining same-file candidate with its navigation score.
  """

  candidates: tuple[RankedCandidate, ...]


@dataclass(frozen=True, slots=True)
class SweepFloorDismissed:
  """Sub-floor same-file candidates dismissed without reading.

  Attributes:
    count: How many candidates were dismissed.
  """

  count: int


@dataclass(frozen=True, slots=True)
class SweepSelection:
  """What the sweep decided to read.

  Attributes:
    auto_selected: Candidates auto-read because they cluster with the committed score.
    selected: Candidates the agent additionally picked at triage.
    anchor: The reference score the cluster band was measured against.
  """

  auto_selected: tuple[RankedCandidate, ...]
  selected: tuple[RankedCandidate, ...]
  anchor: float


@dataclass(frozen=True, slots=True)
class SweepDeferred:
  """Candidates neither auto-read nor picked, kept recoverable.

  Attributes:
    count: How many candidates were deferred.
  """

  count: int


@dataclass(frozen=True, slots=True)
class SweepRead:
  """One section read during the sweep, and what was decided about it.

  Attributes:
    node: The section's name.
    score: Its navigation score.
    decision: ``take`` or ``skip``.
    adds: What the section adds, when taken.
    reasoning: Why it was skipped, when skipped.
    auto: Whether it was auto-read by the cluster band rather than picked at triage.
  """

  node: str
  score: float
  decision: str
  adds: str
  reasoning: str
  auto: bool


@dataclass(frozen=True, slots=True)
class SweepBreadthEvict:
  """The evict-only sweep synthesis mode runs instead of deep-reading a file.

  Attributes:
    node: The document whose remaining sections were evicted.
    evicted_same_file: How many frontier entries under it were dropped.
  """

  node: str
  evicted_same_file: int


SweepRecord: TypeAlias = (
  SweepTriaged
  | SweepFloorDismissed
  | SweepSelection
  | SweepDeferred
  | SweepRead
  | SweepBreadthEvict
)


# ---------------------------------------------------------------------------
# trace events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubjectAnchorEvent:
  """The question names one specific subject; it has been pinned into working memory.

  Attributes:
    event: Discriminator, always ``subject_anchor``.
    subject: The specific thing the question asks about.
  """

  subject: str
  event: Literal["subject_anchor"] = "subject_anchor"


@dataclass(frozen=True, slots=True)
class ModeEvent:
  """A question-shape detector reported its verdict.

  Attributes:
    event: Discriminator, always ``mode``.
    mode: Which detector reported - ``synthesis`` or ``definitional``.
    active: Whether the shape was detected.
    why: The detector's reason, empty when it did not fire.
  """

  mode: str
  active: bool
  why: str
  event: Literal["mode"] = "mode"


@dataclass(frozen=True, slots=True)
class LexicalSeedEvent:
  """Documents seeded onto the frontier by the lexical body scan.

  Attributes:
    event: Discriminator, always ``lexical_seed``.
    files: The seeded documents and their seed scores.
  """

  files: tuple[RankedCandidate, ...]
  event: Literal["lexical_seed"] = "lexical_seed"


@dataclass(frozen=True, slots=True)
class DescendEvent:
  """The agent ranked a node's children and descended into the best one.

  Attributes:
    event: Discriminator, always ``descend``.
    step: The step number this happened on.
    at: The node whose children were ranked.
    node_type: That node's type.
    chose: The child descended into.
    chose_score: Its relevance score.
    reasoning: The ranker's one-sentence justification.
    ranked: The scored candidate list, best first.
  """

  step: int
  at: str
  node_type: str
  chose: str
  chose_score: float
  reasoning: str
  ranked: tuple[RankedCandidate, ...]
  event: Literal["descend"] = "descend"


@dataclass(slots=True)
class ReadFileEvent:
  """The agent reached a document and decided whether to keep it as evidence.

  This event is emitted as soon as the take/reject decision is made, so the reasoning
  panel updates without waiting for the intra-file sweep. The sweep records are filled in
  afterwards on the same object, so the completed trace carries them nested here exactly
  as the source benchmark recorded them.

  Attributes:
    event: Discriminator, always ``read_file``.
    step: The step number this happened on.
    at: The document or passage that was read.
    node_type: Its node type.
    decision: ``take`` or ``reject``.
    reasoning: The agent's one-sentence justification.
    granularity: The escalation decisions of the granularity climb, if any.
    kept_unit: The unit finally kept, when the climb escalated.
    kept_scope: ``passage`` or ``section/file``.
    distributed_passages: Pieces kept when a distributed answer was preserved.
    swept_file: The document the intra-file sweep covered.
    intra_file_sweep: Sweep records from the take path.
    reject_sweep: Sweep records from the reject path.
  """

  step: int
  at: str
  node_type: str
  decision: str
  reasoning: str
  granularity: tuple[GranularityStep, ...] = ()
  kept_unit: str = ""
  kept_scope: str = ""
  distributed_passages: int = 0
  swept_file: str = ""
  intra_file_sweep: list[SweepRecord] = field(default_factory=list)
  reject_sweep: list[SweepRecord] = field(default_factory=list)
  event: Literal["read_file"] = "read_file"


@dataclass(frozen=True, slots=True)
class SufficiencyCheckEvent:
  """The sufficiency gate judged whether the evidence in hand would answer the question.

  Attributes:
    event: Discriminator, always ``sufficiency_check``.
    step: The step number this happened on.
    sufficient: Whether the evidence would produce a real answer.
    reasoning: The gate's one-sentence verdict.
    n_evidence: How many evidence pieces were judged.
    basis: For a definition question, whether the description is ``subject`` or
      ``incidental``.
    unread_same_file: How many unread same-file headings the gate was shown.
    after_deferred: Whether this verdict followed a deferred-section recovery read.
  """

  step: int
  sufficient: bool
  reasoning: str
  n_evidence: int
  basis: str = ""
  unread_same_file: int = 0
  after_deferred: bool = False
  event: Literal["sufficiency_check"] = "sufficiency_check"


@dataclass(frozen=True, slots=True)
class DeferredTakeEvent:
  """A deferred same-file section was recovered and kept.

  Attributes:
    event: Discriminator, always ``deferred_take``.
    step: The step number this happened on.
    node: The recovered section.
    score: Its navigation score.
    adds: What it adds to the evidence.
  """

  step: int
  node: str
  score: float
  adds: str
  event: Literal["deferred_take"] = "deferred_take"


@dataclass(frozen=True, slots=True)
class DeferredSkipEvent:
  """A deferred same-file section was recovered, read and rejected.

  Attributes:
    event: Discriminator, always ``deferred_skip``.
    step: The step number this happened on.
    node: The section that was read.
    score: Its navigation score.
    reasoning: Why it was not admitted.
  """

  step: int
  node: str
  score: float
  reasoning: str
  event: Literal["deferred_skip"] = "deferred_skip"


@dataclass(frozen=True, slots=True)
class ContrastExploreEvent:
  """The agent chose an alternative document to cross-check before answering.

  Attributes:
    event: Discriminator, always ``contrast_explore``.
    step: The step number this happened on.
    node: The alternative being cross-checked.
    score: Its frontier score.
    committed: The score committed to on the current line.
    n_eligible: How many alternatives the agent chose between.
    reasoning: Why this one was chosen.
  """

  step: int
  node: str
  score: float
  committed: float
  n_eligible: int
  reasoning: str
  event: Literal["contrast_explore"] = "contrast_explore"


@dataclass(frozen=True, slots=True)
class ContrastConfirmedEvent:
  """A contrast excursion changed nothing, confirming the committed account.

  Attributes:
    event: Discriminator, always ``contrast_confirmed``.
    step: The step number this happened on.
    n_evidence: How many evidence pieces are held.
  """

  step: int
  n_evidence: int
  event: Literal["contrast_confirmed"] = "contrast_confirmed"


@dataclass(frozen=True, slots=True)
class ContrastAbortEvent:
  """A contrast target's own children all ranked as noise, so the excursion stopped.

  Attributes:
    event: Discriminator, always ``contrast_abort``.
    step: The step number this happened on.
    at: The contrast target that collapsed on inspection.
    best_score: Its best child's score.
    n_evidence: How many evidence pieces are held.
  """

  step: int
  at: str
  best_score: float
  n_evidence: int
  event: Literal["contrast_abort"] = "contrast_abort"


@dataclass(frozen=True, slots=True)
class NoSignalAbortEvent:
  """A line of descent was abandoned because folder ranking found no signal.

  Attributes:
    event: Discriminator, always ``nosignal_abort``.
    step: The step number this happened on.
    at: The folder whose children all scored below the noise floor.
    best_score: The best child's score.
    target: Where the search jumped instead.
    target_score: That target's score.
  """

  step: int
  at: str
  best_score: float
  target: str
  target_score: float
  event: Literal["nosignal_abort"] = "nosignal_abort"


@dataclass(frozen=True, slots=True)
class BreadthEscapeEvent:
  """A persistent stall pushed the search into a top-level region it had not entered.

  Attributes:
    event: Discriminator, always ``breadth_escape``.
    step: The step number this happened on.
    node: The entry point of the unexplored region.
    score: Its frontier score.
  """

  step: int
  node: str
  score: float
  event: Literal["breadth_escape"] = "breadth_escape"


@dataclass(frozen=True, slots=True)
class ResidualTeleportEvent:
  """The agent chose the next jump against what the gate said was still missing.

  Attributes:
    event: Discriminator, always ``residual_teleport``.
    step: The step number this happened on.
    node: The chosen target.
    score: Its frontier score.
    missing: The parts of the question the evidence does not yet settle.
    n_shortlist: How many candidates the agent chose between.
    top_score: The best score on the shortlist.
    reasoning: Why this target was chosen.
  """

  step: int
  node: str
  score: float
  missing: tuple[str, ...]
  n_shortlist: int
  top_score: float
  reasoning: str
  event: Literal["residual_teleport"] = "residual_teleport"


@dataclass(frozen=True, slots=True)
class TeleportEvent:
  """The search jumped to the best node left on the global frontier.

  Attributes:
    event: Discriminator, always ``teleport``.
    step: The step number this happened on.
    origin: The node the search jumped from.
    frontier: A snapshot of the frontier at the moment of the jump.
    target: The jump target, or empty when the frontier was exhausted.
    target_score: The target's score, or ``None`` when there was no target.
  """

  step: int
  origin: str
  frontier: tuple[FrontierEntry, ...]
  target: str
  target_score: float | None
  event: Literal["teleport"] = "teleport"


@dataclass(frozen=True, slots=True)
class SynthesisBreadthEvent:
  """Synthesis mode reported how much of an enumeration it has gathered.

  Attributes:
    event: Discriminator, always ``synthesis_breadth``.
    step: The step number this happened on.
    n_files: Distinct documents gathered so far.
    n_evidence: Evidence pieces held.
    strong_left: Whether a comparably relevant unread document remains.
    best_new: The best unexplored new-source score, when one exists.
  """

  step: int
  n_files: int
  n_evidence: int
  strong_left: bool
  best_new: float | None
  event: Literal["synthesis_breadth"] = "synthesis_breadth"


@dataclass(frozen=True, slots=True)
class BudgetStopEvent:
  """Navigation stopped on its budget rather than because it was satisfied.

  Attributes:
    event: Discriminator, always ``budget_stop``.
    step: The step number reached.
    reason: Which limit was hit.
    elapsed_seconds: How long navigation ran.
    llm_calls: How many LLM calls it made.
    n_evidence: How many evidence pieces it had collected.
  """

  step: int
  reason: str
  elapsed_seconds: float
  llm_calls: int
  n_evidence: int
  event: Literal["budget_stop"] = "budget_stop"


@dataclass(frozen=True, slots=True)
class PartCoverageEvent:
  """The multi-part hand-off settled which evidence piece carries which sub-question.

  Attributes:
    event: Discriminator, always ``part_coverage``.
    parts: The sub-questions a complete answer must settle.
    kept: How many evidence pieces were kept.
    of: How many were collected in total.
    per_part: Evidence indices placed against each part, keyed by 1-based part number.
  """

  parts: tuple[str, ...]
  kept: int
  of: int
  per_part: dict[int, tuple[int, ...]]
  event: Literal["part_coverage"] = "part_coverage"


TraceEvent: TypeAlias = (
  SubjectAnchorEvent
  | ModeEvent
  | LexicalSeedEvent
  | DescendEvent
  | ReadFileEvent
  | SufficiencyCheckEvent
  | DeferredTakeEvent
  | DeferredSkipEvent
  | ContrastExploreEvent
  | ContrastConfirmedEvent
  | ContrastAbortEvent
  | NoSignalAbortEvent
  | BreadthEscapeEvent
  | ResidualTeleportEvent
  | TeleportEvent
  | SynthesisBreadthEvent
  | PartCoverageEvent
  | BudgetStopEvent
)


# ---------------------------------------------------------------------------
# human-readable rendering
# ---------------------------------------------------------------------------

_MODE_LABEL = {
  "synthesis": "Listing mode — gathering several documents rather than drilling one",
  "definitional": "Definition question — looking for the document *about* this, "
  "not records that merely mention it",
}


def _tail(reason: str, limit: int = 120) -> str:
  """Format an optional reasoning clause for a progress line.

  Args:
    reason: The agent's reasoning sentence, possibly empty.
    limit: Maximum characters to keep.

  Returns:
    The reasoning prefixed with an em dash, or the empty string when there is none.
  """
  text = clip(reason, limit)
  return f" — {text}" if text else ""


def render_event(event: TraceEvent) -> str | None:
  """Render one trace event as a single user-legible progress line.

  The wording deliberately avoids the agent's internal vocabulary: a teleport reads as
  "jumping to", a sufficiency check as "enough / not enough yet". Events that carry no
  news for a reader return ``None`` and are not shown.

  Args:
    event: The trace event to render.

  Returns:
    A Markdown line for the reasoning panel, or ``None`` when the event should be hidden.
  """
  match event:
    case SubjectAnchorEvent():
      return f"🎯 Question is about **{clip(event.subject, 100)}** specifically"
    case ModeEvent():
      if not event.active:
        return None
      return f"🧭 {_MODE_LABEL.get(event.mode, event.mode)}"
    case LexicalSeedEvent():
      names = ", ".join(c.node for c in event.files)
      return f"🔦 Distinctive wording points at: {clip(names, 160)}"
    case DescendEvent():
      return (
        f"📂 Opening **{clip(event.chose, 80)}** "
        f"({event.chose_score:.2f}){_tail(event.reasoning)}"
      )
    case ReadFileEvent():
      if event.decision == "take":
        return f"📄 Reading **{clip(event.at, 80)}** — useful{_tail(event.reasoning)}"
      return f"📄 Reading **{clip(event.at, 80)}** — not relevant{_tail(event.reasoning)}"
    case SufficiencyCheckEvent():
      if event.sufficient:
        return f"✅ Enough to answer{_tail(event.reasoning)}"
      return f"🔄 Not enough yet{_tail(event.reasoning)}"
    case DeferredTakeEvent():
      return f"➕ Went back for **{clip(event.node, 80)}**{_tail(event.adds, 90)}"
    case DeferredSkipEvent():
      return (
        f"➖ Checked **{clip(event.node, 80)}**, nothing new{_tail(event.reasoning, 90)}"
      )
    case ContrastExploreEvent():
      return (
        f"🔍 Cross-checking an alternative source: **{clip(event.node, 80)}**"
        f"{_tail(event.reasoning)}"
      )
    case ContrastConfirmedEvent():
      return "🔒 Cross-check agreed with what was found — answering"
    case ContrastAbortEvent():
      return f"🔒 **{clip(event.at, 80)}** held nothing relevant — answering"
    case NoSignalAbortEvent():
      return (
        f"↪️ Nothing relevant under **{clip(event.at, 60)}** — "
        f"jumping to **{clip(event.target, 60)}**"
      )
    case BreadthEscapeEvent():
      return f"🧭 Searching a different area: **{clip(event.node, 80)}**"
    case ResidualTeleportEvent():
      missing = clip("; ".join(event.missing), 120)
      target = clip(event.node, 80)
      return f"🔄 Still missing {missing} — jumping to **{target}**"
    case TeleportEvent():
      if not event.target:
        return "🛑 Nothing left to explore"
      return f"⤴️ Jumping to **{clip(event.target, 80)}**"
    case SynthesisBreadthEvent():
      return f"📚 {event.n_files} document(s) gathered so far"
    case PartCoverageEvent():
      return (
        f"🧩 Matched evidence to each part of the question ({event.kept} pieces kept)"
      )
    case BudgetStopEvent():
      return (
        f"⏱️ Stopped searching — {event.reason}. "
        f"Answering from the {event.n_evidence} piece(s) found so far."
      )
