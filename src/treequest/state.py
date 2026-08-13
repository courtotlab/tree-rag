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

The mutable state of one traversal, and the frontier operations that act on it.

In the source notebook these were locals of ``run_agent`` and closures over them. They are
a dataclass here so the intra-file sweep and the evidence judges can be separate modules
without passing eleven positional arguments, and so nothing needs a module-level global.
"""

from collections import Counter
from dataclasses import dataclass, field, replace

from treequest.config import TreeRagConfig
from treequest.text import clip
from treequest.tree import TreeIndex, all_chunks, source_of, whole_unit
from treequest.types import TreeNode

#: Prefix marking a steering directive in working memory rather than a gathered fact.
_PIN = "!! "


@dataclass(slots=True)
class FrontierItem:
  """One runner-up remembered for a possible teleport.

  Attributes:
    node: The candidate node.
    score: Its relevance score; decayed in place when its subtree is demoted.
    reason: The ranker's reasoning sentence for the ranking that produced it.
    origin: Name of the node whose ranking produced it.
  """

  node: TreeNode
  score: float
  reason: str
  origin: str


def add_memory(memory: list[str], fact: str, cap: int = 20) -> None:
  """Add a gathered fact to working memory, evicting by recency but never a directive.

  On any real run the evidence facts overflow ``cap`` several times over, and an evicted
  directive silently stops steering every decision that reads memory - so pinned entries
  are exempt from the cap.

  Args:
    memory: The working memory list, modified in place.
    fact: The fact to add; clipped, and ignored when empty or already present.
    cap: Maximum number of unpinned facts to retain.
  """
  clipped = clip(fact, 500)
  if not clipped or clipped.lower() in ("", "none", "n/a") or clipped in memory:
    return
  memory.append(clipped)
  if len(memory) > cap:
    pinned = [m for m in memory if m.startswith(_PIN)]
    facts = [m for m in memory if not m.startswith(_PIN)]
    del facts[: -max(1, cap - len(pinned))]
    memory[:] = pinned + facts


def distinct_files(evidence: list[TreeNode]) -> int:
  """Count the distinct source documents the collected evidence spans.

  This is the breadth counter budgets are measured against, whereas ``len(evidence)``
  counts individual pieces - conflating the two once let one file's sweep exhaust the
  cross-file budget.

  Args:
    evidence: The evidence collected so far.

  Returns:
    The number of distinct source documents.
  """
  return len({(e.source_file() or e.path or e.name) for e in evidence})


def evidence_sources(evidence: list[TreeNode]) -> list[str]:
  """List the distinct source documents of the evidence, in collection order.

  Args:
    evidence: The evidence collected so far.

  Returns:
    The distinct source document identifiers, first appearance first.
  """
  out: list[str] = []
  for item in evidence:
    source = str(item.source_file() or item.path or item.name)
    if source not in out:
      out.append(source)
  return out


@dataclass(slots=True)
class AgentState:
  """Everything one traversal mutates as it runs.

  Attributes:
    config: The tuning knobs, so the frontier operations can consult the noise floor.
    index: The corpus tree indices.
    memory: Working memory: gathered facts, plus pinned steering directives.
    evidence: The evidence pieces collected so far.
    seen: Identifiers of nodes already collected as evidence.
    visited: Identifiers of nodes the traversal has consumed.
    deferred: Unread same-file candidates, recoverable if the evidence proves incomplete.
    frontier: The global teleport queue of above-floor runner-ups.
    reserve: Sub-floor runner-ups, drawn from only once the frontier is exhausted.
    reject_folders: Barren-file tally per ancestor folder, for the cluster-reject decay.
    committed_scores: Per source document, the score the search committed to it at.
    evidence_regions: Top-level regions the evidence has come from.
    entered_regions: Top-level regions the search has descended into or jumped to.
    line_entry_score: The folder/file score by which the current line was reached.
  """

  config: TreeRagConfig
  index: TreeIndex
  memory: list[str] = field(default_factory=list)
  evidence: list[TreeNode] = field(default_factory=list)
  seen: set[str] = field(default_factory=set)
  visited: set[str] = field(default_factory=set)
  deferred: list[tuple[TreeNode, float]] = field(default_factory=list)
  frontier: list[FrontierItem] = field(default_factory=list)
  reserve: list[FrontierItem] = field(default_factory=list)
  reject_folders: Counter[str] = field(default_factory=Counter)
  committed_scores: dict[str, float] = field(default_factory=dict)
  evidence_regions: set[str] = field(default_factory=set)
  entered_regions: set[str] = field(default_factory=set)
  line_entry_score: float = 1.0

  def collect(
    self, node: TreeNode, *, whole: bool, clip_chars: int | None = None
  ) -> bool:
    """Take a node as evidence, unless it has already been taken.

    Args:
      node: The node to collect.
      whole: Collapse the node's whole subtree into one unit carrying its full text.
      clip_chars: Clip the collected text to this many characters, when given.

    Returns:
      True when the node was newly collected, False when it was already held.
    """
    if node.node_id in self.seen or len(self.evidence) >= self.config.max_evidence:
      return False
    item = whole_unit(node) if whole else node
    if clip_chars and item.content and len(item.content) > clip_chars:
      item = replace(item, content=clip(item.content, clip_chars))
    self.evidence.append(item)
    self.seen.add(node.node_id)
    self.committed_scores.setdefault(source_of(node), self.line_entry_score)
    self.evidence_regions.add(self.index.region_of(node.node_id))
    if whole:
      for chunk in all_chunks(node):
        self.seen.add(chunk.node_id)
    if item.content:
      add_memory(self.memory, clip(item.content, 600))
    return True

  def push_frontier(self, items: list[tuple[TreeNode, float, str]], origin: str) -> None:
    """Remember a ranking's runner-ups for a possible teleport.

    Sub-floor candidates go to the reserve tier rather than the active frontier: they are
    the noise long runs were grinding through, but keeping them reachable preserves
    recall, so nothing is ever unreachable - only deprioritised.

    Args:
      items: The runner-ups, as node, score and reasoning triples.
      origin: Name of the node whose ranking produced them.
    """
    self.prune_frontiers()
    for node, score, reason in items:
      if node.node_id in self.visited or node.node_id in self.seen:
        continue
      if any(f.node.node_id == node.node_id for f in self.frontier):
        continue
      if any(f.node.node_id == node.node_id for f in self.reserve):
        continue
      entry = FrontierItem(node=node, score=score, reason=reason, origin=origin)
      if score >= self.config.noise_floor:
        self.frontier.append(entry)
      else:
        self.reserve.append(entry)

  def entry_available(self, entry: FrontierItem) -> bool:
    """Return whether an entry still names an undispatched, uncollected node."""
    node_id = entry.node.node_id
    return node_id not in self.visited and node_id not in self.seen

  def prune_frontiers(self) -> int:
    """Remove stale and duplicate entries from both frontier tiers.

    Active entries take precedence if a malformed queue contains the same node in both
    tiers. This operation is trajectory-neutral for valid pending entries.

    Returns:
      Number of stale or duplicate entries removed.
    """
    pending: set[str] = set()
    removed = 0
    for home in (self.frontier, self.reserve):
      kept: list[FrontierItem] = []
      for entry in home:
        node_id = entry.node.node_id
        if not self.entry_available(entry) or node_id in pending:
          removed += 1
          continue
        pending.add(node_id)
        kept.append(entry)
      home[:] = kept
    return removed

  def penalize_subtree(self, root_id: str) -> int:
    """Demote a rejected subtree so the search stops re-entering it.

    Moves the subtree's frontier entries into the reserve tier with a decayed score and
    decays any of its entries already in reserve. Repeated rejects in the same subtree
    compound the decay, so a subtree that keeps yielding nothing quickly sinks below
    fresh, never-tried branches. Nothing is removed, so recall is preserved.

    Args:
      root_id: Identifier of the subtree root to demote.

    Returns:
      How many frontier entries were moved to reserve.
    """
    decay = self.config.reject_decay
    for entry in self.reserve:
      if self.index.is_under(entry.node.node_id, root_id):
        entry.score *= decay
    kept: list[FrontierItem] = []
    moved = 0
    for entry in self.frontier:
      if self.index.is_under(entry.node.node_id, root_id):
        entry.score *= decay
        self.reserve.append(entry)
        moved += 1
      else:
        kept.append(entry)
    self.frontier[:] = kept
    return moved

  def pop_frontier(self) -> tuple[FrontierItem | None, bool]:
    """Take the highest-scoring node left anywhere on the frontier.

    Returns:
      A pair of the chosen entry - ``None`` when nothing is left - and a flag that is True
      when the entry came from the sub-floor reserve because the active frontier was
      exhausted.
    """
    self.prune_frontiers()
    if self.frontier:
      self.frontier.sort(key=lambda f: f.score, reverse=True)
      return self.frontier.pop(0), False
    if self.reserve:
      self.reserve.sort(key=lambda f: f.score, reverse=True)
      return self.reserve.pop(0), True
    return None, False

  def drop_frontier_entry(self, target: FrontierItem) -> bool:
    """Remove one specific entry from either frontier tier, by identity.

    Args:
      target: The entry to remove.

    Returns:
      True when the target was pending in either tier, otherwise False.
    """
    for home in (self.frontier, self.reserve):
      for position, entry in enumerate(home):
        if entry is target:
          del home[position]
          return True
    return False
