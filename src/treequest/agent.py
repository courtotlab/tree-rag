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

The traversal: greedy descent with a teleport frontier, and no backtracking.

The agent descends the single best-ranked child at each level, pushing every runner-up
sibling onto ONE global frontier with its relevance score. On reaching a document it reads
it and decides, in its own words, whether to keep it as evidence. After a take it finishes
the enclosing document before anything else. Only when the sufficiency gate says the
evidence still is not enough does it teleport to the highest-scored node left on the
frontier - which can be a top-level folder, so a wrong first choice is fully recoverable.

This is a faithful port of the benchmarked implementation. Its decision logic is
unchanged; what changed is that state lives in objects rather than module globals, that
events are emitted through a callback as they occur so a UI can stream them, and that the
LLM calls underneath are bounded rather than retried forever.
"""

from dataclasses import dataclass, field

from treequest.context import SearchContext
from treequest.errors import SearchBudgetError
from treequest.events import (
  BreadthEscapeEvent,
  BudgetStopEvent,
  ContrastAbortEvent,
  ContrastConfirmedEvent,
  ContrastExploreEvent,
  DeferredSkipEvent,
  DeferredTakeEvent,
  DescendEvent,
  FrontierEntry,
  LexicalSeedEvent,
  ModeEvent,
  NoSignalAbortEvent,
  PartCoverageEvent,
  RankedCandidate,
  ReadFileEvent,
  ResidualTeleportEvent,
  SubjectAnchorEvent,
  SufficiencyCheckEvent,
  SynthesisBreadthEvent,
  TeleportEvent,
  TraceEvent,
)
from treequest.judging import (
  SufficiencyVerdict,
  choose_alternative,
  choose_by_residual,
  decompose_parts,
  evidence_would_answer,
  granularity_unit,
  intra_file_sweep,
  judge_file,
  read_selection,
  remember_fact,
  select_by_part_coverage,
  set_contrast,
  set_residual,
)
from treequest.prompts import (
  DEFINITION_DIRECTIVE,
  PIN,
  POLARITY_DIRECTIVE,
  RECENCY_DIRECTIVE,
  subject_directive,
)
from treequest.ranking import diverse_shortlist, rank_children
from treequest.shapes import (
  is_definitional_question,
  is_polarity_question,
  is_recency_question,
  is_synthesis_question,
)
from treequest.state import (
  AgentState,
  FrontierItem,
  add_memory,
  distinct_files,
)
from treequest.text import clip
from treequest.tree import source_of
from treequest.types import TreeNode


@dataclass(slots=True)
class TraversalResult:
  """What one traversal collected, before the answer is written.

  Attributes:
    evidence: The evidence pieces handed to answer assembly.
    memory: Working memory at the end of the run.
    steps: Node visits consumed.
    teleports: Frontier jumps taken.
    path: The traversal trail, as a readable arrow-separated string.
    trace: Every trace event, in order.
    synthesis: Whether the run used enumeration (breadth) mode.
  """

  evidence: list[TreeNode]
  memory: list[str]
  steps: int
  teleports: int
  path: str
  trace: list[TraceEvent]
  synthesis: bool = False
  parts: list[str] = field(default_factory=list)
  budget_stop: str = ""


def _short(node: TreeNode) -> str:
  """Abbreviate a node name for the traversal trail.

  Args:
    node: The node to abbreviate.

  Returns:
    The first 22 characters of its name.
  """
  return (node.name or "?")[:22]


def run_agent(ctx: SearchContext, question: str) -> TraversalResult:
  """Navigate the corpus tree and collect the evidence that answers a question.

  Args:
    ctx: The search context, carrying the config, client, tree and event sink.
    question: The question to answer.

  Returns:
    The traversal's evidence, counters and full trace.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted on any LLM call.
  """
  config = ctx.config
  state = AgentState(config=config, index=ctx.index)
  root = ctx.index.root

  trail: list[str] = ["root"]
  teleports = 0
  steps = 0
  last_take_file: TreeNode | None = None
  # The best relevance score committed to on the CURRENT line: the max descend score
  # since the last teleport, seeded by a teleport's own target score.
  descent_top = 0.0
  tie_explores = 0
  # The frontier entry the agent itself chose as the alternative worth checking. The old
  # guard picked one and then let score priority teleport elsewhere - usually straight
  # back into the family it had committed to - so its choice never steered the search.
  forced_next: FrontierItem | None = None
  contrast_active = False
  contrast_evidence_mark = 0
  definition_pushes = 0
  stall = 0
  nosignal_run = 0
  residual: tuple[str, ...] = ()

  # ---- one-time question analysis --------------------------------------------------
  parts: list[str] = []
  subject = ""
  parts, subject = decompose_parts(ctx, question)
  if parts:
    add_memory(state.memory, "The answer must cover: " + "; ".join(parts))

  if subject:
    # Inserted directly rather than through add_memory, whose clip would truncate a
    # directive of this length; the cap preserves pinned entries, so it survives.
    state.memory.insert(0, subject_directive(clip(subject, 160)))
    ctx.emit(SubjectAnchorEvent(subject=clip(subject, 160)))

  polarity = is_polarity_question(question)
  if polarity.matched:
    add_memory(state.memory, POLARITY_DIRECTIVE)

  synthesis = is_synthesis_question(question)
  ctx.emit(ModeEvent(mode="synthesis", active=synthesis.matched, why=synthesis.why))

  definitional_shape = is_definitional_question(question)
  definitional = definitional_shape.matched and not (
    synthesis.matched or polarity.matched
  )
  ctx.emit(
    ModeEvent(mode="definitional", active=definitional, why=definitional_shape.why)
  )
  if definitional:
    state.memory.insert(0, PIN + DEFINITION_DIRECTIVE)

  recency = is_recency_question(question)
  if recency.matched:
    add_memory(state.memory, RECENCY_DIRECTIVE)

  def gate(unread: list[str]) -> SufficiencyVerdict:
    """Run the sufficiency gate, honouring the entity-definition provenance budget.

    The provenance rule is passed through only while it still has pushes left: once it has
    forced the search onward ``defn_max_push`` times and no document actually about the
    entity has turned up, the corpus evidently holds none within reach, so the rule stands
    down rather than running the search to the step cap.

    Args:
      unread: Headings of unread same-document sections to show the gate.

    Returns:
      The gate's verdict.
    """
    nonlocal definition_pushes
    provenance_on = definitional and definition_pushes < config.defn_max_push
    verdict = evidence_would_answer(
      ctx,
      question,
      state.evidence,
      unread=unread,
      polarity=polarity.matched,
      definitional=provenance_on,
      recency=recency.matched,
    )
    if provenance_on and not verdict.sufficient and verdict.basis == "incidental":
      definition_pushes += 1
    return verdict

  # ---- start the descent at root ----------------------------------------------------
  current = root
  state.visited.add(root.node_id)

  seeds = ctx.index.lexical_seed_files(question, config)
  if seeds:
    state.push_frontier(seeds, "lexical-seed")
    ctx.emit(
      LexicalSeedEvent(
        files=tuple(
          RankedCandidate(node=node.name, score=round(score, 3))
          for node, score, _ in seeds
        )
      )
    )

  # The budget is checked between decisions, so a search that runs out of time stops
  # cleanly and answers from the evidence it holds rather than being interrupted. The
  # handler below is the backstop for a single decision that overruns outright - the
  # ranking of a very wide node is the case that can do it.
  budget_stop = ""
  try:
    while (
      steps < config.max_steps
      and len(state.evidence) < config.max_evidence
      and (synthesis.matched or distinct_files(state.evidence) < config.max_files)
    ):
      budget_stop = ctx.stop_reason()
      if budget_stop:
        break
      steps += 1
      state.entered_regions.add(ctx.index.region_of(current.node_id))
      kids = [
        c
        for c in current.children
        if c.node_id not in state.visited and c.node_id not in state.seen
      ]

      # Synthesis: at a document node, take the whole document as one unit rather than
      # drilling its sections. Folders still descend normally to reach the documents
      # inside.
      synth_file = synthesis.matched and current.is_file()
      if current.is_leaf() or not kids or synth_file:
        verdict = judge_file(
          ctx, question, current, state.memory, full_content=synth_file
        )
        step_record = ReadFileEvent(
          step=steps,
          at=current.name,
          node_type=current.node_type,
          decision=verdict.decision,
          reasoning=verdict.reasoning,
        )
        ctx.emit(step_record)
        remember_fact(state.memory, verdict.remember)

        if verdict.decision == "take":
          if current.is_leaf():
            unit, took_whole, escalations = granularity_unit(
              ctx, question, current, state.memory
            )
            # Cluster-preserving granularity: climbing to a whole unit collapses the
            # section into one blob and marks every descendant seen, which buries
            # co-equal answer-bearing passages. When the answer is spread across several
            # comparably high-scoring sibling passages, keep each of them as DISTINCT
            # evidence so every fact stays independently quotable.
            cluster = (
              [
                f
                for f in state.frontier
                if f.score >= config.cluster_high
                and f.node.node_type not in ("folder", "file")
                and f.node.node_id not in state.seen
                and ctx.index.is_under(f.node.node_id, unit.node_id)
              ]
              if took_whole
              else []
            )
            if cluster:
              state.collect(current, whole=False)
              kept = 0
              for entry in cluster:
                if state.collect(entry.node, whole=not entry.node.is_leaf()):
                  kept += 1
              state.frontier[:] = [
                f for f in state.frontier if f.node.node_id not in state.seen
              ]
              step_record.distributed_passages = kept + 1
            else:
              state.collect(unit, whole=took_whole)
            last_take_file = ctx.index.enclosing_file(current)
            if escalations:
              step_record.granularity = tuple(escalations)
              step_record.kept_unit = unit.name
              step_record.kept_scope = "section/file" if took_whole else "passage"
          else:
            state.collect(
              current,
              whole=True,
              clip_chars=config.synth_file_chars if synthesis.matched else None,
            )
            last_take_file = ctx.index.enclosing_file(current)
          trail.append(_short(current) + "*")

          # Finish the current document before any teleport. Never sweep a folder: that
          # would pull in other documents, which is the teleport's job.
          if last_take_file is not None and last_take_file.node_type == "folder":
            last_take_file = None
          if last_take_file is not None and len(state.evidence) < config.max_evidence:
            sweep = intra_file_sweep(
              ctx,
              question,
              last_take_file,
              state,
              breadth=synthesis.matched,
              anchor=descent_top,
            )
            if sweep:
              step_record.intra_file_sweep = sweep
              step_record.swept_file = last_take_file.name
          # A folder that just produced real evidence is not barren: clear the reject
          # tally on its ancestors so later rejects cannot demote a fruitful region.
          ancestor = ctx.index.parent.get(current.node_id)
          while ancestor:
            state.reject_folders.pop(ancestor, None)
            ancestor = ctx.index.parent.get(ancestor)
        else:
          trail.append(_short(current) + "✗")
          # A reject of one section is weak evidence that the whole document lacks the
          # answer, so a confidently-reached document gets the same sweep a take gets
          # before anything is condemned.
          reject_file = ctx.index.enclosing_file(current)
          evidence_before = len(state.evidence)
          if (
            reject_file is not None
            and reject_file.is_file()
            and descent_top >= config.reject_sweep_min
            and len(state.evidence) < config.max_evidence
          ):
            has_more = any(
              c.node_id not in state.visited and c.node_id not in state.seen
              for c in reject_file.children
            ) or any(
              ctx.index.is_under(f.node.node_id, reject_file.node_id)
              for f in state.frontier + state.reserve
            )
            if has_more:
              sweep = intra_file_sweep(
                ctx,
                question,
                reject_file,
                state,
                breadth=synthesis.matched,
                anchor=descent_top,
              )
              if sweep:
                step_record.reject_sweep = sweep
                step_record.swept_file = reject_file.name
          # Barren reject: demote this region. First the rejected node's own document
          # subtree; then bubble the reject up the ancestor folders so a homogeneous
          # cluster of look-alike decoys is abandoned as a whole.
          if len(state.evidence) == evidence_before:
            target = reject_file or current
            state.penalize_subtree(target.node_id)
            ancestor = ctx.index.parent.get(target.node_id)
            while ancestor:
              ancestor_node = ctx.index.nodes.get(ancestor)
              if ancestor_node is None or ancestor_node.node_type == "root":
                break
              if ancestor_node.node_type == "folder":
                state.reject_folders[ancestor] += 1
                if state.reject_folders[ancestor] == config.cluster_reject_n:
                  state.penalize_subtree(ancestor)
              ancestor = ctx.index.parent.get(ancestor)

        # ---- consider stopping ----------------------------------------------------
        if state.evidence:
          if synthesis.matched:
            # Breadth, gate-free: the sufficiency gate under-counts an enumeration, so
            # trusting it stops short of the full list.
            n_files = distinct_files(state.evidence)
            evidence_sources_set = {
              e.source_file() or e.path or e.name for e in state.evidence
            }
            best_new = next(
              (
                f
                for f in sorted(state.frontier, key=lambda f: f.score, reverse=True)
                if source_of(f.node) not in evidence_sources_set
              ),
              None,
            )
            base = min(state.committed_scores.values()) if state.committed_scores else 0.0
            strong_left = (
              best_new is not None
              and best_new.score >= base - config.synth_breadth_window
              and best_new.score >= config.noise_floor
            )
            ctx.emit(
              SynthesisBreadthEvent(
                step=steps,
                n_files=n_files,
                n_evidence=len(state.evidence),
                strong_left=bool(strong_left),
                best_new=round(best_new.score, 3) if best_new else None,
              )
            )
            if n_files >= config.max_files:
              break
            if n_files >= config.synth_max_files and not strong_left:
              break
            if not state.frontier and not state.reserve:
              break
          else:
            unread = [
              f"{node.name}  (relevance {score:.2f})"
              for node, score in sorted(
                state.deferred, key=lambda item: item[1], reverse=True
              )[:8]
              if score >= 0.40
            ]
            suff = gate(unread)
            # A contrast excursion ends at the first completed read after dispatch, and
            # this gate call always follows one. Whether it CONFIRMED the committed
            # account is decided below by whether it changed the evidence.
            was_contrast = contrast_active
            contrast_active = False
            ctx.emit(
              SufficiencyCheckEvent(
                step=steps,
                sufficient=suff.sufficient,
                reasoning=suff.reasoning,
                n_evidence=len(state.evidence),
                basis=suff.basis,
                unread_same_file=len(unread),
              )
            )
            residual = () if suff.sufficient else suff.missing
            set_residual(state.memory, residual, config.parts_max)
            # A contrast directive is stale once the gate reports the evidence
            # incomplete: the job is back to filling the named gap, not to seeking a
            # rival account.
            if not suff.sufficient:
              set_contrast(state.memory, None)

            # Recover deferred sections before leaving the document.
            drained = 0
            while (
              not suff.sufficient
              and state.deferred
              and drained < config.deferred_max_reads
              and len(state.evidence) < config.max_evidence
            ):
              state.deferred.sort(key=lambda item: item[1], reverse=True)
              node, score = state.deferred.pop(0)
              drained += 1
              decision, adds, why = read_selection(
                ctx, question, node, state.evidence, fallback=True
              )
              if decision == "take" and state.collect(node, whole=not node.is_leaf()):
                ctx.emit(
                  DeferredTakeEvent(
                    step=steps, node=node.name, score=round(score, 3), adds=adds
                  )
                )
              else:
                ctx.emit(
                  DeferredSkipEvent(
                    step=steps, node=node.name, score=round(score, 3), reasoning=why
                  )
                )
                continue
              unread = [
                f"{n.name}  (relevance {s:.2f})"
                for n, s in sorted(state.deferred, key=lambda item: item[1], reverse=True)
              ]
              suff = gate(unread)
              ctx.emit(
                SufficiencyCheckEvent(
                  step=steps,
                  sufficient=suff.sufficient,
                  reasoning=suff.reasoning,
                  n_evidence=len(state.evidence),
                  basis=suff.basis,
                  after_deferred=True,
                )
              )
              residual = () if suff.sufficient else suff.missing
              set_residual(state.memory, residual, config.parts_max)

            # Stall signal: an insufficiency verdict after evidence already spans
            # several distinct documents means this region is not yielding the answer.
            # There is deliberately no reset on a sufficient verdict - a gate that
            # oscillates is the signature of a region that keeps ALMOST answering, and
            # resetting there is what made the breadth escape dead on exactly the runs
            # that need it.
            if not suff.sufficient and (
              distinct_files(state.evidence) >= config.stall_min_docs
              or (definitional and suff.basis == "incidental")
            ):
              stall += 1

            stop = False
            if suff.sufficient:
              # A failed contrast is confirmation: the excursion was dispatched by the
              # agent's own choice of the alternative most likely to contradict the
              # account in hand; it went there, read it, and nothing changed. Re-running
              # the check until the explore budget is exhausted treats that confirmation
              # as doubt.
              confirmed = was_contrast and len(state.evidence) == contrast_evidence_mark
              near: FrontierItem | None = None
              alt_why = ""
              shortlist: list[FrontierItem] = []
              if (
                not confirmed and tie_explores < config.tie_max_explore and state.frontier
              ):
                state.frontier.sort(key=lambda f: f.score, reverse=True)
                evidence_srcs = {
                  e.source_file() or e.path or e.name for e in state.evidence
                }
                # Anchor on the WEAKEST document actually accepted as evidence, on the
                # frontier's own folder/file scale - not the section-level descent top,
                # which runs higher and made a genuine peer sibling look too low.
                base = (
                  min(state.committed_scores.values())
                  if state.committed_scores
                  else descent_top
                )
                # Single ACCOUNT, not merely single source: two documents in the same
                # top-level region restating the same fact are one route's findings, not
                # independent verification.
                single_account = (
                  len(evidence_srcs) == 1 or len(state.evidence_regions) <= 1
                )
                window = config.single_src_window if single_account else config.tie_window
                eligible = [
                  f
                  for f in state.frontier
                  if source_of(f.node) not in evidence_srcs
                  and f.score >= base - window
                  and f.score >= config.noise_floor
                ]
                if single_account:
                  # Banished-region admission: on a single-source stop, also offer the
                  # best entry from each top-level region not yet entered, regardless of
                  # score. A floored region's low score is the root ranker's ignorance
                  # of body content, not a finding, and the score window cannot reach it
                  # otherwise.
                  have_ids = {id(f) for f in eligible}
                  best_by_region: dict[str, FrontierItem] = {}
                  for entry in state.frontier + state.reserve:
                    if source_of(entry.node) in evidence_srcs:
                      continue
                    region = ctx.index.region_of(entry.node.node_id)
                    if region in state.entered_regions:
                      continue
                    current_best = best_by_region.get(region)
                    if current_best is None or entry.score > current_best.score:
                      best_by_region[region] = entry
                  for entry in best_by_region.values():
                    if id(entry) not in have_ids:
                      eligible.append(entry)
                      have_ids.add(id(entry))
                shortlist = diverse_shortlist(ctx, eligible, config.alt_shortlist)
                if shortlist:
                  near, alt_why = choose_alternative(
                    ctx, question, state.evidence, shortlist
                  )
              if near is None:
                set_contrast(state.memory, None)
                if confirmed:
                  ctx.emit(
                    ContrastConfirmedEvent(step=steps, n_evidence=len(state.evidence))
                  )
                stop = True
              else:
                tie_explores += 1
                forced_next = near
                contrast_active = True
                contrast_evidence_mark = len(state.evidence)
                set_contrast(state.memory, state.evidence)
                ctx.emit(
                  ContrastExploreEvent(
                    step=steps,
                    node=near.node.name,
                    score=round(near.score, 3),
                    committed=round(descent_top, 3),
                    n_eligible=len(shortlist),
                    reasoning=alt_why,
                  )
                )
            elif not state.frontier and not state.reserve:
              stop = True
            if stop:
              break

        # ---- teleport ---------------------------------------------------------------
        state.prune_frontiers()
        nxt: FrontierItem | None = None
        if forced_next is not None:
          if state.drop_frontier_entry(forced_next) and state.entry_available(forced_next):
            nxt = forced_next
          forced_next = None
        if nxt is None and not synthesis.matched and stall >= config.stall_trigger:
          unentered = [
            (entry, home)
            for home in (state.frontier, state.reserve)
            for entry in home
            if ctx.index.region_of(entry.node.node_id) not in state.entered_regions
          ]
          if unentered:
            escape, home = max(unentered, key=lambda item: item[0].score)
            for position, entry in enumerate(home):
              if entry is escape:
                del home[position]
                break
            stall = 0
            nxt = escape
            ctx.emit(
              BreadthEscapeEvent(
                step=steps, node=escape.node.name, score=round(escape.score, 3)
              )
            )
        if nxt is None and residual and not synthesis.matched:
          pool = diverse_shortlist(ctx, state.frontier, config.resid_shortlist)
          if len(pool) >= 2:
            pick, why = choose_by_residual(ctx, question, residual, state.evidence, pool)
            state.drop_frontier_entry(pick)
            nxt = pick
            ctx.emit(
              ResidualTeleportEvent(
                step=steps,
                node=pick.node.name,
                score=round(pick.score, 3),
                missing=residual[: config.parts_max],
                n_shortlist=len(pool),
                top_score=round(pool[0].score, 3),
                reasoning=why,
              )
            )
        if nxt is None:
          nxt, _from_reserve = state.pop_frontier()

        snapshot = sorted(
          state.frontier + ([nxt] if nxt else []),
          key=lambda f: f.score,
          reverse=True,
        )
        ctx.emit(
          TeleportEvent(
            step=steps,
            origin=current.name,
            frontier=tuple(
              FrontierEntry(node=f.node.name, score=round(f.score, 3), origin=f.origin)
              for f in snapshot[:12]
            ),
            target=nxt.node.name if nxt else "",
            target_score=round(nxt.score, 3) if nxt else None,
          )
        )
        if nxt is None:
          break
        teleports += 1
        current = nxt.node
        state.visited.add(current.node_id)
        trail.append("⇥" + _short(current))
        descent_top = nxt.score
        state.line_entry_score = nxt.score
        # Deferred candidates belong to the document just left; do not carry them over.
        state.deferred.clear()
        nosignal_run = 0
        continue

      # ---- interior node: rank children, descend the best -----------------------------
      ranked = rank_children(ctx, question, kids, state.memory)
      best, best_score, best_reason = ranked[0]

      # Contrast collapse abort: mid-excursion the ranker has judged every child of the
      # contrast target irrelevant. The excursion exists to check a rival ACCOUNT, and a
      # subtree whose contents all rank as noise holds none, so grinding down to a leaf
      # only re-derives this verdict at higher cost.
      if contrast_active and state.evidence and best_score < config.noise_floor:
        state.penalize_subtree(current.node_id)
        set_contrast(state.memory, None)
        contrast_active = False
        ctx.emit(
          ContrastAbortEvent(
            step=steps,
            at=current.name,
            best_score=round(best_score, 3),
            n_evidence=len(state.evidence),
          )
        )
        break

      # No-signal descent abort: at folder routing levels a best-child score below the
      # noise floor is the ranker declaring IGNORANCE, so the descent taken on it is a
      # guess. One guess is allowed - a level deeper it sees richer names and may
      # recover real signal - but a run of them falsifies the guess. Folders only:
      # inside a document, low section scores are often summary lossiness, and reads are
      # the mechanism there.
      if current.node_type in ("root", "folder"):
        nosignal_run = nosignal_run + 1 if best_score < config.noise_floor else 0
        if nosignal_run >= config.nosignal_abort and state.frontier:
          state.push_frontier(ranked, current.name)
          jump, _from_reserve = state.pop_frontier()
          if jump is not None:
            nosignal_run = 0
            teleports += 1
            ctx.emit(
              NoSignalAbortEvent(
                step=steps,
                at=current.name,
                best_score=round(best_score, 3),
                target=jump.node.name,
                target_score=round(jump.score, 3),
              )
            )
            current = jump.node
            state.visited.add(current.node_id)
            trail.append("⇥" + _short(current))
            descent_top = jump.score
            state.line_entry_score = jump.score
            state.deferred.clear()
            continue

      descent_top = max(descent_top, best_score)
      if best.node_type in ("folder", "file"):
        # Document-commitment score, on the same scale as frontier entries; section
        # scores are excluded on purpose.
        state.line_entry_score = best_score
      state.push_frontier(ranked[1:], current.name)
      ctx.emit(
        DescendEvent(
          step=steps,
          at=current.name,
          node_type=current.node_type,
          chose=best.name,
          chose_score=round(best_score, 3),
          reasoning=best_reason,
          ranked=tuple(
            RankedCandidate(node=node.name, score=round(score, 3))
            for node, score, _ in ranked[:10]
          ),
        )
      )
      state.visited.add(best.node_id)
      trail.append(_short(best))
      current = best
  except SearchBudgetError as exc:
    budget_stop = str(exc)

  if budget_stop:
    ctx.emit(
      BudgetStopEvent(
        step=steps,
        reason=budget_stop,
        elapsed_seconds=ctx.budget.elapsed() if ctx.budget else 0.0,
        llm_calls=ctx.counters.calls,
        n_evidence=len(state.evidence),
      )
    )

  # Last resort: if nothing at all was gathered, keep wherever the search ended.
  if not state.evidence and current.node_id not in state.seen:
    state.collect(current, whole=not current.is_leaf())

  # The traversal is over; release the allowance reserved for writing the answer, so a
  # search that spent every second of its search budget still produces one.
  if ctx.budget is not None:
    ctx.budget.begin_answer()
    ctx.budget.stopped_reason = budget_stop

  # Retrieval hand-off on a genuinely multi-part question. Skipped when the budget is
  # already spent: it is an extra LLM call that reorders evidence rather than finding
  # any.
  final_evidence = state.evidence
  if (
    not budget_stop
    and not synthesis.matched
    and len(parts) >= 2
    and len(state.evidence) >= config.part_select_min
  ):
    final_evidence, per_part = select_by_part_coverage(
      ctx, question, parts, state.evidence
    )
    if per_part is not None:
      ctx.emit(
        PartCoverageEvent(
          parts=tuple(parts),
          kept=len(final_evidence),
          of=len(state.evidence),
          per_part=per_part,
        )
      )

  return TraversalResult(
    evidence=final_evidence[: config.max_evidence],
    memory=state.memory,
    steps=steps,
    teleports=teleports,
    path=" › ".join(trail),
    trace=ctx.trace,
    synthesis=synthesis.matched,
    parts=parts,
    budget_stop=budget_stop,
  )
