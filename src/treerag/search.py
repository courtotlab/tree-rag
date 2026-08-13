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

The public entry point.

Two shapes, one implementation: :func:`treerag_search` blocks and returns the finished
result, for programmatic callers; :func:`treerag_search_stream` yields each navigation
event as it happens and then the result, for the UI. The streaming form runs the same
traversal on a worker thread and drains its events through a queue - the agent's decision
logic is identical either way, and only the event sink differs.
"""

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from treerag.agent import run_agent
from treerag.answer import AnswerDiagnostics, DraftedAnswer, write_answer
from treerag.budget import SearchBudget
from treerag.citations import render_citations
from treerag.client import HealthStatus, OllamaClient, get_client
from treerag.config import TreeRAGConfig, TreeRAGMode
from treerag.context import SearchContext
from treerag.errors import SearchBudgetError
from treerag.events import TraceEvent
from treerag.tree import TreeIndex, get_tree
from treerag.types import TreeNode


@dataclass(frozen=True, slots=True)
class SearchTick:
  """An idle heartbeat emitted between navigation events.

  A single LLM call can run for a minute or more, and a progress panel that stops updating
  for that long reads as a hung request. Ticks let a caller refresh an elapsed timer
  without inventing progress it does not have. They are not trace events and never appear
  in :attr:`TreeRAGResult.trace`.

  Attributes:
    elapsed_seconds: Seconds since the search started.
    llm_calls: Successful model calls completed so far.
  """

  elapsed_seconds: float
  llm_calls: int = 0


@dataclass(frozen=True, slots=True)
class EvidenceItem:
  """One piece of evidence the traversal kept.

  Attributes:
    source_path: Path of the source document the piece came from.
    name: Name of the document or section it was taken from.
    node_type: The node type it was taken at.
    text: The piece's text.
  """

  source_path: str
  name: str
  node_type: str
  text: str


@dataclass(frozen=True, slots=True)
class TreeRAGResult:
  """Everything one TreeRAG search produced.

  Attributes:
    question: The question that was asked.
    answer: The final answer, with citations rendered as the UI shows them.
    raw_answer: The answer before citation rendering, citing full source paths.
    sources: Source document paths cited by the answer, in the order shown.
    evidence: The evidence pieces the answer was written from.
    steps: Node visits consumed by the traversal.
    teleports: Frontier jumps taken.
    path: The traversal trail.
    elapsed_seconds: Wall-clock time the whole search took.
    in_tokens: Prompt tokens consumed.
    out_tokens: Completion tokens produced.
    llm_calls: Number of LLM calls made.
    trace: Every navigation event, in order.
    synthesis: Whether enumeration (breadth) mode was used.
    diagnostics: How the answer was produced.
    mode: Which named preset the search ran under.
    stopped_early: Why navigation stopped on its budget, empty if it finished naturally.
  """

  question: str
  answer: str
  raw_answer: str
  sources: tuple[str, ...]
  evidence: tuple[EvidenceItem, ...]
  steps: int
  teleports: int
  path: str
  elapsed_seconds: float
  in_tokens: int
  out_tokens: int
  llm_calls: int
  trace: tuple[TraceEvent, ...]
  synthesis: bool
  diagnostics: AnswerDiagnostics = field(default_factory=AnswerDiagnostics)
  mode: TreeRAGMode = TreeRAGMode.THOROUGH
  stopped_early: str = ""


def _to_evidence(nodes: list[TreeNode]) -> tuple[EvidenceItem, ...]:
  """Convert collected tree nodes into the result's evidence items.

  Args:
    nodes: The evidence nodes the traversal kept.

  Returns:
    One typed evidence item per node.
  """
  return tuple(
    EvidenceItem(
      source_path=node.source_file() or node.path or node.name,
      name=node.name,
      node_type=node.node_type,
      text=node.content or node.summary,
    )
    for node in nodes
  )


def build_context(
  config: TreeRAGConfig,
  *,
  tree: TreeIndex | None = None,
  client: OllamaClient | None = None,
) -> SearchContext:
  """Assemble a search context from the shared tree and client.

  Args:
    config: The configuration for this search.
    tree: An already-loaded tree; the process-wide singleton is used when omitted.
    client: An already-built client; the process-wide one is used when omitted.

  Returns:
    A fresh context, with its own counters and trace.

  Raises:
    TreeUnavailableError: If the tree is missing or unreadable.
    MalformedTreeError: If the tree does not match the node schema.
  """
  return SearchContext(
    config=config,
    client=client or get_client(config),
    index=tree if tree is not None else get_tree(config),
    budget=SearchBudget(
      traversal_seconds=config.time_budget_s,
      answer_seconds=config.answer_budget_s,
      max_calls=config.max_llm_calls,
    ),
  )


def health_check(config: TreeRAGConfig, *, force: bool = False) -> HealthStatus:
  """Check that the Ollama endpoint is reachable and the required model is loaded.

  Args:
    config: The configuration naming the endpoint and model.
    force: Re-probe even when a fresh cached verdict exists.

  Returns:
    The health status. Never raises: an unreachable endpoint is a status, not an error.
  """
  return get_client(config).health_check(force=force)


def _run(ctx: SearchContext, question: str) -> TreeRAGResult:
  """Run one traversal and write its answer.

  Args:
    ctx: The search context.
    question: The question to answer.

  Returns:
    The finished result.

  Raises:
    OllamaUnavailableError: If the bounded retry policy is exhausted.
  """
  started = time.perf_counter()
  traversal = run_agent(ctx, question)
  try:
    drafted = write_answer(ctx, question, traversal.evidence)
    rendered = render_citations(drafted.text, drafted.doc_ids)
  except SearchBudgetError as exc:
    # Even the reserved answer allowance is gone. Report what was found rather than
    # failing the request outright: the sources are still useful to the user.
    drafted = DraftedAnswer(text="", doc_ids=[], diagnostics=AnswerDiagnostics())
    sources = ", ".join(
      Path(e.source_file() or e.path or e.name).stem for e in traversal.evidence
    )
    rendered = (
      f"TreeRAG ran out of time before it could write an answer ({exc}). "
      + (f"It had gathered evidence from: {sources}." if sources else "")
      + " Try the quick mode, or vector search, for a faster response."
    )
  elapsed = time.perf_counter() - started
  return TreeRAGResult(
    question=question,
    answer=rendered,
    raw_answer=drafted.text,
    sources=tuple(drafted.doc_ids),
    evidence=_to_evidence(traversal.evidence),
    steps=traversal.steps,
    teleports=traversal.teleports,
    path=traversal.path,
    elapsed_seconds=elapsed,
    in_tokens=ctx.counters.in_tok,
    out_tokens=ctx.counters.out_tok,
    llm_calls=ctx.counters.calls,
    trace=tuple(traversal.trace),
    synthesis=traversal.synthesis,
    diagnostics=drafted.diagnostics,
    mode=ctx.config.mode,
    stopped_early=traversal.budget_stop,
  )


def treerag_search(
  question: str,
  config: TreeRAGConfig,
  *,
  tree: TreeIndex | None = None,
  client: OllamaClient | None = None,
) -> TreeRAGResult:
  """Answer a question by navigating the corpus tree.

  This is the non-streaming path, for programmatic callers such as the CLI and the tests.

  Args:
    question: The question to answer.
    config: The configuration for this search.
    tree: An already-loaded tree; the process-wide singleton is used when omitted.
    client: An already-built client; the process-wide one is used when omitted.

  Returns:
    The finished result.

  Raises:
    ValueError: If the question is empty or only whitespace.
    OllamaUnavailableError: If the Ollama endpoint cannot be reached within the bounded
      retry policy.
    TreeUnavailableError: If the corpus tree is missing or unreadable.
    MalformedTreeError: If the corpus tree does not match the node schema.
  """
  stripped = question.strip()
  if not stripped:
    raise ValueError("question must not be empty")
  ctx = build_context(config, tree=tree, client=client)
  return _run(ctx, stripped)


def treerag_search_stream(
  question: str,
  config: TreeRAGConfig,
  *,
  tree: TreeIndex | None = None,
  client: OllamaClient | None = None,
  tick_seconds: float = 1.0,
) -> Iterator[TraceEvent | SearchTick | TreeRAGResult]:
  """Answer a question, yielding each navigation event as it happens.

  The traversal runs on a worker thread and its events are drained through a queue, so the
  caller can render progress while a search that typically takes five to ten minutes is
  still running. Between events a :class:`SearchTick` is yielded every ``tick_seconds`` so
  a live elapsed timer keeps moving even across one slow LLM call - a search that goes
  quiet for two minutes must not look frozen. Ticks are not part of the result's trace.
  The final item yielded is always the :class:`TreeRAGResult`.

  Args:
    question: The question to answer.
    config: The configuration for this search.
    tree: An already-loaded tree; the process-wide singleton is used when omitted.
    client: An already-built client; the process-wide one is used when omitted.
    tick_seconds: How often to emit an idle heartbeat.

  Yields:
    Each navigation event in order, interleaved with idle ticks, then the finished result.

  Raises:
    ValueError: If the question is empty or only whitespace.
    OllamaUnavailableError: If the Ollama endpoint cannot be reached within the bounded
      retry policy.
    TreeUnavailableError: If the corpus tree is missing or unreadable.
    MalformedTreeError: If the corpus tree does not match the node schema.
  """
  stripped = question.strip()
  if not stripped:
    raise ValueError("question must not be empty")

  ctx = build_context(config, tree=tree, client=client)
  events: queue.Queue[TraceEvent] = queue.Queue()
  outcome: dict[str, TreeRAGResult | BaseException] = {}
  finished = threading.Event()

  def sink(event: TraceEvent) -> None:
    events.put(event)

  ctx.sink = sink

  def worker() -> None:
    try:
      outcome["result"] = _run(ctx, stripped)
    except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
      outcome["error"] = exc
    finally:
      finished.set()

  thread = threading.Thread(target=worker, name="treerag-search", daemon=True)
  thread.start()
  started = time.perf_counter()

  poll = min(0.25, tick_seconds)
  next_tick = started + tick_seconds
  while True:
    try:
      yield events.get(timeout=poll)
    except queue.Empty:
      if finished.is_set():
        break
      now = time.perf_counter()
      if now >= next_tick:
        next_tick = now + tick_seconds
        yield SearchTick(
          elapsed_seconds=now - started,
          llm_calls=ctx.counters.calls,
        )
  # Drain anything queued between the last get and the worker finishing.
  while True:
    try:
      yield events.get_nowait()
    except queue.Empty:
      break

  thread.join()
  error = outcome.get("error")
  if isinstance(error, BaseException):
    raise error
  result = outcome.get("result")
  if isinstance(result, TreeRAGResult):
    yield result
