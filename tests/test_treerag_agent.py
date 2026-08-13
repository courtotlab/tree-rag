"""End-to-end tests of the TreeRAG traversal against a scripted LLM.

These drive the real ``run_agent`` and the real answer assembly; only the model is faked.
The fake routes on the distinctive first sentence of each prompt, so if a prompt is
reworded in a way that changes which decision it is, these tests notice.
"""

import re

import pytest
from treerag_fixtures import FakeClient, chunk, make_context, sample_tree

from treerag import events as ev
from treerag.agent import run_agent
from treerag.budget import SearchBudget
from treerag.config import TreeRAGConfig, TreeRAGMode
from treerag.context import SearchContext
from treerag.errors import OllamaUnavailableError
from treerag.search import (
  SearchTick,
  TreeRAGResult,
  treerag_search,
  treerag_search_stream,
)

_CANDIDATE_RE = re.compile(r"^\[(\d+)\] ([^\n—]+?)(?: —|$)", re.M)

# What the scripted ranker thinks of each node, by name. Anything unlisted scores low.
_SCORES = {
  "Policies": 0.95,
  "Training": 0.10,
  "Instrument Calibration Policy": 0.92,
  "Sample Storage Policy": 0.30,
  "Pipette Training Record": 0.05,
  "Calibration Frequency": 0.90,
  "Responsibilities": 0.35,
  "Storage Temperature": 0.25,
  "Trainee Sign-off": 0.05,
  "Calibration Frequency 1": 0.90,
  "Responsibilities 1": 0.30,
}


def _score_reply(prompt: str) -> str:
  """Score every candidate the prompt lists, by name.

  Args:
    prompt: The ranking prompt.

  Returns:
    A JSON scores object covering every listed index.
  """
  block = prompt.split("CANDIDATES:\n", 1)[-1]
  scores: dict[str, float] = {}
  for index, name in _CANDIDATE_RE.findall(block):
    scores[index] = _SCORES.get(name.strip(), 0.08)
  body = ", ".join(f'"{k}":{v}' for k, v in scores.items())
  return '{"reasoning":"scored by name","scores":{' + body + "}}"


def script(prompt: str) -> str:
  """Answer any TreeRAG prompt deterministically.

  Args:
    prompt: The prompt the agent produced.

  Returns:
    The scripted reply.

  Raises:
    AssertionError: If the prompt matches no known decision, which means a prompt was
      reworded and this fake no longer covers it.
  """
  if "Break the question into the DISTINCT sub-questions" in prompt:
    return '{"parts":["how often balances are calibrated"], "subject":""}'
  if "Rate how likely EACH entry" in prompt:
    return _score_reply(prompt)
  if "Which ONE of these sections is MOST likely" in prompt:
    return '{"choice":0}'
  if "You have navigated to a specific document and can now read it" in prompt:
    take = "calibrated every six months" in prompt or "Calibration" in prompt
    decision = "take" if take else "reject"
    return '{"reasoning":"judged on content","remember":"","decision":"%s"}' % decision
  if "You have found a passage relevant to the question" in prompt:
    return '{"reasoning":"the passage stands alone","scope":"self"}'
  if "You just took evidence from a document you deliberately navigated to" in prompt:
    return '{"reasoning":"nothing else belongs","read":[]}'
  if "You selected this section of the document as worth reading" in prompt:
    return '{"reasoning":"already covered","decision":"skip","adds":""}'
  if "Decide whether the evidence below is SUFFICIENT" in prompt:
    enough = "six months" in prompt
    return '{"reasoning":"states the interval","missing":[],"sufficient":%s}' % (
      "true" if enough else "false"
    )
  if "you get ONE look at an unexplored document" in prompt:
    return '{"reasoning":"nothing would change the answer","choice":-1}'
  if "have just judged the evidence you hold INCOMPLETE" in prompt:
    return '{"reasoning":"try the other policy","choice":0}'
  if "The search is finished. Below are the PARTS" in prompt:
    return '{"coverage":{"1":[0]}}'
  if "Rewrite the answer below so that the information from each document" in prompt:
    return (
      "Balances are calibrated every six months by the metrology vendor "
      "[Policies/Instrument Calibration Policy.docx]."
    )
  if "DOCUMENTS:" in prompt and prompt.rstrip().endswith("ANSWER:"):
    return (
      "Balances are calibrated every six months by the metrology vendor "
      "[Policies/Instrument Calibration Policy.docx]."
    )
  if "DOCUMENTS:" in prompt:
    return (
      "Balances are calibrated every six months by the metrology vendor "
      "[Policies/Instrument Calibration Policy.docx]."
    )
  raise AssertionError(f"unscripted prompt: {prompt[:160]!r}")


QUESTION = "How often are balances calibrated?"


def test_traversal_reaches_the_right_document_and_collects_evidence() -> None:
  ctx = make_context(script)
  result = run_agent(ctx, QUESTION)

  assert result.evidence, "the traversal must collect evidence"
  sources = {e.source_file() or e.path or e.name for e in result.evidence}
  assert sources == {"Policies/Instrument Calibration Policy.docx"}
  assert result.steps > 0
  assert "root" in result.path


def test_traversal_emits_the_documented_event_types() -> None:
  ctx = make_context(script)
  run_agent(ctx, QUESTION)
  kinds = {e.event for e in ctx.trace}

  assert "mode" in kinds
  assert "descend" in kinds
  assert "read_file" in kinds
  assert "sufficiency_check" in kinds

  descends = [e for e in ctx.trace if isinstance(e, ev.DescendEvent)]
  assert descends
  first = descends[0]
  assert first.at == "Corpus"
  assert first.chose == "Policies"
  assert first.chose_score == pytest.approx(0.95, abs=0.13)
  assert first.ranked, "a descend event must carry its ranked candidates"

  reads = [e for e in ctx.trace if isinstance(e, ev.ReadFileEvent)]
  assert reads and reads[0].decision in ("take", "reject")


def test_events_reach_the_sink_in_order_as_they_happen() -> None:
  seen: list[str] = []
  ctx = make_context(script)
  ctx.sink = lambda event: seen.append(event.event)
  run_agent(ctx, QUESTION)
  assert seen == [e.event for e in ctx.trace]
  assert seen.index("mode") < seen.index("descend")


def test_runner_ups_are_pushed_to_the_frontier_and_used() -> None:
  # 'Training' scores below 'Policies' at the root, so it must be remembered rather than
  # discarded: the frontier is what makes a wrong first choice recoverable.
  ctx = make_context(script)
  run_agent(ctx, QUESTION)
  descends = [e for e in ctx.trace if isinstance(e, ev.DescendEvent)]
  root_ranked = {c.node for c in descends[0].ranked}
  assert {"Policies", "Training"} <= root_ranked


def test_full_search_writes_a_cited_answer() -> None:
  ctx = make_context(script)
  from treerag.search import _run

  result = _run(ctx, QUESTION)
  assert isinstance(result, TreeRAGResult)
  assert "six months" in result.answer
  assert result.sources == ("Policies/Instrument Calibration Policy.docx",)
  assert "[1]" in result.answer
  assert "#### References" in result.answer
  assert result.llm_calls > 0
  assert result.in_tokens > 0
  assert result.out_tokens > 0
  assert result.elapsed_seconds >= 0.0
  assert result.trace
  assert result.evidence
  assert result.evidence[0].source_path.endswith("Instrument Calibration Policy.docx")


def test_treerag_search_rejects_an_empty_question() -> None:
  with pytest.raises(ValueError, match="must not be empty"):
    treerag_search("   ", TreeRAGConfig(), tree=sample_tree(), client=FakeClient(script))


def test_stream_yields_events_then_the_result() -> None:
  items = list(
    treerag_search_stream(
      QUESTION,
      TreeRAGConfig(),
      tree=sample_tree(),
      client=FakeClient(script),
      tick_seconds=0.05,
    )
  )
  assert items, "the stream must yield something"
  assert isinstance(items[-1], TreeRAGResult)
  events = [i for i in items if not isinstance(i, (SearchTick, TreeRAGResult))]
  assert events
  assert all(hasattr(e, "event") for e in events)
  # The streamed events are exactly the result's trace, in order.
  assert [e.event for e in events] == [e.event for e in items[-1].trace]


def test_stream_surfaces_a_dead_endpoint_rather_than_hanging() -> None:
  def dead(prompt: str) -> str:
    raise OllamaUnavailableError("Ollama did not respond after 4 attempt(s)")

  with pytest.raises(OllamaUnavailableError, match="did not respond"):
    list(
      treerag_search_stream(
        QUESTION,
        TreeRAGConfig(),
        tree=sample_tree(),
        client=FakeClient(dead),
        tick_seconds=0.05,
      )
    )


def test_non_streaming_and_streaming_agree() -> None:
  blocking = treerag_search(
    QUESTION, TreeRAGConfig(), tree=sample_tree(), client=FakeClient(script)
  )
  streamed = [
    i
    for i in treerag_search_stream(
      QUESTION,
      TreeRAGConfig(),
      tree=sample_tree(),
      client=FakeClient(script),
      tick_seconds=0.05,
    )
    if isinstance(i, TreeRAGResult)
  ][0]
  assert blocking.answer == streamed.answer
  assert blocking.sources == streamed.sources
  assert blocking.steps == streamed.steps
  assert [e.event for e in blocking.trace] == [e.event for e in streamed.trace]


def test_step_and_evidence_caps_are_honoured() -> None:
  # Force the gate to never be satisfied; the run must still terminate on its caps.
  def never_enough(prompt: str) -> str:
    if "Decide whether the evidence below is SUFFICIENT" in prompt:
      return '{"reasoning":"not yet","missing":["more"],"sufficient":false}'
    if "You have navigated to a specific document and can now read it" in prompt:
      return '{"reasoning":"keep","remember":"","decision":"take"}'
    return script(prompt)

  config = TreeRAGConfig(max_steps=6, max_files=8, max_evidence=50)
  ctx = SearchContext(config=config, client=FakeClient(never_enough), index=sample_tree())
  result = run_agent(ctx, QUESTION)
  assert result.steps <= 6


def test_a_rejecting_run_still_returns_something() -> None:
  def reject_everything(prompt: str) -> str:
    if "You have navigated to a specific document and can now read it" in prompt:
      return '{"reasoning":"off topic","remember":"","decision":"reject"}'
    if "Decide whether the evidence below is SUFFICIENT" in prompt:
      return '{"reasoning":"nothing","missing":["everything"],"sufficient":false}'
    return script(prompt)

  ctx = SearchContext(
    config=TreeRAGConfig(max_steps=8),
    client=FakeClient(reject_everything),
    index=sample_tree(),
  )
  result = run_agent(ctx, QUESTION)
  # The last-resort collection guarantees the answer step always has something to say.
  assert result.evidence


def test_synthesis_mode_is_detected_and_recorded() -> None:
  ctx = make_context(script)
  run_agent(ctx, "List all of our calibration policies")
  modes = [e for e in ctx.trace if isinstance(e, ev.ModeEvent)]
  synthesis = [m for m in modes if m.mode == "synthesis"]
  assert synthesis and synthesis[0].active


# ---------------------------------------------------------------------------
# the work budget: a search must answer within its stated time
# ---------------------------------------------------------------------------


def test_traversal_stops_on_its_time_budget_and_still_answers() -> None:
  # A budget already spent: the traversal must stop at its first check, say so, and hand
  # whatever it holds to the answer step rather than running on or raising.
  ctx = SearchContext(
    config=TreeRAGConfig(),
    client=FakeClient(script),
    index=sample_tree(),
    budget=SearchBudget(traversal_seconds=0.0, answer_seconds=60.0, max_calls=500),
  )
  result = run_agent(ctx, QUESTION)

  assert result.budget_stop, "the traversal must record why it stopped"
  assert "time budget" in result.budget_stop
  stops = [e for e in ctx.trace if isinstance(e, ev.BudgetStopEvent)]
  assert len(stops) == 1
  assert stops[0].reason == result.budget_stop
  # The last-resort collection guarantees the answer step has something to work with.
  assert result.evidence


def test_traversal_stops_on_its_call_budget() -> None:
  ctx = SearchContext(
    config=TreeRAGConfig(),
    client=FakeClient(script),
    index=sample_tree(),
    budget=SearchBudget(traversal_seconds=600.0, answer_seconds=60.0, max_calls=3),
  )
  result = run_agent(ctx, QUESTION)
  assert "LLM calls" in result.budget_stop
  assert ctx.counters.calls <= 6, "the call ceiling must actually bite"


def test_a_budget_stop_still_produces_a_cited_answer() -> None:
  from treerag.search import _run

  ctx = SearchContext(
    config=TreeRAGConfig(),
    client=FakeClient(script),
    index=sample_tree(),
    budget=SearchBudget(traversal_seconds=0.0, answer_seconds=120.0, max_calls=500),
  )
  result = _run(ctx, QUESTION)
  assert result.stopped_early
  assert result.answer.strip(), "running out of search time must not cost the answer"


def test_budget_stop_event_reads_as_progress() -> None:
  line = ev.render_event(
    ev.BudgetStopEvent(
      step=7,
      reason="time budget of 150s reached",
      elapsed_seconds=151.2,
      llm_calls=61,
      n_evidence=2,
    )
  )
  assert line is not None
  assert "Stopped searching" in line
  assert "time budget of 150s reached" in line


def test_quick_mode_makes_fewer_calls_than_thorough() -> None:
  def counted(mode: TreeRAGMode) -> int:
    ctx = SearchContext(
      config=TreeRAGConfig.for_mode(mode),
      client=FakeClient(script),
      index=sample_tree(),
    )
    run_agent(ctx, "What are the storage requirements and who approves them?")
    return ctx.counters.calls

  assert counted(TreeRAGMode.QUICK) <= counted(TreeRAGMode.THOROUGH)


def test_a_wide_node_costs_a_bounded_number_of_calls() -> None:
  # The regression this guards: ranking scored every child, so one wide node cost
  # ceil(n / score_batch) LLM calls - 533 for the widest node in the real corpus.
  from treerag.ranking import rank_children

  ctx = SearchContext(
    config=TreeRAGConfig(max_rank_candidates=20, score_batch=5),
    client=FakeClient(script),
    index=sample_tree(),
  )
  kids = [
    chunk(f"w{i}", f"Section {i} on storage", "text", "A/Doc.docx") for i in range(400)
  ]
  ranked = rank_children(ctx, "storage", kids, [])

  assert ctx.counters.calls <= 4, "20 candidates at 5 per call is 4 calls, not 80"
  assert len(ranked) == 400, "every candidate must still be accounted for"
  assert sum(1 for _, score, _ in ranked if score == 0.0) == 380, "overflow is deferred"
