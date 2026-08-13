"""Unit tests for the extracted TreeRAG agent's pure logic.

Nothing here contacts Ollama or reads the real corpus tree: the LLM is a scripted fake and
every tree is invented. The suite therefore runs in CI without the SSH tunnel.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from treerag_fixtures import FakeClient, chunk, sample_tree

from treerag import events as ev
from treerag.answer import dedupe_citations
from treerag.budget import SearchBudget
from treerag.citations import render_citations, source_names
from treerag.client import Counters, HealthStatus, LlmResponse, OllamaClient
from treerag.config import (
  DEFAULT_OLLAMA_URL,
  TreeRAGConfig,
  TreeRAGMode,
  resolve_ollama_url,
)
from treerag.context import SearchContext
from treerag.errors import (
  MalformedTreeError,
  OllamaUnavailableError,
  SearchBudgetError,
)
from treerag.ranking import (
  parse_choice,
  parse_scores,
  shortlist_by_fanout,
)
from treerag.shapes import (
  is_definitional_question,
  is_polarity_question,
  is_recency_question,
  is_synthesis_question,
)
from treerag.state import AgentState, FrontierItem, add_memory, distinct_files
from treerag.text import clip, full, name_stem_set, parse_json_object, stem
from treerag.tree import all_chunks, load_tree, source_of, whole_unit
from treerag.types import TreeNode, node_from_dict

# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
  config = TreeRAGConfig()
  config.validate()
  assert config.max_steps == 40
  assert config.max_files == 8
  assert config.noise_floor == pytest.approx(0.20)
  assert config.model == "gpt-oss:120b"


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("max_steps", 0),
    ("max_files", -1),
    ("noise_floor", 1.5),
    ("reject_decay", -0.1),
    ("max_attempts", 0),
    ("retry_deadline_s", 0.0),
    ("ollama_url", "  "),
    ("model", ""),
  ],
)
def test_invalid_config_rejected(field: str, value: object) -> None:
  kwargs: dict[str, Any] = {field: value}
  with pytest.raises(ValueError, match=field):
    TreeRAGConfig(**kwargs)


def test_max_evidence_must_cover_max_files() -> None:
  with pytest.raises(ValueError, match="max_evidence"):
    TreeRAGConfig(max_files=10, max_evidence=4)


def test_backoff_cap_must_not_be_below_base() -> None:
  with pytest.raises(ValueError, match="retry_backoff_cap_s"):
    TreeRAGConfig(retry_backoff_base_s=10.0, retry_backoff_cap_s=1.0)


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("TREERAG_OLLAMA_URL", "http://ollama.example.invalid:9999")
  monkeypatch.setenv("TREERAG_MODEL", "some-model:1b")
  monkeypatch.setenv("TREERAG_TREE_PATH", "/tmp/tree.json")
  config = TreeRAGConfig.from_env()
  assert config.ollama_url == "http://ollama.example.invalid:9999"
  assert config.model == "some-model:1b"
  assert config.tree_path == Path("/tmp/tree.json")


def test_endpoint_follows_the_vector_agent_by_default(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # One place configures Ollama for the deployment. TreeRAG must not carry a second copy
  # of the address, or the two retrieval modes can silently point at different servers.
  monkeypatch.delenv("TREERAG_OLLAMA_URL", raising=False)
  monkeypatch.setenv("OLLAMA_HOST", "host.docker.internal")
  monkeypatch.setenv("OLLAMA_PORT", "11434")
  assert resolve_ollama_url() == "http://host.docker.internal:11434"
  assert TreeRAGConfig.from_env().ollama_url == "http://host.docker.internal:11434"


def test_endpoint_accepts_a_host_written_with_a_scheme(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # The existing agent interpolates OLLAMA_HOST as "{host}:{port}", and deployments write
  # it both with and without a scheme.
  monkeypatch.delenv("TREERAG_OLLAMA_URL", raising=False)
  monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.204")
  monkeypatch.setenv("OLLAMA_PORT", "11434")
  assert resolve_ollama_url() == "http://10.0.0.204:11434"


def test_treerag_url_overrides_the_shared_endpoint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OLLAMA_HOST", "tunnel")
  monkeypatch.setenv("OLLAMA_PORT", "11437")
  monkeypatch.setenv("TREERAG_OLLAMA_URL", "http://elsewhere.invalid:1234")
  assert resolve_ollama_url() == "http://elsewhere.invalid:1234"


def test_endpoint_falls_back_only_when_nothing_is_configured(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  for name in ("TREERAG_OLLAMA_URL", "OLLAMA_HOST", "OLLAMA_PORT"):
    monkeypatch.delenv(name, raising=False)
  assert resolve_ollama_url() == DEFAULT_OLLAMA_URL


def test_config_describe_has_every_field() -> None:
  described = TreeRAGConfig().describe()
  assert "noise_floor" in described
  assert "ollama_url" in described


# ---------------------------------------------------------------------------
# tree index construction
# ---------------------------------------------------------------------------


def test_index_builds_parent_pointers() -> None:
  index = sample_tree()
  assert index.parent["doc-a"] == "fold-policies"
  assert index.parent["chk-a1"] == "sec-a1"
  assert "root" not in index.parent
  assert index.node_of("sec-a1") is not None
  assert index.parent_of(index.nodes["root"]) is None


def test_is_under_and_region_of() -> None:
  index = sample_tree()
  assert index.is_under("chk-a1", "fold-policies")
  assert not index.is_under("chk-a1", "fold-training")
  assert index.region_of("chk-a1") == "fold-policies"
  assert index.region_of("chk-c1") == "fold-training"
  assert index.region_of("root") == "root"


def test_enclosing_file_stops_at_the_document() -> None:
  index = sample_tree()
  enclosing = index.enclosing_file(index.nodes["chk-a1"])
  assert enclosing.node_id == "doc-a"
  # A document node is its own enclosing document, and a folder is never returned.
  assert index.enclosing_file(index.nodes["doc-a"]).node_id == "doc-a"


def test_single_source_is_false_for_a_multi_document_folder() -> None:
  index = sample_tree()
  assert index.single_source(index.nodes["doc-a"])
  assert not index.single_source(index.nodes["fold-policies"])


def test_all_chunks_is_in_document_order() -> None:
  index = sample_tree()
  names = [c.name for c in all_chunks(index.nodes["doc-a"])]
  assert names == ["Calibration Frequency 1", "Responsibilities 1"]


def test_whole_unit_assembles_text_and_resolves_source() -> None:
  index = sample_tree()
  unit = whole_unit(index.nodes["doc-a"])
  assert "calibrated every six months" in unit.content
  assert "laboratory manager approves" in unit.content
  assert unit.metadata["source_file"] == "Policies/Instrument Calibration Policy.docx"
  assert source_of(index.nodes["chk-a1"]) == unit.metadata["source_file"]


def test_lexical_seed_finds_a_rare_body_term() -> None:
  index = sample_tree()
  seeds = index.lexical_seed_files(
    "what is the metrology vendor arrangement", TreeRAGConfig()
  )
  assert seeds
  assert seeds[0][0].node_id == "doc-a"
  assert 0.0 < seeds[0][1] <= TreeRAGConfig().lex_seed_cap


def test_lexical_seed_is_empty_for_a_generic_question() -> None:
  index = sample_tree()
  assert index.lexical_seed_files("the and of", TreeRAGConfig()) == []


# ---------------------------------------------------------------------------
# strict tree parsing
# ---------------------------------------------------------------------------


def _minimal_node() -> dict[str, object]:
  return {
    "node_id": "root",
    "node_type": "root",
    "name": "Corpus",
    "path": "",
    "summary": "",
    "content": "",
    "children": [],
    "metadata": {},
  }


def test_node_from_dict_accepts_the_schema() -> None:
  node = node_from_dict(_minimal_node())
  assert isinstance(node, TreeNode)
  assert node.node_type == "root"


def test_node_from_dict_is_iterative_and_handles_deep_trees() -> None:
  deep: dict[str, object] = _minimal_node()
  cursor = deep
  for depth in range(3000):
    child: dict[str, object] = {
      "node_id": f"n{depth}",
      "node_type": "section",
      "name": f"S{depth}",
      "path": "",
      "summary": "",
      "content": "",
      "children": [],
      "metadata": {},
    }
    cursor["children"] = [child]
    cursor = child
  root = node_from_dict(deep)
  assert root.count_leaves() >= 0


@pytest.mark.parametrize(
  ("mutate", "match"),
  [
    (lambda d: d.pop("node_id"), "node_id"),
    (lambda d: d.pop("node_type"), "node_type"),
    (lambda d: d.update(node_type="banana"), "unknown node_type"),
    (lambda d: d.update(children={}), "must be a list"),
    (lambda d: d.update(metadata=[]), "must be an object"),
    (lambda d: d.update(name=17), "must be a string"),
    (lambda d: d.update(metadata={"level": "deep"}), "must be an integer"),
    (lambda d: d.update(metadata={"source_file": 3}), "must be a string"),
  ],
)
def test_node_from_dict_rejects_malformed_input(mutate: object, match: str) -> None:
  raw = _minimal_node()
  mutate(raw)  # type: ignore[operator]
  with pytest.raises(MalformedTreeError, match=match):
    node_from_dict(raw)


def test_node_from_dict_rejects_a_non_object_root() -> None:
  with pytest.raises(MalformedTreeError, match="must be a JSON object"):
    node_from_dict([1, 2, 3])


def test_load_tree_reports_malformed_json(tmp_path: Path) -> None:
  path = tmp_path / "tree.json"
  path.write_text("{not json", encoding="utf-8")
  with pytest.raises(MalformedTreeError, match="not valid JSON"):
    load_tree(path)


def test_load_tree_measures_its_cost(tmp_path: Path) -> None:
  path = tmp_path / "tree.json"
  raw = _minimal_node()
  raw["children"] = [
    {
      "node_id": "doc",
      "node_type": "document",
      "name": "A Document",
      "path": "A Document.docx",
      "summary": "",
      "content": "",
      "metadata": {"source_file": "A Document.docx"},
      "children": [
        {
          "node_id": "c1",
          "node_type": "chunk",
          "name": "c1",
          "path": "A Document.docx",
          "summary": "",
          "content": "body",
          "metadata": {"source_file": "A Document.docx", "page": None},
          "children": [],
        }
      ],
    }
  ]
  path.write_text(json.dumps(raw), encoding="utf-8")
  index = load_tree(path)
  assert index.stats.documents == 1
  assert index.stats.chunks == 1
  assert index.stats.nodes == 3
  assert index.stats.total_seconds >= 0.0
  assert "3 nodes" in index.stats.summary()


# ---------------------------------------------------------------------------
# trace event shaping
# ---------------------------------------------------------------------------


def test_every_event_type_is_rendered_or_deliberately_hidden() -> None:
  samples: list[ev.TraceEvent] = [
    ev.SubjectAnchorEvent(subject="the balance in room 2"),
    ev.ModeEvent(mode="synthesis", active=True, why="list/enumerate"),
    ev.ModeEvent(mode="synthesis", active=False, why=""),
    ev.LexicalSeedEvent(files=(ev.RankedCandidate(node="A Policy", score=0.4),)),
    ev.DescendEvent(
      step=1,
      at="Corpus",
      node_type="root",
      chose="Policies",
      chose_score=0.9,
      reasoning="the folder holds policy documents",
      ranked=(ev.RankedCandidate(node="Policies", score=0.9),),
    ),
    ev.ReadFileEvent(
      step=2, at="A Policy", node_type="document", decision="take", reasoning="states it"
    ),
    ev.ReadFileEvent(
      step=3,
      at="B Policy",
      node_type="document",
      decision="reject",
      reasoning="off topic",
    ),
    ev.SufficiencyCheckEvent(
      step=4, sufficient=False, reasoning="no interval", n_evidence=1
    ),
    ev.SufficiencyCheckEvent(step=5, sufficient=True, reasoning="complete", n_evidence=2),
    ev.DeferredTakeEvent(step=6, node="Records", score=0.5, adds="the retention period"),
    ev.DeferredSkipEvent(step=7, node="Scope", score=0.4, reasoning="repeats"),
    ev.ContrastExploreEvent(
      step=8,
      node="Other Policy",
      score=0.8,
      committed=0.85,
      n_eligible=4,
      reasoning="rival",
    ),
    ev.ContrastConfirmedEvent(step=9, n_evidence=3),
    ev.ContrastAbortEvent(step=10, at="Other Policy", best_score=0.1, n_evidence=3),
    ev.NoSignalAbortEvent(
      step=11, at="Admin", best_score=0.05, target="Policies", target_score=0.6
    ),
    ev.BreadthEscapeEvent(step=12, node="Training", score=0.4),
    ev.ResidualTeleportEvent(
      step=13,
      node="Storage Policy",
      score=0.5,
      missing=("the freezer temperature",),
      n_shortlist=6,
      top_score=0.7,
      reasoning="states the set point",
    ),
    ev.TeleportEvent(
      step=14,
      origin="A Policy",
      frontier=(ev.FrontierEntry(node="B Policy", score=0.5, origin="Policies"),),
      target="B Policy",
      target_score=0.5,
    ),
    ev.TeleportEvent(
      step=15, origin="A Policy", frontier=(), target="", target_score=None
    ),
    ev.SynthesisBreadthEvent(
      step=16, n_files=2, n_evidence=2, strong_left=True, best_new=0.6
    ),
    ev.PartCoverageEvent(parts=("a", "b"), kept=2, of=5, per_part={1: (0,), 2: (1,)}),
  ]
  rendered = [ev.render_event(e) for e in samples]
  # Only the two inactive-mode events are hidden.
  assert sum(1 for r in rendered if r is None) == 1
  for event, line in zip(samples, rendered, strict=True):
    if line is None:
      assert isinstance(event, ev.ModeEvent) and not event.active
      continue
    assert line.strip()
    assert "{" not in line, "events must render as prose, never as raw JSON"


def test_descend_event_reads_as_opening_a_folder() -> None:
  line = ev.render_event(
    ev.DescendEvent(
      step=1,
      at="Corpus",
      node_type="root",
      chose="Policies",
      chose_score=0.91,
      reasoning="holds the calibration policy",
      ranked=(),
    )
  )
  assert line is not None
  assert line.startswith("📂 Opening")
  assert "Policies" in line
  assert "0.91" in line


def test_sufficiency_event_reads_as_progress_not_jargon() -> None:
  not_enough = ev.render_event(
    ev.SufficiencyCheckEvent(step=1, sufficient=False, reasoning="x", n_evidence=1)
  )
  enough = ev.render_event(
    ev.SufficiencyCheckEvent(step=2, sufficient=True, reasoning="y", n_evidence=2)
  )
  assert not_enough is not None and "Not enough yet" in not_enough
  assert enough is not None and "Enough to answer" in enough


def test_read_file_event_is_emitted_before_its_sweep_is_populated() -> None:
  # The event is mutable on purpose: it is emitted the moment the decision is made, and
  # the sweep records are attached afterwards on the same object.
  event = ev.ReadFileEvent(
    step=1, at="A Policy", node_type="document", decision="take", reasoning="states it"
  )
  line = ev.render_event(event)
  event.intra_file_sweep.append(ev.SweepDeferred(count=2))
  assert line == ev.render_event(event)
  assert event.intra_file_sweep


# ---------------------------------------------------------------------------
# citation formatting
# ---------------------------------------------------------------------------


def test_dedupe_citations_keeps_the_last_occurrence() -> None:
  text = "Balances are calibrated [A.docx] every six months [A.docx]."
  assert dedupe_citations(text) == "Balances are calibrated every six months [A.docx]."


def test_dedupe_citations_leaves_distinct_documents_alone() -> None:
  text = "First fact [A.docx]. Second fact [B.docx]."
  assert dedupe_citations(text) == text


def test_dedupe_citations_handles_empty_text() -> None:
  assert dedupe_citations("") == ""


def test_render_citations_renumbers_paths(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("treerag.citations._lookup_urls", lambda paths: {})
  answer = "Stored at minus eighty [Policies/Sample Storage Policy.docx]."
  out = render_citations(answer, ["Policies/Sample Storage Policy.docx"])
  assert "[1]" in out
  assert "Sample Storage Policy.docx" not in out.split("#### References")[0]
  assert "#### References" in out
  assert "Sample Storage Policy" in out


def test_render_citations_links_when_urls_resolve(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    "treerag.citations._lookup_urls",
    lambda paths: {p: f"https://example.invalid/{Path(p).name}" for p in paths},
  )
  answer = "Fact one [A/One.docx]. Fact two [A/Two.docx]."
  out = render_citations(answer, ["A/One.docx", "A/Two.docx"])
  assert "[1]" in out and "[2]" in out
  assert "https://example.invalid/One.docx" in out
  assert "https://example.invalid/Two.docx" in out


def test_render_citations_ignores_unknown_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("treerag.citations._lookup_urls", lambda paths: {})
  answer = "A bracketed [aside] that is not a citation."
  assert render_citations(answer, ["A/One.docx"]) == answer


def test_render_citations_survives_an_unreachable_index() -> None:
  # The standalone package has no database dependency and must always render
  # references as plain text when no host URL resolver is configured.
  out = render_citations("Fact [A/One.docx].", ["A/One.docx"])
  assert "[1]" in out
  assert "#### References" in out


def test_source_names_uses_the_document_stem() -> None:
  assert source_names(["Policies/A Policy.docx"]) == ["A Policy"]


# ---------------------------------------------------------------------------
# text helpers and response parsing
# ---------------------------------------------------------------------------


def test_clip_and_full_normalise_whitespace() -> None:
  assert clip("  a   b  ", 10) == "a b"
  assert clip("abcdefghij", 5) == "abcde …"
  assert full("a\r\n\n\n\nb") == "a\n\nb"
  assert clip(None, 5) == ""


def test_stem_is_conservative() -> None:
  assert stem("checklists") == "checklist"
  assert stem("cleaning") == "clean"
  assert stem("qc") == "qc"


def test_name_stem_set_drops_format_words() -> None:
  assert "procedur" not in name_stem_set("Cleaning Procedure.docx")
  assert "clean" in name_stem_set("Cleaning Procedure.docx")


def test_parse_json_object_is_strict() -> None:
  assert parse_json_object('prose {"a": 1} more') == {"a": 1}
  assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
  assert parse_json_object("{'a': 1}") is None, "Python dict literals must not parse"
  assert parse_json_object("[1,2]") is None
  assert parse_json_object(None) is None


def test_parse_scores_reads_object_list_and_loose_forms() -> None:
  assert parse_scores('{"scores":{"0":0.9,"1":0.1}}', 2) == {0: 0.9, 1: 0.1}
  assert parse_scores('{"scores":[0.5,0.25]}', 2) == {0: 0.5, 1: 0.25}
  assert parse_scores('garbage "0": 0.7, "1": 0.2 end', 2) == {0: 0.7, 1: 0.2}
  assert parse_scores("nothing here", 2) == {}
  assert parse_scores('{"scores":{"5":0.9}}', 2) == {}


def test_parse_scores_clamps_to_the_unit_interval() -> None:
  assert parse_scores('{"scores":{"0":5,"1":-2}}', 2) == {0: 1.0, 1: 0.0}


def test_parse_choice_handles_json_and_loose_forms() -> None:
  assert parse_choice('{"reasoning":"r","choice":2}') == (2, "r")
  assert parse_choice('"choice": -1') == (-1, "")
  assert parse_choice("no choice at all") == (None, "")


# ---------------------------------------------------------------------------
# question-shape detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "stem_text",
  [
    "List all of our validated assays",
    "What are all our SOPs for storage",
    "Which instruments are accredited",
  ],
)
def test_synthesis_questions_detected(stem_text: str) -> None:
  assert is_synthesis_question(stem_text).matched


def test_scoped_steps_question_is_not_synthesis() -> None:
  assert not is_synthesis_question(
    "What are the steps of the calibration procedure"
  ).matched


@pytest.mark.parametrize(
  "stem_text",
  [
    "Does a deviation always lead to a CAPA",
    "Is there a dedicated freezer for reagents",
    "Do we have a documented escalation path",
  ],
)
def test_polarity_questions_detected(stem_text: str) -> None:
  assert is_polarity_question(stem_text).matched


def test_definitional_question_detected_but_not_a_fact_question() -> None:
  assert is_definitional_question("What is Widgetron").matched
  assert is_definitional_question("What is the purpose of the CAPA log").matched
  assert not is_definitional_question(
    "What is the retention period for completed training records"
  ).matched


def test_recency_question_detected() -> None:
  verdict = is_recency_question("What is the current version of the quality manual")
  assert verdict.matched
  assert "current" in verdict.why
  assert not is_recency_question("What is the version numbering scheme").matched


# ---------------------------------------------------------------------------
# traversal state: frontier, reserve, decay
# ---------------------------------------------------------------------------


def _state() -> AgentState:
  return AgentState(config=TreeRAGConfig(), index=sample_tree())


def test_push_frontier_splits_on_the_noise_floor() -> None:
  state = _state()
  index = state.index
  state.push_frontier(
    [(index.nodes["doc-a"], 0.8, "r"), (index.nodes["doc-b"], 0.05, "r")], "Policies"
  )
  assert [f.node.node_id for f in state.frontier] == ["doc-a"]
  assert [f.node.node_id for f in state.reserve] == ["doc-b"]


def test_push_frontier_skips_visited_and_duplicates() -> None:
  state = _state()
  index = state.index
  state.visited.add("doc-b")
  state.push_frontier(
    [(index.nodes["doc-a"], 0.8, "r"), (index.nodes["doc-b"], 0.9, "r")], "Policies"
  )
  state.push_frontier([(index.nodes["doc-a"], 0.7, "r")], "Policies")
  assert len(state.frontier) == 1


def test_penalize_subtree_demotes_to_reserve_and_compounds() -> None:
  state = _state()
  index = state.index
  state.push_frontier([(index.nodes["sec-a2"], 0.8, "r")], "doc-a")
  moved = state.penalize_subtree("doc-a")
  assert moved == 1
  assert not state.frontier
  assert state.reserve[0].score == pytest.approx(0.8 * 0.30)
  state.penalize_subtree("doc-a")
  assert state.reserve[0].score == pytest.approx(0.8 * 0.30 * 0.30)


def test_pop_frontier_prefers_the_active_tier_then_the_reserve() -> None:
  state = _state()
  index = state.index
  state.push_frontier(
    [(index.nodes["doc-a"], 0.8, "r"), (index.nodes["doc-b"], 0.05, "r")], "Policies"
  )
  first, from_reserve = state.pop_frontier()
  assert first is not None and first.node.node_id == "doc-a" and not from_reserve
  second, from_reserve = state.pop_frontier()
  assert second is not None and second.node.node_id == "doc-b" and from_reserve
  empty, _ = state.pop_frontier()
  assert empty is None


def test_drop_frontier_entry_removes_by_identity() -> None:
  state = _state()
  entry = FrontierItem(node=state.index.nodes["doc-a"], score=0.9, reason="", origin="x")
  state.frontier.append(entry)
  state.drop_frontier_entry(entry)
  assert not state.frontier


def test_collect_marks_descendants_seen_when_taking_a_whole_unit() -> None:
  state = _state()
  assert state.collect(state.index.nodes["doc-a"], whole=True)
  assert "chk-a1" in state.seen
  assert not state.collect(state.index.nodes["doc-a"], whole=True)
  assert distinct_files(state.evidence) == 1


def test_collect_clips_when_asked() -> None:
  state = _state()
  state.collect(state.index.nodes["doc-a"], whole=True, clip_chars=20)
  assert len(state.evidence[0].content) <= 24


def test_add_memory_never_evicts_a_pinned_directive() -> None:
  memory = ["!! PINNED DIRECTIVE"]
  for i in range(60):
    add_memory(memory, f"fact number {i}", cap=5)
  assert memory[0] == "!! PINNED DIRECTIVE"
  assert len(memory) <= 6


def test_add_memory_ignores_empty_and_duplicate_facts() -> None:
  memory: list[str] = []
  add_memory(memory, "")
  add_memory(memory, "none")
  add_memory(memory, "a fact")
  add_memory(memory, "a fact")
  assert memory == ["a fact"]


# ---------------------------------------------------------------------------
# bounded retries and health checks
# ---------------------------------------------------------------------------


class _ExplodingClient(OllamaClient):
  """A client whose transport always fails, to exercise the retry bound."""

  def __init__(self, config: TreeRAGConfig) -> None:
    self._config = config
    self.attempts = 0

  @property
  def config(self) -> TreeRAGConfig:
    return self._config


def test_chat_raises_a_typed_error_once_retries_are_exhausted(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config = TreeRAGConfig(
    max_attempts=3,
    retry_deadline_s=1.0,
    retry_backoff_base_s=0.01,
    retry_backoff_cap_s=0.01,
  )
  client = OllamaClient.__new__(OllamaClient)
  attempts = {"n": 0}

  class _Transport:
    def chat(self, **kwargs: object) -> object:
      attempts["n"] += 1
      raise ConnectionError("tunnel is down")

  object.__setattr__(client, "_config", config)
  object.__setattr__(client, "_client", _Transport())
  object.__setattr__(client, "_lock", __import__("threading").Lock())
  object.__setattr__(client, "_health", None)
  object.__setattr__(client, "_embed_cache", {})

  with pytest.raises(OllamaUnavailableError, match="did not respond"):
    client.chat("hello", Counters())
  assert attempts["n"] == 3, "the retry loop must be bounded, not infinite"


def test_health_message_names_the_tunnel_only_for_a_forwarded_local_port() -> None:
  # A developer machine reaches Ollama through an SSH tunnel on a loopback port.
  local = HealthStatus(
    ok=False,
    endpoint="http://localhost:11434",
    model="gpt-oss:120b",
    detail="Ollama endpoint not reachable (ConnectError)",
    checked_at=0.0,
  )
  assert local.is_local_endpoint()
  assert "check the SSH tunnel" in local.ui_message()

  # The deployment server reaches Ollama directly, so naming a tunnel would mislead.
  remote = HealthStatus(
    ok=False,
    endpoint="http://host.docker.internal:11434",
    model="gpt-oss:120b",
    detail="Ollama endpoint not reachable (ConnectError)",
    checked_at=0.0,
  )
  assert not remote.is_local_endpoint()
  assert "SSH tunnel" not in remote.ui_message()
  assert "host.docker.internal:11434" in remote.ui_message()

  good = HealthStatus(ok=True, endpoint="e", model="m", detail="loaded", checked_at=0.0)
  assert good.ui_message() == ""


def test_llm_response_detects_reasoning_exhaustion() -> None:
  exhausted = LlmResponse(
    text="",
    thinking="lots of reasoning",
    done_reason="length",
    content_len=0,
    thinking_len=17,
  )
  assert exhausted.reasoning_exhausted()
  fine = LlmResponse(
    text="an answer", thinking="", done_reason="stop", content_len=9, thinking_len=0
  )
  assert not fine.reasoning_exhausted()


# ---------------------------------------------------------------------------
# a chunk with no source_file still resolves to something usable
# ---------------------------------------------------------------------------


def test_source_of_falls_back_to_path_then_name() -> None:
  bare = chunk("x", "Some Chunk", "text", "")
  bare.metadata.pop("source_file", None)
  bare.path = ""
  assert source_of(bare) == "Some Chunk"


# ---------------------------------------------------------------------------
# work budget: what keeps a search answerable
# ---------------------------------------------------------------------------


def test_budget_reports_a_reason_when_time_runs_out() -> None:
  budget = SearchBudget(traversal_seconds=0.0, answer_seconds=60.0, max_calls=100)
  assert "time budget" in budget.traversal_exhausted()
  assert budget.remaining() == 0.0


def test_budget_reports_a_reason_when_calls_run_out() -> None:
  budget = SearchBudget(traversal_seconds=600.0, answer_seconds=60.0, max_calls=2)
  assert budget.traversal_exhausted() == ""
  budget.note_call()
  budget.note_call()
  assert "LLM calls" in budget.traversal_exhausted()


def test_budget_leaves_room_to_write_the_answer() -> None:
  # Navigation is out of time, but check() must not fire: the answer allowance is what
  # guarantees the user still gets an answer.
  budget = SearchBudget(traversal_seconds=0.0, answer_seconds=60.0, max_calls=100)
  assert budget.traversal_exhausted()
  budget.check()
  budget.begin_answer()
  budget.check()


def test_budget_hard_check_raises_once_even_the_reserve_is_gone() -> None:
  budget = SearchBudget(traversal_seconds=0.0, answer_seconds=0.001, max_calls=100)
  time.sleep(0.01)
  with pytest.raises(SearchBudgetError, match="ceiling"):
    budget.check()


def test_budget_hard_check_raises_past_the_exact_call_ceiling() -> None:
  budget = SearchBudget(traversal_seconds=600.0, answer_seconds=60.0, max_calls=10)
  for _ in range(9):
    budget.note_call()
  budget.check()
  budget.note_call()
  with pytest.raises(SearchBudgetError, match="ceiling"):
    budget.check()


def test_quick_mode_is_strictly_cheaper_than_thorough() -> None:
  quick = TreeRAGConfig.for_mode(TreeRAGMode.QUICK)
  thorough = TreeRAGConfig.for_mode(TreeRAGMode.THOROUGH)
  assert quick.mode is TreeRAGMode.QUICK
  assert thorough.mode is TreeRAGMode.THOROUGH
  for field_name in (
    "time_budget_s",
    "max_llm_calls",
    "max_rank_candidates",
    "max_steps",
    "max_files",
    "max_evidence",
  ):
    assert getattr(quick, field_name) < getattr(thorough, field_name), field_name
  # Quick drops the two mechanisms that cost the most for the least frequent benefit.
  assert quick.tie_max_explore == 0
  assert thorough.tie_max_explore == 3


def test_thorough_mode_preserves_the_canonical_navigation_knobs() -> None:
  thorough = TreeRAGConfig.for_mode(TreeRAGMode.THOROUGH)
  assert thorough.max_steps == 40
  assert thorough.max_files == 8
  assert thorough.max_evidence == 50
  assert thorough.noise_floor == pytest.approx(0.20)
  assert thorough.cluster_delta == pytest.approx(0.15)
  assert thorough.score_batch == 5
  assert thorough.tie_max_explore == 3
  assert thorough.deferred_max_reads == 4


def test_mode_preset_keeps_the_endpoint_and_tree() -> None:
  base = TreeRAGConfig(ollama_url="http://example.invalid:1234", model="m:1b")
  quick = base.with_mode(TreeRAGMode.QUICK)
  assert quick.ollama_url == "http://example.invalid:1234"
  assert quick.model == "m:1b"
  assert quick.tree_path == base.tree_path


# ---------------------------------------------------------------------------
# fan-out cap: the reason a query could run for hours
# ---------------------------------------------------------------------------


def _wide_context(cap: int) -> tuple[SearchContext, list[TreeNode]]:
  ctx = SearchContext(
    config=TreeRAGConfig(max_rank_candidates=cap),
    client=FakeClient(lambda p: '{"scores":{"0":0.5}}'),
    index=sample_tree(),
  )
  kids = [
    chunk(f"n{i}", f"Section {i} about storage", "text", "A/Doc.docx") for i in range(200)
  ]
  return ctx, kids


def test_narrow_nodes_are_untouched_by_the_fan_out_cap() -> None:
  ctx, _ = _wide_context(60)
  kids = ctx.index.nodes["doc-a"].children
  selected, overflow = shortlist_by_fanout(ctx, "calibration", kids)
  assert selected == list(kids), "a node under the cap must be passed through unchanged"
  assert overflow == []


def test_wide_nodes_are_capped_and_the_remainder_deferred() -> None:
  ctx, kids = _wide_context(25)
  selected, overflow = shortlist_by_fanout(ctx, "storage temperature", kids)
  assert len(selected) == 25
  assert len(overflow) == 175
  assert len(selected) + len(overflow) == len(kids), "nothing may be dropped"


def test_the_cap_keeps_the_question_relevant_names() -> None:
  ctx, _ = _wide_context(3)
  kids = [
    chunk("a", "Unrelated appendix", "x", "A/Doc.docx"),
    chunk("b", "Freezer temperature monitoring", "x", "A/Doc.docx"),
    chunk("c", "Another unrelated annex", "x", "A/Doc.docx"),
    chunk("d", "Storage temperature limits", "x", "A/Doc.docx"),
    chunk("e", "Yet another annex", "x", "A/Doc.docx"),
  ]
  selected, _ = shortlist_by_fanout(ctx, "what is the storage temperature", kids)
  names = {n.name for n in selected}
  assert "Storage temperature limits" in names
  assert "Freezer temperature monitoring" in names


def test_overflow_lands_in_the_reserve_tier_not_the_bin() -> None:
  # Overflow is returned at score 0.0, which is below the noise floor, so the existing
  # frontier push routes it to the reserve where it stays reachable.
  state = AgentState(config=TreeRAGConfig(), index=sample_tree())
  overflow = [chunk(f"o{i}", f"Section {i}", "x", "A/Doc.docx") for i in range(3)]
  state.push_frontier([(n, 0.0, "deferred") for n in overflow], "Doc")
  assert not state.frontier
  assert len(state.reserve) == 3
  popped, from_reserve = state.pop_frontier()
  assert popped is not None and from_reserve
