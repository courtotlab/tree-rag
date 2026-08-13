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

Every tuning constant of the extracted TreeQuest agent lives here as a field of one frozen
dataclass, with the benchmarked value as its default. The source notebook scattered these
as module-level globals; the values are unchanged, only their home is.
"""

import os
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path


class TreeRagMode(str, Enum):
  """How much work a TreeQuest search is allowed to do.

  Attributes:
    QUICK: A bounded look, aimed at roughly two minutes. Narrower budgets and none of the
      pre-answer cross-checking; it answers from the first good evidence it finds.
    THOROUGH: The high-budget operating point used by the modular release. The immutable
      evaluated-v0 runner is retained separately for exact result provenance.
  """

  QUICK = "quick"
  THOROUGH = "thorough"


#: Last-resort endpoint, used only when the deployment configures nothing at all. This is
#: the Ollama project's own default address; a real deployment always sets the endpoint,
#: either through the shared OLLAMA_HOST/OLLAMA_PORT pair or through TREEQUEST_OLLAMA_URL.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11528"
DEFAULT_MODEL = "gpt-oss:120b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TREE_PATH = "corpus_tree.json"
IMPLEMENTATION_VERSION = "modular-v1"

_ENV_URL = "TREEQUEST_OLLAMA_URL"
_ENV_MODEL = "TREEQUEST_MODEL"
_ENV_EMBED_MODEL = "TREEQUEST_EMBED_MODEL"
_ENV_TREE = "TREEQUEST_TREE_PATH"
_ENV_ENABLED = "TREEQUEST_ENABLED"

#: The endpoint the existing vector-search agent already uses. TreeQuest follows it rather
#: than carrying a second copy of the same address, so there is one place to configure
#: Ollama for the deployment and the two modes cannot drift apart.
_ENV_SHARED_HOST = "OLLAMA_HOST"
_ENV_SHARED_PORT = "OLLAMA_PORT"


def resolve_ollama_url() -> str:
  """Work out which Ollama endpoint to use, from the deployment's environment.

  Resolution order:

  1. ``TREEQUEST_OLLAMA_URL``, when TreeQuest needs to point somewhere of its own.
  2. ``OLLAMA_HOST`` and ``OLLAMA_PORT`` - the pair the vector-search agent already uses.
     Following it means the deployment configures Ollama once and both retrieval modes
     agree; whatever the server reaches Ollama by, TreeQuest uses the same thing.
  3. :data:`DEFAULT_OLLAMA_URL`, only when nothing at all is configured.

  ``OLLAMA_HOST`` is accepted with or without a scheme, because the existing agent
  interpolates it as ``{host}:{port}`` and deployments write it both ways.

  Returns:
    The base URL of the Ollama server.
  """
  explicit = os.getenv(_ENV_URL, "").strip()
  if explicit:
    return explicit

  host = os.getenv(_ENV_SHARED_HOST, "").strip()
  if not host:
    return DEFAULT_OLLAMA_URL

  port = os.getenv(_ENV_SHARED_PORT, "").strip()
  if "://" not in host:
    host = f"http://{host}"
  host = host.rstrip("/")
  # A host that already carries its own port is used as given.
  if port and ":" not in host.split("://", 1)[1]:
    return f"{host}:{port}"
  return host


@dataclass(frozen=True, slots=True)
class TreeRagConfig:
  """Configuration for one TreeQuest search.

  Defaults preserve the evaluated controller's high-level operating point while applying
  the hardened modular-v1 semantics documented in REPRODUCIBILITY.md. Exact evaluated-v0
  source is retained separately; changing navigation fields changes retrieval behavior.

  Attributes:
    ollama_url: Base URL of the Ollama server hosting the agent model.
    model: Name of the agent model, e.g. ``gpt-oss:120b``.
    embed_model: Name of the embedding model used to order wide folder listings.
    tree_path: Filesystem path of the corpus tree JSON.
    request_timeout_s: Per-request timeout handed to the Ollama HTTP client.
    keep_alive: How long Ollama should keep the model resident between calls.
    thinking: Whether to enable the model's reasoning channel (latency lever).
    max_attempts: Bounded retry count for one LLM call before giving up.
    retry_deadline_s: Total wall-clock budget for the retries of one LLM call.
    retry_backoff_base_s: First backoff interval; doubles up to retry_backoff_cap_s.
    retry_backoff_cap_s: Upper bound on a single backoff interval.
    health_timeout_s: Timeout for a health check probe.
    health_ttl_s: How long a health verdict may be reused before re-probing.
    mode: Which named preset this configuration came from.
    time_budget_s: Wall-clock allowance for navigation, after which the agent stops and
      answers from the evidence it holds.
    answer_budget_s: Additional wall-clock allowance reserved for writing the answer, so
      exhausting the search budget never costs the user the answer.
    max_llm_calls: Exact ceiling on successful LLM calls for one search, including answer
      generation. A reserve for answer generation is held inside this ceiling.
    max_rank_candidates: Most children scored in one ranking decision. Ranking costs one
      LLM call per five candidates, so an uncapped wide node is what turns a query into
      hours; anything past the cap is deferred to the reserve tier rather than dropped.
    max_embed_per_decision: Most names embedded when ordering one over-long list.
      Defaults to 0, i.e. lexical ordering only. Each embedding is a round-trip against a
      DIFFERENT model on the same Ollama server; a measured chat call costs ~4s while the
      agent model is resident and ~43s when it has been evicted and its 65GB has to be
      reloaded, so interleaving a second model into the hot path risks a tenfold penalty
      on every subsequent call. Lexical overlap is what the ranking already treats as the
      primary signal. Raise this to re-enable embedding similarity as a tiebreak on
      servers with headroom for both models.
    max_branch: Retained options shown per decision.
    score_batch: Children scored per LLM call, small so the JSON never truncates.
    score_preview: Characters of a candidate's actual text shown to the ranker.
    preview_scan_cap: Max chunks walked to build one preview.
    preview_relevance: Build previews from question-relevant chunks, not opening words.
    contains_chars: Character budget for a folder candidate's ``contains:`` line.
    deferred_max_reads: Deferred same-file sections readable after an insufficiency.
    max_steps: Total node visits; a hard cap so teleporting can never loop forever.
    max_files: Distinct evidence documents collected before answering.
    max_evidence: Cap on evidence pieces fed to the final answer.
    synth_max_files: Distinct whole documents gathered by synthesis mode.
    synth_file_chars: Clip applied to each whole-document synthesis unit.
    synth_breadth_window: Relevance window that extends synthesis past its base cap.
    answer_max_words: The house answer limit, enforced only by the prompt.
    noise_floor: Candidates below this go to the reserve tier, not the active frontier.
    nosignal_abort: Consecutive all-floor folder rankings before abandoning a line.
    cluster_delta: Same-file candidates within this of the committed score are auto-read.
    cluster_high: A same-file candidate this relevant is auto-read regardless.
    parts_max: Max distinct sub-questions the decomposition may name.
    part_select_min: Bundle size below which part-coverage selection is skipped.
    part_keep: Evidence pieces led with per sub-question.
    name_boost_step: Nudge per distinctive stemmed token shared with a candidate name.
    name_boost_max: Cap on the name-match nudge.
    name_boost_window: Proximity to the top raw score required to earn the nudge.
    shortlist_region_quota: Max shortlist slots one top-level region may occupy.
    triage_batch: Heading/score/snippet lines per intra-file triage call.
    reject_decay: Score multiplier applied to a rejected subtree's frontier entries.
    cluster_reject_n: Barren files under one folder before the folder is demoted.
    reject_sweep_min: Line score above which a reject still triggers a same-file sweep.
    tie_window: Score window making an unexplored other-document entry a near-tie.
    tie_max_explore: Max extra near-tied documents visited before answering.
    single_src_window: Widened contrast band used while the answer is single-sourced.
    alt_shortlist: Eligible alternatives shown to the contrast chooser.
    alt_preview: Characters of a candidate's own words shown to the contrast chooser.
    resid_shortlist: Frontier entries offered to the residual-aimed chooser.
    lex_seed_k: Max files seeded onto the frontier from the lexical scan.
    lex_seed_cap: Frontier score a full distinctive-term lexical match seeds at.
    lex_rare_df_frac: Document-frequency fraction below which a token counts as rare.
    stall_trigger: Consecutive insufficiency verdicts before a breadth escape.
    stall_min_docs: Distinct evidence documents required before a stall may count.
    defn_max_push: Times the provenance rule may override a sufficient verdict.
  """

  # ---- endpoint / runtime ----
  ollama_url: str = DEFAULT_OLLAMA_URL
  model: str = DEFAULT_MODEL
  embed_model: str = DEFAULT_EMBED_MODEL
  tree_path: Path = field(default_factory=lambda: Path(DEFAULT_TREE_PATH))
  request_timeout_s: float = 1500.0
  keep_alive: str = "30m"
  thinking: bool = False

  # ---- bounded retry policy (replaces the notebook's unbounded retry loops) ----
  max_attempts: int = 4
  retry_deadline_s: float = 120.0
  retry_backoff_base_s: float = 2.0
  retry_backoff_cap_s: float = 30.0
  health_timeout_s: float = 5.0
  health_ttl_s: float = 30.0

  # ---- work budget: what keeps a search answerable ----
  mode: TreeRagMode = TreeRagMode.THOROUGH
  time_budget_s: float = 900.0
  answer_budget_s: float = 120.0
  max_llm_calls: int = 400
  max_rank_candidates: int = 60
  max_embed_per_decision: int = 0

  # ---- ranking / preview ----
  max_branch: int = 6
  score_batch: int = 5
  score_preview: int = 700
  preview_scan_cap: int = 60
  preview_relevance: bool = True
  contains_chars: int = 1500

  # ---- budgets ----
  deferred_max_reads: int = 4
  max_steps: int = 40
  max_files: int = 8
  max_evidence: int = 50

  # ---- synthesis (enumeration) mode ----
  synth_max_files: int = 4
  synth_file_chars: int = 8000
  synth_breadth_window: float = 0.20

  # ---- answer ----
  answer_max_words: int = 100

  # ---- navigation thresholds ----
  noise_floor: float = 0.20
  nosignal_abort: int = 2
  cluster_delta: float = 0.15
  cluster_high: float = 0.60

  # ---- compound questions ----
  parts_max: int = 4
  part_select_min: int = 6
  part_keep: int = 2

  # ---- name-match boost ----
  name_boost_step: float = 0.06
  name_boost_max: float = 0.12
  name_boost_window: float = 0.25

  # ---- shortlists ----
  shortlist_region_quota: int = 3
  triage_batch: int = 25

  # ---- reject decay / sweeps ----
  reject_decay: float = 0.30
  cluster_reject_n: int = 3
  reject_sweep_min: float = 0.60

  # ---- near-tie / contrast ----
  tie_window: float = 0.05
  tie_max_explore: int = 3
  single_src_window: float = 0.5
  alt_shortlist: int = 8
  alt_preview: int = 400
  resid_shortlist: int = 8

  # ---- lexical seeding ----
  lex_seed_k: int = 3
  lex_seed_cap: float = 0.55
  lex_rare_df_frac: float = 0.15

  # ---- breadth escape ----
  stall_trigger: int = 3
  stall_min_docs: int = 2
  defn_max_push: int = 6

  def __post_init__(self) -> None:
    """Validate the configuration as soon as it is constructed.

    Raises:
      ValueError: If any field holds a value outside its permitted range - a
        non-positive budget, a probability-like threshold outside ``[0, 1]``, a
        non-positive timeout, or an empty endpoint/model string.
    """
    self.validate()

  def validate(self) -> None:
    """Check every field against its permitted range.

    Called automatically by ``__post_init__``; exposed separately so callers that build a
    config field by field can check it before use.

    Raises:
      ValueError: If any field holds a value outside its permitted range - a
        non-positive budget, a probability-like threshold outside ``[0, 1]``, a
        non-positive timeout, or an empty endpoint/model string.
    """
    for name in ("ollama_url", "model", "embed_model", "keep_alive"):
      if not str(getattr(self, name)).strip():
        raise ValueError(f"TreeRagConfig.{name} must be a non-empty string")

    positive_ints = (
      "max_attempts",
      "max_branch",
      "score_batch",
      "score_preview",
      "preview_scan_cap",
      "contains_chars",
      "max_steps",
      "max_files",
      "max_evidence",
      "synth_max_files",
      "synth_file_chars",
      "answer_max_words",
      "nosignal_abort",
      "parts_max",
      "part_select_min",
      "part_keep",
      "shortlist_region_quota",
      "triage_batch",
      "cluster_reject_n",
      "max_llm_calls",
      "max_rank_candidates",
      "alt_shortlist",
      "alt_preview",
      "resid_shortlist",
      "lex_seed_k",
      "stall_trigger",
      "stall_min_docs",
      "defn_max_push",
    )
    for name in positive_ints:
      value = int(getattr(self, name))
      if value <= 0:
        raise ValueError(f"TreeRagConfig.{name} must be > 0, got {value}")

    # Zero is meaningful for both: it switches the mechanism off, which is how the quick
    # mode drops the pre-answer cross-check and the deferred-section recovery.
    for name in ("deferred_max_reads", "tie_max_explore", "max_embed_per_decision"):
      value = int(getattr(self, name))
      if value < 0:
        raise ValueError(f"TreeRagConfig.{name} must be >= 0, got {value}")

    positive_floats = (
      "request_timeout_s",
      "retry_deadline_s",
      "retry_backoff_base_s",
      "retry_backoff_cap_s",
      "health_timeout_s",
      "time_budget_s",
      "answer_budget_s",
    )
    for name in positive_floats:
      fvalue = float(getattr(self, name))
      if fvalue <= 0.0:
        raise ValueError(f"TreeRagConfig.{name} must be > 0, got {fvalue}")

    if self.health_ttl_s < 0.0:
      raise ValueError(
        f"TreeRagConfig.health_ttl_s must be >= 0, got {self.health_ttl_s}"
      )

    unit_floats = (
      "synth_breadth_window",
      "noise_floor",
      "cluster_delta",
      "cluster_high",
      "name_boost_step",
      "name_boost_max",
      "name_boost_window",
      "reject_decay",
      "reject_sweep_min",
      "tie_window",
      "single_src_window",
      "lex_seed_cap",
      "lex_rare_df_frac",
    )
    for name in unit_floats:
      ufvalue = float(getattr(self, name))
      if not 0.0 <= ufvalue <= 1.0:
        raise ValueError(f"TreeRagConfig.{name} must be in [0, 1], got {ufvalue}")

    if self.max_evidence < self.max_files:
      raise ValueError(
        "TreeRagConfig.max_evidence must be >= max_files "
        f"({self.max_evidence} < {self.max_files})"
      )
    if self.retry_backoff_cap_s < self.retry_backoff_base_s:
      raise ValueError(
        "TreeRagConfig.retry_backoff_cap_s must be >= retry_backoff_base_s "
        f"({self.retry_backoff_cap_s} < {self.retry_backoff_base_s})"
      )

  def with_mode(self, mode: TreeRagMode) -> "TreeRagConfig":
    """Return this configuration retuned for a named mode.

    The endpoint, model and tree path are kept; only the work budget changes.

    ``THOROUGH`` is the benchmarked configuration unchanged, plus the wall-clock ceiling.
    ``QUICK`` narrows every budget and switches off the two mechanisms that cost the most
    for the least frequent benefit - the pre-answer contrast excursion and the deferred
    section recovery - so it answers from the first good evidence it finds.

    Args:
      mode: The preset to apply.

    Returns:
      A new, validated configuration.

    Raises:
      ValueError: If the resulting configuration is out of range.
    """
    if mode is TreeRagMode.THOROUGH:
      return replace(
        self,
        mode=mode,
        time_budget_s=900.0,
        answer_budget_s=120.0,
        max_llm_calls=400,
        max_rank_candidates=60,
        max_embed_per_decision=0,
        max_steps=40,
        max_files=8,
        max_evidence=50,
        tie_max_explore=3,
        deferred_max_reads=4,
        score_preview=700,
        contains_chars=1500,
        preview_scan_cap=60,
      )
    return replace(
      self,
      mode=mode,
      time_budget_s=240.0,
      answer_budget_s=60.0,
      max_llm_calls=60,
      max_rank_candidates=25,
      max_embed_per_decision=0,
      max_steps=12,
      max_files=3,
      max_evidence=15,
      tie_max_explore=0,
      deferred_max_reads=1,
      score_preview=400,
      contains_chars=700,
      preview_scan_cap=30,
      synth_max_files=3,
      alt_shortlist=5,
      resid_shortlist=5,
    )

  @classmethod
  def for_mode(cls, mode: TreeRagMode) -> "TreeRagConfig":
    """Build a configuration for a named mode, reading the endpoint from the environment.

    Args:
      mode: The preset to apply.

    Returns:
      A validated configuration.

    Raises:
      ValueError: If an environment variable holds a value outside its permitted range.
    """
    return cls.from_env().with_mode(mode)

  @classmethod
  def from_env(cls) -> "TreeRagConfig":
    """Build a configuration from environment variables, falling back to the defaults.

    Reads ``TREEQUEST_MODEL``, ``TREEQUEST_EMBED_MODEL`` and ``TREEQUEST_TREE_PATH``, and
    resolves the endpoint through :func:`resolve_ollama_url` - which prefers
    ``TREEQUEST_OLLAMA_URL`` and otherwise follows the ``OLLAMA_HOST``/``OLLAMA_PORT`` pair
    the vector-search agent already uses. No host name is hardcoded anywhere.

    Returns:
      A validated configuration.

    Raises:
      ValueError: If an environment variable holds a value outside its permitted range.
    """
    return cls(
      ollama_url=resolve_ollama_url(),
      model=os.getenv(_ENV_MODEL, DEFAULT_MODEL),
      embed_model=os.getenv(_ENV_EMBED_MODEL, DEFAULT_EMBED_MODEL),
      tree_path=Path(os.getenv(_ENV_TREE, DEFAULT_TREE_PATH)),
    )

  def describe(self) -> dict[str, str]:
    """Summarise the configuration for logs and the diagnostics CLI.

    Returns:
      A mapping of field name to its string form. Contains no credentials: the endpoint
      is a URL the operator supplied and the remaining fields are numeric knobs.
    """
    return {f.name: str(getattr(self, f.name)) for f in fields(self)}


def treerag_enabled() -> bool:
  """Report whether the TreeQuest search mode is switched on for this deployment.

  Controlled by ``TREEQUEST_ENABLED``; anything other than a recognised false value enables
  it, so the mode is available by default once the package is installed.

  Returns:
    True when TreeQuest should be offered in the UI.
  """
  raw = os.getenv(_ENV_ENABLED)
  if raw is None:
    return True
  return raw.strip().lower() not in ("0", "false", "no", "off", "")
