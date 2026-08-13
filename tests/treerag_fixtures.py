"""Shared fixtures for the TreeRAG tests.

Every tree here is invented: the names are generic laboratory-management placeholders, not
corpus documents, so no real document name, path or passage ever reaches a test fixture.
"""

from collections.abc import Callable, Sequence

from treerag.client import Counters, HealthStatus, LlmResponse, OllamaClient
from treerag.config import TreeRAGConfig
from treerag.context import SearchContext
from treerag.tree import TreeIndex, TreeStats
from treerag.types import NodeMetadata, TreeNode

#: A scripted responder: given the prompt, return the model's reply.
Responder = Callable[[str], str]


class FakeClient(OllamaClient):
  """An OllamaClient stand-in that never touches the network.

  Attributes:
    prompts: Every prompt it was asked to complete, in order.
    healthy: What health_check should report.
  """

  def __init__(self, responder: Responder, *, healthy: bool = True) -> None:
    """Build a fake client around a scripted responder.

    Args:
      responder: Called with each prompt; returns the reply text.
      healthy: What health_check should report.
    """
    self._responder = responder
    self._fake_config = TreeRAGConfig()
    self.prompts: list[str] = []
    self.healthy = healthy

  @property
  def config(self) -> TreeRAGConfig:
    """The configuration this fake was built with.

    Returns:
      A default configuration.
    """
    return self._fake_config

  def chat(
    self,
    prompt: str,
    counters: Counters,
    *,
    num_predict: int = 512,
    temperature: float = 0.0,
    think: bool | None = None,
    thinking_fallback: bool = True,
  ) -> LlmResponse:
    """Return the scripted reply for a prompt.

    Args:
      prompt: The prompt to answer.
      counters: Tally object, updated as a real call would.
      num_predict: Ignored.
      temperature: Ignored.
      think: Ignored.
      thinking_fallback: Ignored.

    Returns:
      The scripted response.
    """
    self.prompts.append(prompt)
    text = self._responder(prompt)
    counters.calls += 1
    counters.in_tok += len(prompt) // 4
    counters.out_tok += len(text) // 4
    return LlmResponse(
      text=text,
      thinking="",
      done_reason="stop",
      content_len=len(text),
      thinking_len=0,
    )

  def embed(self, text: str) -> list[float] | None:
    """Report that no embedder is available.

    Args:
      text: Ignored.

    Returns:
      None, so callers exercise the lexical fallback.
    """
    return None

  def health_check(self, *, force: bool = False) -> HealthStatus:
    """Report the configured health verdict.

    Args:
      force: Ignored.

    Returns:
      A healthy or unhealthy status, per the flag given at construction.
    """
    return HealthStatus(
      ok=self.healthy,
      endpoint=self._fake_config.ollama_url,
      model=self._fake_config.model,
      detail="loaded" if self.healthy else "Ollama endpoint not reachable (ConnectError)",
      checked_at=0.0,
    )


def chunk(node_id: str, name: str, text: str, source: str) -> TreeNode:
  """Build a leaf chunk node.

  Args:
    node_id: The node identifier.
    name: The chunk's display name.
    text: The chunk's body text.
    source: The source document path it belongs to.

  Returns:
    The chunk node.
  """
  meta = NodeMetadata(source_file=source, kind="text", unit_index=0)
  return TreeNode(
    node_id=node_id,
    node_type="chunk",
    name=name,
    path=source,
    summary="",
    content=text,
    metadata=meta,
  )


def branch(
  node_id: str,
  node_type: str,
  name: str,
  summary: str,
  children: Sequence[TreeNode],
  source: str = "",
) -> TreeNode:
  """Build an interior node.

  Args:
    node_id: The node identifier.
    node_type: One of the interior node types.
    name: The node's display name.
    summary: Its summary.
    children: Its children, in order.
    source: The source document path, for document-level nodes.

  Returns:
    The interior node.
  """
  meta = NodeMetadata(source_file=source) if source else NodeMetadata()
  return TreeNode(
    node_id=node_id,
    node_type=node_type,
    name=name,
    path=source,
    summary=summary,
    content="",
    children=list(children),
    metadata=meta,
  )


def sample_tree() -> TreeIndex:
  """Build a small, entirely invented corpus tree.

  Returns:
    An indexed tree with two top-level folders, three documents and several sections.
  """
  doc_a = "Policies/Instrument Calibration Policy.docx"
  doc_b = "Policies/Sample Storage Policy.docx"
  doc_c = "Training/Pipette Training Record.docx"

  calibration = branch(
    "doc-a",
    "document",
    "Instrument Calibration Policy",
    "How instruments are calibrated and how often.",
    [
      branch(
        "sec-a1",
        "section",
        "Calibration Frequency",
        "States the calibration interval.",
        [
          chunk(
            "chk-a1",
            "Calibration Frequency 1",
            "Balances are calibrated every six months by the metrology vendor.",
            doc_a,
          )
        ],
        source=doc_a,
      ),
      branch(
        "sec-a2",
        "section",
        "Responsibilities",
        "Who signs off on calibration.",
        [
          chunk(
            "chk-a2",
            "Responsibilities 1",
            "The laboratory manager approves each calibration certificate.",
            doc_a,
          )
        ],
        source=doc_a,
      ),
    ],
    source=doc_a,
  )
  storage = branch(
    "doc-b",
    "document",
    "Sample Storage Policy",
    "Storage temperatures and retention.",
    [
      branch(
        "sec-b1",
        "section",
        "Storage Temperature",
        "States the freezer set point.",
        [
          chunk(
            "chk-b1",
            "Storage Temperature 1",
            "Extracted nucleic acid is stored at minus eighty degrees Celsius.",
            doc_b,
          )
        ],
        source=doc_b,
      )
    ],
    source=doc_b,
  )
  training = branch(
    "doc-c",
    "document",
    "Pipette Training Record",
    "A completed training record.",
    [
      branch(
        "sec-c1",
        "section",
        "Trainee Sign-off",
        "Signatures for a training session.",
        [
          chunk(
            "chk-c1",
            "Trainee Sign-off 1",
            "Trainee signature recorded on completion of pipette training.",
            doc_c,
          )
        ],
        source=doc_c,
      )
    ],
    source=doc_c,
  )

  policies = branch(
    "fold-policies",
    "folder",
    "Policies",
    "Laboratory policies.",
    [calibration, storage],
  )
  records = branch("fold-training", "folder", "Training", "Training records.", [training])
  root = branch("root", "root", "Corpus", "The whole corpus.", [policies, records])

  stats = TreeStats(
    path=__import__("pathlib").Path("test-fixture.json"),
    file_mb=0.0,
    parse_seconds=0.0,
    build_seconds=0.0,
    index_seconds=0.0,
    total_seconds=0.0,
    peak_rss_mb=0.0,
    rss_delta_mb=0.0,
    nodes=0,
    documents=3,
    chunks=4,
    top_level=2,
  )
  nodes: dict[str, TreeNode] = {}
  parent: dict[str, str] = {}
  stack: list[tuple[TreeNode, str | None]] = [(root, None)]
  while stack:
    node, parent_id = stack.pop()
    nodes[node.node_id] = node
    if parent_id is not None:
      parent[node.node_id] = parent_id
    for child in node.children:
      stack.append((child, node.node_id))
  return TreeIndex(root=root, nodes=nodes, parent=parent, stats=stats)


def make_context(responder: Responder) -> SearchContext:
  """Build a search context wired to a fake client and the sample tree.

  Args:
    responder: The scripted responder for the fake client.

  Returns:
    A context whose LLM calls never leave the process.
  """
  return SearchContext(
    config=TreeRAGConfig(),
    client=FakeClient(responder),
    index=sample_tree(),
  )
