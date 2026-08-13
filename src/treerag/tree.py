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

Loading the corpus tree and answering structural questions about it.

The tree is roughly 218 MB of JSON; parsing it per request is fatal, so it is loaded once
into a process-wide singleton and the node and parent indices are built once alongside it.
The source notebook kept those indices in module-level ``_NODES`` / ``_PARENT`` dicts,
which cannot serve two trees or two threads; they are fields of :class:`TreeIndex` here.

The tree is READ-ONLY. Nothing in this module writes to the tree path, and nothing
triggers a build or a repair: indexing is a manual action, never a side effect of serving
a query.
"""

import json
import math
import resource
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from loguru import logger

from treerag.config import TreeRAGConfig
from treerag.errors import MalformedTreeError, TreeUnavailableError
from treerag.text import full, tokens
from treerag.types import TreeNode, node_from_dict


def _max_rss_mb() -> float:
  """Read the process's peak resident set size in megabytes.

  Args:
    None.

  Returns:
    Peak RSS in MB. ``ru_maxrss`` is reported in bytes on macOS and in kilobytes on
    Linux, and both are normalised here.
  """
  raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
  return float(raw) / divisor


@dataclass(frozen=True, slots=True)
class TreeStats:
  """Measured cost of loading the corpus tree.

  Attributes:
    path: The tree file that was loaded.
    file_mb: Size of the tree file on disk, in MB.
    parse_seconds: Seconds spent decoding the JSON.
    build_seconds: Seconds spent converting it into nodes.
    index_seconds: Seconds spent building the node and parent indices.
    total_seconds: Total cold-start cost.
    peak_rss_mb: Process peak resident set size after loading, in MB.
    rss_delta_mb: Increase in peak RSS attributable to the load, in MB.
    nodes: Total node count.
    documents: Number of document-level nodes.
    chunks: Number of leaf chunks.
    top_level: Number of root children.
  """

  path: Path
  file_mb: float
  parse_seconds: float
  build_seconds: float
  index_seconds: float
  total_seconds: float
  peak_rss_mb: float
  rss_delta_mb: float
  nodes: int
  documents: int
  chunks: int
  top_level: int

  def summary(self) -> str:
    """Render the statistics as one log line.

    Returns:
      A single line naming the counts and the measured cost. It deliberately reports the
      file size and node counts rather than any document name.
    """
    return (
      f"corpus tree loaded: {self.nodes} nodes "
      f"({self.documents} documents, {self.chunks} chunks, {self.top_level} top-level) "
      f"from {self.file_mb:.0f} MB in {self.total_seconds:.2f}s "
      f"(parse {self.parse_seconds:.2f}s, build {self.build_seconds:.2f}s, "
      f"index {self.index_seconds:.2f}s); "
      f"peak RSS {self.peak_rss_mb:.0f} MB (+{self.rss_delta_mb:.0f} MB)"
    )


@dataclass(slots=True)
class LexicalIndex:
  """A cached, LLM-free scan of every chunk's body tokens.

  Attributes:
    source_tokens: Per source document, the set of body tokens it contains.
    source_node: Per source document, the node to seed onto the frontier for it.
    document_frequency: Per token, how many documents contain it.
    n_documents: Number of documents scanned, at least one.
  """

  source_tokens: dict[str, set[str]]
  source_node: dict[str, TreeNode]
  document_frequency: Counter[str]
  n_documents: int


@dataclass(slots=True)
class TreeIndex:
  """The loaded corpus tree plus the indices the agent navigates it with.

  Attributes:
    root: The tree's root node.
    nodes: Every node keyed by ``node_id``.
    parent: Each node's parent id, keyed by ``node_id``; the root is absent.
    stats: Measured cost of loading this tree.
    lexical: The lexical body index, built on first use.
  """

  root: TreeNode
  nodes: dict[str, TreeNode]
  parent: dict[str, str]
  stats: TreeStats
  lexical: LexicalIndex | None = field(default=None)

  # ---- structural queries -------------------------------------------------

  def node_of(self, node_id: str) -> TreeNode | None:
    """Look a node up by identifier.

    Args:
      node_id: The identifier to look up.

    Returns:
      The node, or ``None`` when no node carries that identifier.
    """
    return self.nodes.get(node_id)

  def parent_of(self, node: TreeNode) -> TreeNode | None:
    """Find a node's parent.

    Args:
      node: The node whose parent is wanted.

    Returns:
      The parent node, or ``None`` for the root.
    """
    parent_id = self.parent.get(node.node_id)
    if parent_id is None:
      return None
    return self.nodes.get(parent_id)

  def is_under(self, node_id: str, root_id: str) -> bool:
    """Report whether one node lies in another's subtree.

    Args:
      node_id: The candidate descendant.
      root_id: The candidate ancestor.

    Returns:
      True when ``node_id`` is ``root_id`` or lies beneath it.
    """
    current = node_id
    while True:
      if current == root_id:
        return True
      parent_id = self.parent.get(current)
      if parent_id is None:
        return False
      current = parent_id

  def region_of(self, node_id: str) -> str:
    """Find the top-level region a node lives under.

    A region is a child of the root: the unit the breadth escape reasons about. A root
    child returns itself, and so does the root.

    Args:
      node_id: The node whose region is wanted.

    Returns:
      The identifier of the region's entry node.
    """
    current = node_id
    while current in self.parent:
      parent_id = self.parent[current]
      parent = self.nodes.get(parent_id)
      if parent is None or parent.node_type == "root":
        return current
      current = parent_id
    return current

  def file_of(self, chunk_id: str) -> TreeNode | None:
    """Walk up from a chunk to the document node that contains it.

    Prefers an ancestor the tree explicitly labels as the document level; failing that,
    the highest ancestor all of whose chunks still share this chunk's source document.
    Falls back to the chunk's immediate parent, then the chunk itself.

    Args:
      chunk_id: Identifier of the chunk to start from.

    Returns:
      The enclosing document node, or ``None`` when the identifier is unknown.
    """
    source = self.nodes.get(chunk_id)
    if source is None:
      return None
    source_file = source.source_file()
    current = chunk_id
    best: TreeNode | None = None
    while current in self.parent:
      parent_id = self.parent[current]
      parent = self.nodes.get(parent_id)
      if parent is None:
        break
      if parent.is_file():
        return parent
      if source_file:
        chunks = all_chunks(parent)
        if chunks and all(c.source_file() == source_file for c in chunks):
          best = parent
      current = parent_id
    if best is not None:
      return best
    fallback_id = self.parent.get(chunk_id, chunk_id)
    return self.nodes.get(fallback_id) or self.nodes.get(chunk_id)

  def enclosing_file(self, node: TreeNode) -> TreeNode:
    """Find the document node that contains a chunk, a section, or a document itself.

    Never returns a folder: sweeping a folder would drag in other documents, which is the
    teleport's job rather than the sweep's.

    Args:
      node: The node whose enclosing document is wanted.

    Returns:
      The enclosing document node, or ``node`` itself when it has no document ancestor.
    """
    if node.is_file():
      return node
    candidate = self.file_of(node.node_id)
    if (
      candidate is None
      or candidate.node_type == "folder"
      or candidate.node_id == node.node_id
    ):
      return node
    return candidate

  def single_source(self, node: TreeNode) -> bool:
    """Report whether every chunk under a node comes from one source document.

    This is the ceiling for granularity escalation: the climb may reach the document node
    but must never enter a folder that mixes documents.

    Args:
      node: The node to test.

    Returns:
      True when the subtree draws on at most one source document.
    """
    sources = {c.source_file() for c in all_chunks(node) if c.source_file()}
    return len(sources) <= 1

  # ---- lexical index ------------------------------------------------------

  def lexical_index(self) -> LexicalIndex:
    """Build, or return, the cached lexical body index.

    One pass over every chunk records the body tokens of each source document and the
    corpus document frequency of each token. It uses no LLM and no embeddings.

    Returns:
      The lexical index for this tree.
    """
    if self.lexical is not None:
      return self.lexical
    source_tokens: dict[str, set[str]] = {}
    source_chunk: dict[str, TreeNode] = {}
    for chunk in all_chunks(self.root):
      source = chunk.source_file() or chunk.path
      if not source:
        continue
      if source not in source_tokens:
        source_tokens[source] = set()
        source_chunk[source] = chunk
      source_tokens[source].update(tokens(chunk.content or chunk.summary))
    frequency: Counter[str] = Counter()
    for token_set in source_tokens.values():
      for token in token_set:
        frequency[token] += 1
    source_node: dict[str, TreeNode] = {}
    for source, chunk in source_chunk.items():
      source_node[source] = self.enclosing_file(chunk)
    self.lexical = LexicalIndex(
      source_tokens=source_tokens,
      source_node=source_node,
      document_frequency=frequency,
      n_documents=max(1, len(source_tokens)),
    )
    return self.lexical

  def lexical_seed_files(
    self, query: str, config: TreeRAGConfig
  ) -> list[tuple[TreeNode, float, str]]:
    """Find documents whose body text matches the question's rare, distinctive terms.

    Top-down routing only ever sees names, summaries and child-name lists, so it is blind
    to a distinctive term that appears only in a document's body. Seeding those documents
    onto the frontier restores a content-grounded recall signal without embeddings. It is
    purely additive: the agent must still read and confirm every seed.

    Args:
      query: The question being answered.
      config: The configuration supplying the seed count, cap and rarity threshold.

    Returns:
      Up to ``config.lex_seed_k`` triples of document node, seed score and reason. Empty
      when the question has no distinctive term or nothing matches, so generic questions
      behave exactly as they would without seeding.
    """
    index = self.lexical_index()
    n_docs = index.n_documents
    frequency = index.document_frequency
    idf = {
      token: math.log(1.0 + n_docs / frequency[token])
      for token in set(tokens(query))
      if frequency.get(token)
    }
    total = sum(idf.values())
    if total <= 0:
      return []
    rare_cut = max(1, int(n_docs * config.lex_rare_df_frac))
    scored: list[tuple[TreeNode, float]] = []
    for source, token_set in index.source_tokens.items():
      node = index.source_node.get(source)
      if node is None:
        continue
      hit = set(idf) & token_set
      if not hit or not any(frequency[t] <= rare_cut for t in hit):
        continue
      coverage = sum(idf[t] for t in hit) / total
      scored.append((node, coverage))
    if not scored:
      return []
    scored.sort(key=lambda item: item[1], reverse=True)
    out: list[tuple[TreeNode, float, str]] = []
    for node, coverage in scored[: config.lex_seed_k]:
      score = min(
        config.lex_seed_cap,
        config.noise_floor + coverage * (config.lex_seed_cap - config.noise_floor),
      )
      out.append((node, score, "lexical body match on distinctive term(s)"))
    return out


def all_chunks(node: TreeNode) -> list[TreeNode]:
  """Collect every leaf chunk beneath a node, in document order.

  Args:
    node: The subtree root; a leaf returns itself.

  Returns:
    The subtree's chunks in document order.
  """
  out: list[TreeNode] = []
  stack: list[TreeNode] = [node]
  while stack:
    current = stack.pop()
    if current.is_leaf():
      out.append(current)
    else:
      stack.extend(reversed(current.children))
  return out


def whole_unit(node: TreeNode) -> TreeNode:
  """Collapse a whole document or section into one evidence item carrying its full text.

  Args:
    node: The document or section to collapse.

  Returns:
    A copy of the node whose ``content`` is its assembled body text and whose metadata
    carries the resolved ``source_file``.
  """
  chunks = all_chunks(node)
  body = full("\n\n".join((c.content or c.summary or "") for c in chunks))
  source = (
    next((c.source_file() for c in chunks if c.source_file()), None)
    or node.path
    or node.name
  )
  metadata = dict(node.metadata)
  metadata["source_file"] = source
  return replace(node, content=body, metadata=dict(metadata))  # type: ignore[arg-type]


def source_of(node: TreeNode) -> str:
  """Determine a node's canonical source-document identity.

  Two nodes with the same result belong to the same underlying file. This is the same
  rule :func:`whole_unit` uses to name a document.

  Args:
    node: The node to identify.

  Returns:
    The source document path, falling back to the node's path and then its name.
  """
  return (
    next((c.source_file() for c in all_chunks(node) if c.source_file()), None)
    or node.path
    or node.name
  )


def build_index(root: TreeNode, stats: TreeStats) -> TreeIndex:
  """Index a tree's node and parent pointers.

  Args:
    root: The tree's root node.
    stats: Measured load statistics to attach.

  Returns:
    The indexed tree.
  """
  nodes: dict[str, TreeNode] = {}
  parent: dict[str, str] = {}
  stack: list[tuple[TreeNode, str | None]] = [(root, None)]
  while stack:
    node, parent_id = stack.pop()
    if not str(node.node_id).strip():
      raise MalformedTreeError("corpus tree contains an empty node_id")
    if node.node_id in nodes:
      raise MalformedTreeError(f"corpus tree contains duplicate node_id {node.node_id!r}")
    nodes[node.node_id] = node
    if parent_id is not None:
      parent[node.node_id] = parent_id
    for child in node.children:
      stack.append((child, node.node_id))
  return TreeIndex(root=root, nodes=nodes, parent=parent, stats=stats)


def load_tree(path: Path) -> TreeIndex:
  """Load and index a corpus tree from disk, measuring the cost.

  Args:
    path: Path of the corpus tree JSON.

  Returns:
    The loaded, indexed tree.

  Raises:
    TreeUnavailableError: If the file does not exist or cannot be read.
    MalformedTreeError: If the file is not valid JSON, or does not match the node schema.
  """
  if not path.exists():
    raise TreeUnavailableError(
      f"corpus tree not found at {path}; TreeRAG cannot run without it"
    )
  rss_before = _max_rss_mb()
  file_mb = path.stat().st_size / (1024 * 1024)

  start = time.perf_counter()
  try:
    raw_text = path.read_text(encoding="utf-8")
  except OSError as exc:
    raise TreeUnavailableError(f"corpus tree at {path} could not be read: {exc}") from exc
  try:
    decoded = json.loads(raw_text)
  except json.JSONDecodeError as exc:
    raise MalformedTreeError(
      f"corpus tree at {path} is not valid JSON (line {exc.lineno}, column {exc.colno})"
    ) from exc
  parsed = time.perf_counter()
  del raw_text

  root = node_from_dict(decoded)
  built = time.perf_counter()
  del decoded

  nodes: dict[str, TreeNode] = {}
  parent: dict[str, str] = {}
  stack: list[tuple[TreeNode, str | None]] = [(root, None)]
  documents = 0
  chunks = 0
  while stack:
    node, parent_id = stack.pop()
    if not str(node.node_id).strip():
      raise MalformedTreeError("corpus tree contains an empty node_id")
    if node.node_id in nodes:
      raise MalformedTreeError(f"corpus tree contains duplicate node_id {node.node_id!r}")
    nodes[node.node_id] = node
    if parent_id is not None:
      parent[node.node_id] = parent_id
    if node.is_file():
      documents += 1
    elif node.is_leaf():
      chunks += 1
    for child in node.children:
      stack.append((child, node.node_id))
  indexed = time.perf_counter()

  peak = _max_rss_mb()
  stats = TreeStats(
    path=path,
    file_mb=file_mb,
    parse_seconds=parsed - start,
    build_seconds=built - parsed,
    index_seconds=indexed - built,
    total_seconds=indexed - start,
    peak_rss_mb=peak,
    rss_delta_mb=max(0.0, peak - rss_before),
    nodes=len(nodes),
    documents=documents,
    chunks=chunks,
    top_level=len(root.children),
  )
  return TreeIndex(root=root, nodes=nodes, parent=parent, stats=stats)


@dataclass(slots=True)
class _TreeSingleton:
  """Process-wide holder for the loaded tree.

  Attributes:
    index: The loaded tree, once loaded.
    path: The path it was loaded from.
    error: The failure message, when loading was attempted and failed.
    lock: Guards lazy loading.
  """

  index: TreeIndex | None = None
  path: Path | None = None
  error: str = ""
  lock: threading.Lock = field(default_factory=threading.Lock)


_SINGLETON = _TreeSingleton()


def get_tree(config: TreeRAGConfig) -> TreeIndex:
  """Return the process-wide corpus tree, loading it on first use.

  Args:
    config: The configuration supplying the tree path.

  Returns:
    The loaded, indexed tree.

  Raises:
    TreeUnavailableError: If the tree is missing or unreadable.
    MalformedTreeError: If the tree does not match the node schema.
  """
  with _SINGLETON.lock:
    if _SINGLETON.index is not None and _SINGLETON.path == config.tree_path:
      return _SINGLETON.index
    index = load_tree(config.tree_path)
    _SINGLETON.index = index
    _SINGLETON.path = config.tree_path
    _SINGLETON.error = ""
    logger.info("TreeRAG: {}", index.stats.summary())
    return index


def preload_tree(config: TreeRAGConfig) -> TreeStats | None:
  """Load the tree at application startup, without letting a failure crash the app.

  Vector search must keep working when TreeRAG cannot start, so a missing or malformed
  tree is logged and recorded rather than raised.

  Args:
    config: The configuration supplying the tree path.

  Returns:
    The load statistics, or ``None`` when the tree could not be loaded.
  """
  try:
    return get_tree(config).stats
  except TreeUnavailableError as exc:
    with _SINGLETON.lock:
      _SINGLETON.error = str(exc)
    logger.warning("TreeRAG disabled: {}", exc)
    return None


def tree_error() -> str:
  """Report why the tree failed to load, if it did.

  Returns:
    The recorded failure message, or the empty string when there was none.
  """
  with _SINGLETON.lock:
    return _SINGLETON.error


def reset_tree_cache() -> None:
  """Drop the cached tree so the next call reloads it.

  Used by the tests, and by nothing on the serving path: the application never reloads or
  rebuilds the tree on its own.
  """
  with _SINGLETON.lock:
    _SINGLETON.index = None
    _SINGLETON.path = None
    _SINGLETON.error = ""
