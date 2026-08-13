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

The corpus tree's node schema, typed. Every node in ``corpus_tree.json`` carries the same
eight fields; the per-node-type variation lives entirely in ``metadata``, which is
modelled as a total=False TypedDict with the keys the builder actually emits.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from treequest.errors import MalformedTreeError

#: Node types the tree builder emits. The document level is labelled ``document`` in this
#: corpus but ``file`` in older builds, and both are treated as the document level.
NODE_TYPES = ("root", "folder", "document", "file", "section", "chunk")

#: The node types that represent one whole source document.
FILE_TYPES = ("file", "document")


class NodeMetadata(TypedDict, total=False):
  """Per-node metadata emitted by the tree builder.

  Attributes:
    num_children: Number of direct children; present on folder-like nodes.
    num_sections: Number of sections in a document node.
    file_type: Source file extension, e.g. ``pdf`` or ``docx``.
    source_file: Path of the source document a node's text came from.
    section: Name of the section a node belongs to.
    level: Heading depth of a section node.
    page: Source page number, absent or null for non-paginated formats.
    unit_index: Ordinal of a chunk within its document.
    kind: Chunk kind, e.g. ``text`` or ``image``.
    caption: Caption text of an image chunk.
    image_path: Path of an extracted image asset.
  """

  num_children: int
  num_sections: int
  file_type: str
  source_file: str
  section: str
  level: int
  page: int | None
  unit_index: int
  kind: str
  caption: str
  image_path: str


@dataclass(slots=True)
class TreeNode:
  """One node of the hierarchical corpus tree.

  Attributes:
    node_id: Stable identifier, unique within the tree.
    node_type: One of :data:`NODE_TYPES`.
    name: Human-readable name - a folder name, file name or section heading.
    path: Path of the node within the corpus, where the builder recorded one.
    summary: LLM-generated summary; lossy, which is why the ranker also sees real text.
    content: The node's own text. Populated on chunks; empty on interior nodes.
    children: Direct children, in document order.
    metadata: Builder metadata; see :class:`NodeMetadata`.
  """

  node_id: str
  node_type: str
  name: str
  path: str
  summary: str
  content: str = ""
  children: list["TreeNode"] = field(default_factory=list)
  metadata: NodeMetadata = field(default_factory=lambda: NodeMetadata())

  def is_leaf(self) -> bool:
    """Report whether this node is a chunk, the tree's leaf level.

    Returns:
      True when the node's type is ``chunk``.
    """
    return self.node_type == "chunk"

  def is_file(self) -> bool:
    """Report whether this node is the document level of the tree.

    Corpora label this level either ``document`` or ``file``; both count, because a
    branch gated on one label silently no-ops on a tree that uses the other.

    Returns:
      True when the node's type is one of :data:`FILE_TYPES`.
    """
    return self.node_type in FILE_TYPES

  def source_file(self) -> str | None:
    """Read the source document path recorded on this node, if any.

    Returns:
      The ``source_file`` metadata value, or ``None`` when the node does not carry one.
    """
    return self.metadata.get("source_file")

  def count_leaves(self) -> int:
    """Count the chunk nodes beneath this node.

    Returns:
      The number of leaves in this subtree, counting the node itself when it is a leaf.
    """
    total = 0
    stack: list[TreeNode] = [self]
    while stack:
      node = stack.pop()
      if node.is_leaf():
        total += 1
      else:
        stack.extend(node.children)
    return total


def _require_str(
  raw: dict[str, object], key: str, node_id: str, *, required: bool
) -> str:
  """Read a string field from a raw JSON node.

  Args:
    raw: The decoded JSON object for one node.
    key: The field name to read.
    node_id: The node's identifier, used only in error messages.
    required: Whether a missing field is an error rather than an empty string.

  Returns:
    The field's value as a string, or the empty string when it is absent and optional.

  Raises:
    MalformedTreeError: If the field is required and absent, or is present but is not a
      JSON string.
  """
  if key not in raw:
    if required:
      raise MalformedTreeError(f"tree node {node_id!r} is missing required field {key!r}")
    return ""
  value = raw[key]
  if value is None and not required:
    return ""
  if not isinstance(value, str):
    raise MalformedTreeError(
      f"tree node {node_id!r} field {key!r} must be a string, got {type(value).__name__}"
    )
  return value


def _parse_metadata(raw: object, node_id: str) -> NodeMetadata:
  """Convert a raw JSON metadata object into a :class:`NodeMetadata`.

  Unknown keys are dropped rather than carried as untyped values, so the shape stays
  closed. Known keys are checked against their declared type.

  Args:
    raw: The decoded ``metadata`` value for one node.
    node_id: The node's identifier, used only in error messages.

  Returns:
    The typed metadata for the node; empty when the node carries none.

  Raises:
    MalformedTreeError: If ``metadata`` is present but is not a JSON object, or a known
      key holds a value of the wrong type.
  """
  if raw is None:
    return NodeMetadata()
  if not isinstance(raw, dict):
    raise MalformedTreeError(
      f"tree node {node_id!r} field 'metadata' must be an object, "
      f"got {type(raw).__name__}"
    )
  out = NodeMetadata()
  for key in ("num_children", "num_sections", "level", "unit_index"):
    if key in raw:
      value = raw[key]
      if not isinstance(value, int) or isinstance(value, bool):
        raise MalformedTreeError(
          f"tree node {node_id!r} metadata {key!r} must be an integer"
        )
      out[key] = value
  for key in ("file_type", "source_file", "section", "kind", "caption", "image_path"):
    if key in raw:
      value = raw[key]
      if value is None:
        continue
      if not isinstance(value, str):
        raise MalformedTreeError(
          f"tree node {node_id!r} metadata {key!r} must be a string"
        )
      out[key] = value  # type: ignore[literal-required]
  if "page" in raw:
    page = raw["page"]
    if page is None:
      out["page"] = None
    elif isinstance(page, int) and not isinstance(page, bool):
      out["page"] = page
    else:
      raise MalformedTreeError(
        f"tree node {node_id!r} metadata 'page' must be an integer"
      )
  return out


def node_from_dict(raw: object) -> TreeNode:
  """Build a :class:`TreeNode` from one decoded JSON object, strictly.

  The parse is iterative rather than recursive so a deep corpus cannot exhaust the
  interpreter stack, and it rejects anything that does not match the schema instead of
  guessing at an alternative shape.

  Args:
    raw: The decoded JSON value for the root of a subtree.

  Returns:
    The parsed subtree.

  Raises:
    MalformedTreeError: If any node is not a JSON object, is missing ``node_id`` or
      ``node_type``, declares an unknown ``node_type``, holds a non-list ``children``,
      or carries metadata of the wrong shape.
  """
  if not isinstance(raw, dict):
    raise MalformedTreeError(f"tree root must be a JSON object, got {type(raw).__name__}")

  def shallow(obj: dict[str, object]) -> tuple[TreeNode, list[object]]:
    node_id_value = obj.get("node_id")
    if not isinstance(node_id_value, str) or not node_id_value:
      raise MalformedTreeError("tree node is missing a string 'node_id'")
    node_type = _require_str(obj, "node_type", node_id_value, required=True)
    if node_type not in NODE_TYPES:
      raise MalformedTreeError(
        f"tree node {node_id_value!r} has unknown node_type {node_type!r}; "
        f"expected one of {', '.join(NODE_TYPES)}"
      )
    kids = obj.get("children", [])
    if kids is None:
      kids = []
    if not isinstance(kids, list):
      raise MalformedTreeError(
        f"tree node {node_id_value!r} field 'children' must be a list, "
        f"got {type(kids).__name__}"
      )
    node = TreeNode(
      node_id=node_id_value,
      node_type=node_type,
      name=_require_str(obj, "name", node_id_value, required=False),
      path=_require_str(obj, "path", node_id_value, required=False),
      summary=_require_str(obj, "summary", node_id_value, required=False),
      content=_require_str(obj, "content", node_id_value, required=False),
      metadata=_parse_metadata(obj.get("metadata"), node_id_value),
    )
    return node, kids

  root, root_kids = shallow(raw)
  pending: list[tuple[TreeNode, list[object]]] = [(root, root_kids)]
  while pending:
    parent, raw_kids = pending.pop()
    for raw_kid in raw_kids:
      if not isinstance(raw_kid, dict):
        raise MalformedTreeError(
          f"child of tree node {parent.node_id!r} must be a JSON object, "
          f"got {type(raw_kid).__name__}"
        )
      kid, kid_kids = shallow(raw_kid)
      parent.children.append(kid)
      pending.append((kid, kid_kids))
  return root
