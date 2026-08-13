"""
TreeRAG - bounded interactive reading over a governed document hierarchy
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

The canonical tree-routing policy uses structural and lexical information rather than
query-time vector similarity. Search maintains bounded active and reserve alternatives,
supports local descent, same-file sweep and directed teleportation, and stops through
explicit evidence, visit, file and successful context-call budgets. Wall-clock stopping
is cooperative and does not preempt an in-flight model request.

Tree loading and query execution are read-only. Hierarchy construction is an explicit,
separate operation provided by ``scripts/build_tree.py``; it is never triggered by a
search request.
"""

from treerag.client import Counters, HealthStatus, OllamaClient, get_client
from treerag.config import TreeRAGConfig, TreeRAGMode, treerag_enabled
from treerag.errors import (
  MalformedTreeError,
  OllamaUnavailableError,
  SearchBudgetError,
  TreeRAGError,
  TreeUnavailableError,
)
from treerag.events import TraceEvent, render_event
from treerag.search import (
  EvidenceItem,
  SearchTick,
  TreeRAGResult,
  health_check,
  treerag_search,
  treerag_search_stream,
)
from treerag.tree import (
  TreeIndex,
  TreeStats,
  get_tree,
  preload_tree,
  tree_error,
)
from treerag.types import TreeNode

__all__ = [
  "Counters",
  "EvidenceItem",
  "HealthStatus",
  "MalformedTreeError",
  "SearchBudgetError",
  "SearchTick",
  "OllamaClient",
  "OllamaUnavailableError",
  "TraceEvent",
  "TreeIndex",
  "TreeNode",
  "TreeRAGConfig",
  "TreeRAGError",
  "TreeRAGMode",
  "TreeRAGResult",
  "TreeStats",
  "TreeUnavailableError",
  "get_client",
  "get_tree",
  "health_check",
  "preload_tree",
  "render_event",
  "tree_error",
  "treerag_enabled",
  "treerag_search",
  "treerag_search_stream",
]
