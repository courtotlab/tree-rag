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
"""


class TreeRagError(Exception):
  """Base class for every error raised by the TreeQuest subsystem."""


class OllamaUnavailableError(TreeRagError):
  """The Ollama endpoint could not be reached, or the required model is not loaded.

  Raised once the bounded retry policy is exhausted, so a dead endpoint surfaces as a
  clear error instead of a request that hangs forever. The benchmark notebook this agent
  was extracted from retried without bound, which is unusable inside a web request.
  """


class SearchBudgetError(TreeRagError):
  """The search ran past its wall-clock deadline or its LLM-call ceiling.

  Raised only as a backstop. The traversal checks its budget between decisions and stops
  cleanly, answering from the evidence it has; this exception covers the case where a
  single decision overruns the remaining budget outright, and it is caught so the user
  still gets an answer rather than an error.
  """


class TreeUnavailableError(TreeRagError):
  """The corpus tree could not be loaded, or was requested before it was loaded."""


class MalformedTreeError(TreeUnavailableError):
  """The corpus tree JSON does not match the expected node schema."""
