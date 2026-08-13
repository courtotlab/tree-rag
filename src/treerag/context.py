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

The per-search context every helper receives. The source notebook reached module-level
globals for the client, the counters and the tree indices, which cannot serve two
concurrent requests; bundling them here is the only structural change that required.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from treerag.budget import SearchBudget
from treerag.client import Counters, LlmResponse, OllamaClient
from treerag.config import TreeRAGConfig
from treerag.events import TraceEvent
from treerag.tree import TreeIndex

#: A callback invoked with each trace event as it is produced.
EventSink = Callable[[TraceEvent], None]


def _discard(event: TraceEvent) -> None:
  """Drop a trace event.

  The default sink, used by callers that only want the final result.

  Args:
    event: The event to discard.
  """


@dataclass(slots=True)
class SearchContext:
  """Everything one TreeRAG search needs, threaded through the helpers.

  Attributes:
    config: The tuning knobs and endpoint settings.
    client: The bounded Ollama client.
    index: The loaded corpus tree and its structural indices.
    counters: Token and call tallies for this search.
    trace: Every event produced so far, in order.
    sink: Callback invoked with each event as it is produced.
    budget: The wall-clock and call budget. ``None`` disables enforcement, which is what
      the unit tests use; every real search is given one.
  """

  config: TreeRAGConfig
  client: OllamaClient
  index: TreeIndex
  counters: Counters = field(default_factory=Counters)
  trace: list[TraceEvent] = field(default_factory=list)
  sink: EventSink = _discard
  budget: SearchBudget | None = None

  def stop_reason(self) -> str:
    """Report whether navigation should stop now.

    Returns:
      A short reason when the budget says to stop, or the empty string to continue.
    """
    if self.budget is None:
      return ""
    return self.budget.traversal_exhausted()

  def emit(self, event: TraceEvent) -> None:
    """Record a trace event and hand it to the sink.

    Args:
      event: The event to record.
    """
    self.trace.append(event)
    self.sink(event)

  def llm(
    self,
    prompt: str,
    *,
    num_predict: int = 512,
    temperature: float = 0.0,
    think: bool | None = False,
    thinking_fallback: bool = True,
  ) -> str:
    """Send one prompt and return its text.

    Args:
      prompt: The full user prompt.
      num_predict: Token budget for this call.
      temperature: Sampling temperature.
      think: Override the configured reasoning-channel setting for this call.
      thinking_fallback: Return the reasoning text when the content comes back empty.

    Returns:
      The response text.

    Raises:
      OllamaUnavailableError: If the bounded retry policy is exhausted.
    """
    return self.llm_full(
      prompt,
      num_predict=num_predict,
      temperature=temperature,
      think=think,
      thinking_fallback=thinking_fallback,
    ).text

  def llm_full(
    self,
    prompt: str,
    *,
    num_predict: int = 512,
    temperature: float = 0.0,
    think: bool | None = False,
    thinking_fallback: bool = True,
  ) -> LlmResponse:
    """Send one prompt and return the full response, including its metadata.

    Args:
      prompt: The full user prompt.
      num_predict: Token budget for this call.
      temperature: Sampling temperature.
      think: Override the configured reasoning-channel setting for this call.
      thinking_fallback: Return the reasoning text when the content comes back empty.

    Returns:
      The complete response.

    Raises:
      OllamaUnavailableError: If the bounded retry policy is exhausted.
      SearchBudgetError: If the search has already spent even its reserved allowance.
    """
    if self.budget is not None:
      self.budget.check()
    response = self.client.chat(
      prompt,
      self.counters,
      num_predict=num_predict,
      temperature=temperature,
      think=think,
      thinking_fallback=thinking_fallback,
    )
    if self.budget is not None:
      self.budget.note_call()
    return response

  def embed(self, text: str) -> list[float] | None:
    """Embed a short string for name ordering.

    Args:
      text: The text to embed.

    Returns:
      The embedding vector, or ``None`` when the embedder is unavailable.
    """
    return self.client.embed(text)
