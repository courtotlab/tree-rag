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

The wall-clock and call-count budget that keeps a search answerable.

The source benchmark bounded the traversal by MAX_STEPS (node visits) and MAX_FILES
(documents kept). Neither bounds TIME, because the cost of one visit is proportional to
the visited node's fan-out: ranking scores every child in batches, so a node with 2,661
children costs 533 LLM calls for a single decision. On this corpus the median node has 2-4
children and the benchmark averaged ~75 calls per question, but the tail is unbounded - a
question that routes into a wide node can run for hours.

This budget is cooperative in wall time: it is checked between model requests, so it
cannot preempt an in-flight request. Its successful-call limit is exact. Capacity for
writing the answer is reserved *inside* the configured call ceiling rather than added on
top of it.
"""

import time
from dataclasses import dataclass, field

from treequest.errors import SearchBudgetError


@dataclass(slots=True)
class SearchBudget:
  """Tracks what one search has spent and how much it is allowed.

  Attributes:
    traversal_seconds: Wall-clock allowance for navigation.
    answer_seconds: Additional allowance reserved for writing the answer.
    max_calls: Ceiling on LLM calls for the whole search.
    started: Monotonic clock reading at construction.
    calls: LLM calls made so far.
    answering: True once the traversal has finished and the answer is being written.
    stopped_reason: Why the traversal stopped early, empty if it finished naturally.
  """

  traversal_seconds: float
  answer_seconds: float
  max_calls: int
  started: float = field(default_factory=time.monotonic)
  calls: int = 0
  answering: bool = False
  stopped_reason: str = ""

  def elapsed(self) -> float:
    """Report how long the search has been running.

    Returns:
      Seconds since the budget was created.
    """
    return time.monotonic() - self.started

  def remaining(self) -> float:
    """Report how much traversal time is left.

    Returns:
      Seconds remaining before navigation must stop; zero once it is spent.
    """
    return max(0.0, self.traversal_seconds - self.elapsed())

  def traversal_exhausted(self) -> str:
    """Report whether navigation must stop now.

    Checked between decisions so the search can stop cleanly and answer from what it has,
    rather than being interrupted mid-decision.

    Returns:
      A short reason when the traversal must stop, or the empty string to continue.
    """
    if self.elapsed() >= self.traversal_seconds:
      return f"time budget of {self.traversal_seconds:.0f}s reached"
    if self.calls >= self.traversal_call_limit():
      return (
        f"navigation allowance of {self.traversal_call_limit()} successful LLM calls "
        f"reached ({self.max_calls - self.traversal_call_limit()} reserved for answering)"
      )
    return ""

  def traversal_call_limit(self) -> int:
    """Return the successful-call limit for navigation.

    The remainder of ``max_calls`` is reserved for answer construction. The reserve uses
    the historical completion margin, but now sits within the advertised total instead
    of silently extending it.
    """
    reserve = min(max(8, self.max_calls // 10), self.max_calls - 1)
    return self.max_calls - reserve

  def begin_answer(self) -> None:
    """Switch to the answer phase, releasing the reserved allowance.

    Called once the traversal has stopped, so the answer is written even when navigation
    used every second it was given.
    """
    self.answering = True

  def note_call(self) -> None:
    """Record one completed LLM call."""
    self.calls += 1

  def check(self) -> None:
    """Fail before a call that would violate the hard whole-search allowance.

    The traversal allowance is a soft stopping condition handled by
    :meth:`traversal_exhausted`; raising on it here would prevent the controller from
    leaving traversal cleanly and spending the reserved answer allowance.

    Raises:
      SearchBudgetError: If the combined wall-clock deadline has passed or another call
        would exceed the exact whole-search call ceiling.
    """
    elapsed = self.elapsed()
    limit = self.traversal_seconds + self.answer_seconds
    if elapsed >= limit:
      raise SearchBudgetError(
        f"TreeQuest stopped after {elapsed:.0f}s at its cooperative {limit:.0f}s ceiling"
      )
    if self.calls >= self.max_calls:
      raise SearchBudgetError(
        f"TreeQuest stopped after {self.calls} successful LLM calls at its exact "
        f"{self.max_calls}-call ceiling"
      )
