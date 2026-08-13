#!/usr/bin/env python3
"""Run one TreeRAG query with truthful live progress telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from treerag import (
  SearchTick,
  TreeRAGConfig,
  TreeRAGMode,
  TreeRAGResult,
  treerag_search_stream,
)


def _duration(seconds: float) -> str:
  """Render a compact non-negative duration."""
  total = max(0, int(seconds))
  hours, remainder = divmod(total, 3600)
  minutes, secs = divmod(remainder, 60)
  if hours:
    return f"{hours:d}:{minutes:02d}:{secs:02d}"
  return f"{minutes:02d}:{secs:02d}"


class ProgressDisplay:
  """Render observed work and the configured wall-clock budget without guessing."""

  def __init__(self, config: TreeRAGConfig) -> None:
    self.config = config
    self.started = time.perf_counter()
    self.calls = 0
    self.step = 0
    self.stage = "starting"
    self.last_call_at = self.started
    self.last_plain_update = 0.0
    self.interactive = sys.stderr.isatty()

  def observe(self, item: object) -> None:
    """Update observed counters from one stream item and redraw."""
    now = time.perf_counter()
    if isinstance(item, SearchTick):
      if item.llm_calls > self.calls:
        self.calls = item.llm_calls
        self.last_call_at = now
    else:
      event = getattr(item, "event", "working")
      self.stage = str(event).replace("_", " ")
      event_step = getattr(item, "step", None)
      if isinstance(event_step, int):
        self.step = max(self.step, event_step)
    self.draw(now=now)

  def draw(self, *, now: float | None = None, finished: bool = False) -> None:
    """Draw one terminal line, or periodic plain lines when stderr is redirected."""
    current = now if now is not None else time.perf_counter()
    elapsed = current - self.started
    budget = self.config.time_budget_s + self.config.answer_budget_s
    fraction = 1.0 if finished else min(elapsed / budget, 0.99)
    width = 24
    filled = width if finished else min(width - 1, int(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    call_wait = current - self.last_call_at
    if finished:
      eta = "done"
      stage = "complete"
    elif elapsed < budget:
      eta = f"<= {_duration(budget - elapsed)} + in-flight call"
      stage = self.stage
    else:
      eta = "waiting for in-flight call"
      stage = self.stage
    line = (
      f"TreeRAG [{bar}] {fraction * 100:5.1f}% budget | "
      f"step {self.step}/{self.config.max_steps} | calls {self.calls} | "
      f"current wait {_duration(call_wait)} | elapsed {_duration(elapsed)} | "
      f"ETA {eta} | {stage}"
    )
    if self.interactive:
      print(f"\r\033[2K{line}", end="\n" if finished else "", file=sys.stderr, flush=True)
      return
    if finished or current - self.last_plain_update >= 5.0:
      print(line, file=sys.stderr, flush=True)
      self.last_plain_update = current

  def clear_after_interrupt(self) -> None:
    """Leave the cursor on a clean line after Ctrl-C."""
    if self.interactive:
      print(file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("question")
  parser.add_argument("--tree", required=True, type=Path)
  parser.add_argument("--ollama-url", required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument(
    "--mode",
    choices=[mode.value for mode in TreeRAGMode],
    default=TreeRAGMode.THOROUGH.value,
  )
  return parser


def main() -> int:
  """Run the streaming search and emit the unchanged JSON result on stdout."""
  args = _parser().parse_args()
  mode = TreeRAGMode(args.mode)
  config = replace(
    TreeRAGConfig.from_env().with_mode(mode),
    tree_path=args.tree,
    ollama_url=args.ollama_url,
    model=args.model,
  )
  progress = ProgressDisplay(config)
  result: TreeRAGResult | None = None
  try:
    for item in treerag_search_stream(
      args.question,
      config,
      tick_seconds=1.0,
    ):
      if isinstance(item, TreeRAGResult):
        result = item
      else:
        progress.observe(item)
  except KeyboardInterrupt:
    progress.clear_after_interrupt()
    print("TreeRAG query interrupted.", file=sys.stderr)
    return 130

  if result is None:
    progress.clear_after_interrupt()
    print("TreeRAG ended without a result.", file=sys.stderr)
    return 1
  progress.calls = result.llm_calls
  progress.step = result.steps
  progress.draw(finished=True)
  print(json.dumps(asdict(result), indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
