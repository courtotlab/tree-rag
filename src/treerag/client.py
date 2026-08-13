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

The Ollama client, with the source benchmark's two unbounded loops replaced.

The notebook this agent came from was built for unattended overnight runs: its readiness
probe looped forever printing "waiting for ollama ... wont stop", and its chat wrapper
caught every exception and retried with exponential backoff without limit. Both are fatal
inside a web request - a dropped SSH tunnel becomes a spinner that never resolves. Here
both are bounded by an attempt count AND a total deadline, and raise
:class:`~treerag.errors.OllamaUnavailableError` on exhaustion.
"""

import threading
import time
from dataclasses import dataclass, field

import ollama
from loguru import logger

from treerag.config import TreeRAGConfig
from treerag.errors import OllamaUnavailableError


@dataclass(slots=True)
class Counters:
  """Token and call tallies accumulated across one search.

  Attributes:
    in_tok: Prompt tokens consumed.
    out_tok: Completion tokens produced.
    calls: Number of completed LLM calls.
  """

  in_tok: int = 0
  out_tok: int = 0
  calls: int = 0


@dataclass(frozen=True, slots=True)
class LlmResponse:
  """One completed chat response, with the metadata answer assembly needs.

  The source notebook stashed this in a module-level ``LLM_LAST`` dict, which is not safe
  when two searches run concurrently. Returning it makes the same information per-call.

  Attributes:
    text: The message content, or the reasoning text when the caller allowed that
      fallback and the content came back empty.
    thinking: The reasoning channel's text.
    done_reason: Why generation stopped; ``length`` means the token budget ran out.
    content_len: Length of the raw message content, before any fallback was applied.
    thinking_len: Length of the reasoning text.
  """

  text: str
  thinking: str
  done_reason: str
  content_len: int
  thinking_len: int

  def reasoning_exhausted(self) -> bool:
    """Report whether the model spent its whole budget reasoning and returned nothing.

    Answer assembly uses this to tell "the model had nothing to say" apart from "the model
    never got to the answer", which are the same empty string otherwise.

    Returns:
      True when the content is empty and either reasoning was produced or generation was
      cut off by the token budget.
    """
    return self.content_len == 0 and (
      self.thinking_len > 0 or self.done_reason == "length"
    )


@dataclass(frozen=True, slots=True)
class HealthStatus:
  """The outcome of one health probe against the Ollama endpoint.

  Attributes:
    ok: True only when the endpoint answered AND the required model is loaded.
    endpoint: The base URL that was probed.
    model: The model that was required.
    detail: A short human-readable explanation, suitable for a UI message.
    checked_at: Monotonic clock reading when the probe completed.
    models_available: Model names the endpoint reported.
  """

  ok: bool
  endpoint: str
  model: str
  detail: str
  checked_at: float
  models_available: tuple[str, ...] = ()

  def is_local_endpoint(self) -> bool:
    """Report whether the configured endpoint is a loopback address.

    A loopback endpoint means the Ollama server is being reached through a forwarded
    port - in practice an SSH tunnel on a developer's machine - so the tunnel is the
    thing to check. On the deployment server Ollama is reached directly and the tunnel
    would be the wrong thing to name.

    Returns:
      True when the endpoint host is localhost or a loopback address.
    """
    host = self.endpoint.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    return host in ("localhost", "127.0.0.1", "::1", "[::1]")

  def ui_message(self) -> str:
    """Render the status as the message shown next to a disabled TreeRAG toggle.

    The remedy named depends on how the endpoint is reached: a forwarded local port means
    a tunnel to check, whereas a remote endpoint means the Ollama service itself.

    Returns:
      An empty string when healthy, otherwise a one-line explanation of what to check.
    """
    if self.ok:
      return ""
    remedy = (
      "check the SSH tunnel"
      if self.is_local_endpoint()
      else f"check that Ollama is running and reachable at {self.endpoint}"
    )
    return f"TreeRAG unavailable — {self.detail}, {remedy}"


class OllamaClient:
  """A bounded, health-checkable wrapper around the Ollama chat and embeddings API.

  One instance is shared across requests. It is safe to call from several threads: the
  per-call metadata is returned rather than stashed, and the health verdict and embedding
  cache are guarded by a lock.
  """

  def __init__(self, config: TreeRAGConfig) -> None:
    """Create a client for the configured endpoint.

    Args:
      config: The TreeRAG configuration supplying endpoint, model and retry policy.
    """
    self._config = config
    self._client = ollama.Client(host=config.ollama_url, timeout=config.request_timeout_s)
    self._lock = threading.Lock()
    self._health: HealthStatus | None = None
    self._embed_cache: dict[str, list[float]] = {}

  @property
  def config(self) -> TreeRAGConfig:
    """The configuration this client was built from.

    Returns:
      The client's :class:`~treerag.config.TreeRAGConfig`.
    """
    return self._config

  # ---- health -------------------------------------------------------------

  def health_check(self, *, force: bool = False) -> HealthStatus:
    """Probe the endpoint and confirm the required model is loaded.

    The verdict is cached for ``config.health_ttl_s`` so a UI that re-checks on every
    toggle does not hammer the endpoint, but it is never cached forever: the tunnel drops
    in practice and the UI must reflect that.

    Args:
      force: Re-probe even when a fresh cached verdict exists.

    Returns:
      The health status. This never raises: an unreachable endpoint is reported as a
      status with ``ok=False``, because startup must not crash when TreeRAG is down.
    """
    now = time.monotonic()
    with self._lock:
      cached = self._health
      if (
        not force
        and cached is not None
        and now - cached.checked_at < self._config.health_ttl_s
      ):
        return cached

    status = self._probe(now)
    with self._lock:
      self._health = status
    return status

  def _probe(self, started: float) -> HealthStatus:
    """Run one uncached health probe.

    Args:
      started: Monotonic clock reading at the start of the probe.

    Returns:
      The health status derived from the endpoint's model listing.
    """
    endpoint = self._config.ollama_url
    model = self._config.model
    try:
      probe = ollama.Client(host=endpoint, timeout=self._config.health_timeout_s)
      names = tuple(self._model_names(probe.list()))
    except Exception as exc:  # noqa: BLE001 - any transport failure is "unreachable"
      return HealthStatus(
        ok=False,
        endpoint=endpoint,
        model=model,
        detail=f"Ollama endpoint not reachable ({type(exc).__name__})",
        checked_at=started,
      )
    if not any(model in name for name in names):
      return HealthStatus(
        ok=False,
        endpoint=endpoint,
        model=model,
        detail=f"model {model} is not loaded on the endpoint",
        checked_at=started,
        models_available=names,
      )
    return HealthStatus(
      ok=True,
      endpoint=endpoint,
      model=model,
      detail=f"{model} is loaded",
      checked_at=started,
      models_available=names,
    )

  @staticmethod
  def _model_names(listing: ollama.ListResponse) -> list[str]:
    """Extract model names from a list response.

    Args:
      listing: The response from the endpoint's model listing.

    Returns:
      The names of the models the endpoint reports, skipping any unnamed entry.
    """
    return [m.model for m in listing.models if m.model]

  # ---- chat ---------------------------------------------------------------

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
    """Send one prompt and return the response, retrying a bounded number of times.

    ``thinking_fallback`` decides what happens when the model routes everything into its
    reasoning channel and leaves the content empty. Returning the reasoning text is useful
    for JSON-parsing callers but catastrophic for the answer call, where chain-of-thought
    would leak to the user, so answer-producing callers pass ``False``.

    Args:
      prompt: The full user prompt.
      counters: Tally object updated in place with tokens and call count.
      num_predict: Token budget for this call.
      temperature: Sampling temperature.
      think: Override the configured reasoning-channel setting for this call.
      thinking_fallback: Return the reasoning text when the content comes back empty.

    Returns:
      The completed response.

    Raises:
      OllamaUnavailableError: If every attempt failed, or the total retry deadline
        elapsed, before a response was received.
      TypeError: If the installed Ollama client rejects the request for a reason other
        than the ``think`` keyword, which is the one argument this retries without.
    """
    use_think = self._config.thinking if think is None else think
    options: dict[str, float | int] = {
      "temperature": temperature,
      "num_predict": num_predict,
    }
    deadline = time.monotonic() + self._config.retry_deadline_s
    pass_think = True
    last_error = ""

    for attempt in range(1, self._config.max_attempts + 1):
      try:
        if pass_think:
          raw = self._client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=self._config.keep_alive,
            think=use_think,
          )
        else:
          raw = self._client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=self._config.keep_alive,
          )
      except TypeError:
        # An older ollama client without the `think` keyword; retry without it. This does
        # not consume an attempt, because nothing was sent.
        if pass_think:
          pass_think = False
          continue
        raise
      except Exception as exc:  # noqa: BLE001 - transport, protocol and server errors alike
        last_error = f"{type(exc).__name__}: {exc}"
        remaining = deadline - time.monotonic()
        if attempt >= self._config.max_attempts or remaining <= 0:
          break
        backoff = min(
          self._config.retry_backoff_cap_s,
          self._config.retry_backoff_base_s * (2 ** (attempt - 1)),
        )
        backoff = min(backoff, remaining)
        logger.warning(
          "TreeRAG: Ollama call failed (attempt {}/{}), retrying in {:.1f}s: {}",
          attempt,
          self._config.max_attempts,
          backoff,
          type(exc).__name__,
        )
        time.sleep(backoff)
        continue

      if isinstance(raw, ollama.ChatResponse):
        return self._finish(raw, counters, thinking_fallback=thinking_fallback)
      raise OllamaUnavailableError(
        f"Ollama at {self._config.ollama_url} returned a streaming response to a "
        "non-streaming request"
      )

    raise OllamaUnavailableError(
      f"Ollama at {self._config.ollama_url} did not respond after "
      f"{self._config.max_attempts} attempt(s) within "
      f"{self._config.retry_deadline_s:.0f}s: {last_error or 'no response'}"
    )

  @staticmethod
  def _finish(
    raw: ollama.ChatResponse, counters: Counters, *, thinking_fallback: bool
  ) -> LlmResponse:
    """Convert a successful chat response and update the counters.

    Args:
      raw: The response returned by the Ollama client.
      counters: Tally object updated in place.
      thinking_fallback: Return the reasoning text when the content is empty.

    Returns:
      The converted response.
    """
    counters.calls += 1
    counters.in_tok += raw.prompt_eval_count or 0
    counters.out_tok += raw.eval_count or 0
    content = (raw.message.content or "").strip()
    thinking = (raw.message.thinking or "").strip()
    return LlmResponse(
      text=content or (thinking if thinking_fallback else ""),
      thinking=thinking,
      done_reason=raw.done_reason or "",
      content_len=len(content),
      thinking_len=len(thinking),
    )

  # ---- embeddings ---------------------------------------------------------

  def embed(self, text: str) -> list[float] | None:
    """Embed one short string, caching the result.

    Embeddings are used only to order the names listed for a wide folder, so a failure is
    not an error: the caller falls back to lexical ordering. This therefore never raises
    and never retries.

    Args:
      text: The text to embed; empty text is embedded as a single space.

    Returns:
      The embedding vector, or ``None`` when the endpoint could not produce one.
    """
    key = text or " "
    with self._lock:
      cached = self._embed_cache.get(key)
    if cached is not None:
      return cached
    try:
      response = self._client.embeddings(model=self._config.embed_model, prompt=key)
    except Exception:  # noqa: BLE001 - an unavailable embedder degrades, never fails
      return None
    vector = list(response.embedding)
    if not vector:
      return None
    with self._lock:
      self._embed_cache[key] = vector
    return vector


@dataclass(slots=True)
class _SharedClient:
  """Process-wide client holder, so one HTTP pool is shared across requests.

  Attributes:
    client: The shared client, once built.
    lock: Guards lazy construction.
  """

  client: OllamaClient | None = None
  lock: threading.Lock = field(default_factory=threading.Lock)


_SHARED = _SharedClient()


def get_client(config: TreeRAGConfig) -> OllamaClient:
  """Return the process-wide client, building it on first use.

  A client whose configured endpoint or model differs from ``config`` is replaced, so a
  CLI invocation that overrides the endpoint is honoured.

  Args:
    config: The configuration the client should serve.

  Returns:
    The shared client for this configuration.
  """
  with _SHARED.lock:
    current = _SHARED.client
    if (
      current is None
      or current.config.ollama_url != config.ollama_url
      or current.config.model != config.model
    ):
      current = OllamaClient(config)
      _SHARED.client = current
    return current
