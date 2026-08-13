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

Question-shape detectors.

Four question shapes need the search steered differently, and each is detected by a
deterministic regex test on the STEM alone - no LLM call, no corpus keywords, so nothing
here is tied to a particular corpus and the detection costs nothing on the hot path.
Each detector is deliberately conservative: a false positive only relaxes a gate or adds a
navigation note, which is cheap, whereas a false negative reinstates the failure mode.
"""

import re
from dataclasses import dataclass

_SYNTH_CATEGORY_PLURAL = (
  r"assays|sops|procedures|documents|policies|instruments|tests|methods|reagents|"
  r"controls|records|forms|templates|systems|registers|kits|panels|workflows|"
  r"processes|standards|certifications|accreditations|guidelines|manuals|worksheets|"
  r"logs|assets|analyzers|platforms|analytes|markers"
)
_SYNTH_CATEGORY_SINGULAR = (
  r"assay|sop|procedure|document|policy|instrument|test|method|reagent|control|"
  r"record|form|template|system|register|kit|panel|workflow|process|standard|"
  r"certification|accreditation|guideline|manual|worksheet|log|asset|analyzer|"
  r"platform|analyte|marker"
)
_SYNTH_SINGLE_SCOPE = re.compile(
  r"\b(steps?|stages?|phases?|parts?|components?|sections?|fields?|elements?)"
  r"\s+(of|in|for|to|within)\b",
  re.I,
)
_SYNTH_PATTERNS: tuple[tuple[str, str], ...] = (
  (
    r"(?:^\s*(?:list|enumerate)\b|\b(?:list|enumerate)\s+(?:all|the|our|every|each|of)\b)",
    "list/enumerate",
  ),
  (r"\ball of (?:our|the|your|oicr'?s)\b", "'all of our/the'"),
  (r"\bevery\s+(?:%s)\b" % _SYNTH_CATEGORY_SINGULAR, "'every <category>'"),
  (
    r"\bwhat are (?:all |our |the )+.*\b(?:%s)\b" % _SYNTH_CATEGORY_PLURAL,
    "'what are all/our <categories>'",
  ),
  (
    r"\bwhat\b.*\b(?:%s)\b.*\b(?:do we|does oicr|are (?:there|available|validated|"
    r"approved|in use|current))\b" % _SYNTH_CATEGORY_PLURAL,
    "'what <categories> do we have / are validated'",
  ),
  (
    r"\bwhich\b.*\b(?:%s)\b.*\b(?:validated|approved|active|available|in use|accredited|"
    r"certified|current)\b" % _SYNTH_CATEGORY_PLURAL,
    "'which <categories> ... validated'",
  ),
  (
    r"\bwhich (?:validated|approved|current|available|accredited|certified)\b.*"
    r"\b(?:%s)\b" % _SYNTH_CATEGORY_PLURAL,
    "'which validated/... <categories>'",
  ),
)
_SYNTH_COMPILED = tuple((re.compile(p, re.I), why) for p, why in _SYNTH_PATTERNS)

_POLARITY_MARKERS = re.compile(
  r"\b(always|automatically|necessarily|invariably|inevitably|guarantee[ds]?|mandatory|"
  r"obligatory|without exception|in (?:all|every) cases?|every ?time|each time|whenever|"
  r"in all instances)\b",
  re.I,
)
_POLARITY_MODAL = re.compile(
  r"^(?:does|do|is|are|can|will|must|should|would|has|have|need)\b.*\b(?:must|required|"
  r"mandatory|have to|need to|necessary)\b",
  re.I,
)
_EXISTENCE_FRAME = re.compile(
  r"^\W*(?:is|are)\s+there\b"
  r"|\bdo(?:es)?\s+(?:we|oicr|our\s+\w+|the\s+lab\w*)\s+"
  r"(?:have|own|possess|maintain|keep|hold)\b"
  r"|\bexists?\b",
  re.I,
)

_DEFN_PATTERNS: tuple[tuple[str, str], ...] = (
  (
    r"^\W*what\s+(?:is|are|was|were)\s+(?:(?:an?|the)\s+)?[^\s?]+(?:\s+[^\s?]+){0,2}\s*\??\s*$",
    "'what is <entity>'",
  ),
  (
    r"^\W*what\s+do(?:es)?\b.{0,40}?\b(?:do|mean|stand for|refer to)\b",
    "'what does <entity> do/mean'",
  ),
  (
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:purpose|role|function|use)\s+of\b",
    "'purpose/role of <entity>'",
  ),
  (r"^\W*(?:describe|define|explain)\b", "'describe/define <entity>'"),
  (r"\bwhat\s+(?:is|are)\b.{0,40}?\bused\s+for\b", "'what is <entity> used for'"),
)
_DEFN_COMPILED = tuple((re.compile(p, re.I), why) for p, why in _DEFN_PATTERNS)

_RECENCY_MARKERS = re.compile(
  r"\b(current(?:ly)?|latest|most\s+recent|newest|up[\s-]?to[\s-]?date|"
  r"in\s+effect|in\s+force|presently|now\s+in\s+use|as\s+of)\b",
  re.I,
)


@dataclass(frozen=True, slots=True)
class ShapeVerdict:
  """The outcome of one question-shape test.

  Attributes:
    matched: Whether the shape was detected.
    why: The pattern that fired, for the trace; empty when nothing matched.
  """

  matched: bool
  why: str


def is_synthesis_question(stem: str) -> ShapeVerdict:
  """Detect a question that asks to enumerate a corpus-wide class of items.

  These want BREADTH, not the default DEPTH: "what are our validated clinical assays"
  is answered by a handful of whole documents, not by drilling one validation report to
  the floor. A scoped "steps of <one procedure>" question is single-document and is
  excluded, and the enumeration must be over a plural corpus-class noun.

  Args:
    stem: The question text.

  Returns:
    The verdict, matched only for genuine enumeration questions.
  """
  text = " " + re.sub(r"\s+", " ", (stem or "").strip().lower()) + " "
  if _SYNTH_SINGLE_SCOPE.search(text):
    return ShapeVerdict(False, "")
  for pattern, why in _SYNTH_COMPILED:
    if pattern.search(text):
      return ShapeVerdict(True, why)
  return ShapeVerdict(False, "")


def is_polarity_question(stem: str) -> ShapeVerdict:
  """Detect a question about whether a relationship holds universally, or a thing exists.

  Its correct answer is frequently NEGATIVE, and when it is, no document positively states
  it: the answer lives in the governing definition, condition or inventory that WOULD
  enumerate the item's category. Left undetected, the ranker floats example records, the
  gate demands a confirmation that cannot exist, and the search grinds to the step cap.

  Args:
    stem: The question text.

  Returns:
    The verdict, naming which frame fired.
  """
  text = re.sub(r"\s+", " ", (stem or "").strip().lower())
  marker = _POLARITY_MARKERS.search(" " + text + " ")
  if marker:
    return ShapeVerdict(True, f"universal/necessity marker '{marker.group(1).strip()}'")
  if _POLARITY_MODAL.search(text):
    return ShapeVerdict(True, "necessity-modal polar frame")
  if _EXISTENCE_FRAME.search(text):
    return ShapeVerdict(True, "existence/availability frame")
  return ShapeVerdict(False, "")


def is_definitional_question(stem: str) -> ShapeVerdict:
  """Detect a question asking what a named entity IS, DOES, or is FOR.

  The trap is that the only branches whose NAMES repeat the entity are the transactional
  records that merely USE it, while the document that DEFINES it is titled after its own
  process and never names the entity at all. The trailing-length bounds keep this to
  genuine name-a-thing stems: "what is <entity>" matches, "what is the retention period
  for records" does not.

  Args:
    stem: The question text.

  Returns:
    The verdict, naming which pattern fired.
  """
  text = re.sub(r"\s+", " ", (stem or "").strip())
  for pattern, why in _DEFN_COMPILED:
    if pattern.search(text):
      return ShapeVerdict(True, why)
  return ShapeVerdict(False, "")


def is_recency_question(stem: str) -> ShapeVerdict:
  """Detect a question asking for the current, latest or most recent state of something.

  Such values live in documents that record a SERIES over time - change logs, version
  histories, revision tables - where every entry was current only until the next replaced
  it. The answer is the most recent entry, not the first version-bearing entry the search
  happens to read.

  Args:
    stem: The question text.

  Returns:
    The verdict, naming the recency marker that fired.
  """
  text = re.sub(r"\s+", " ", (stem or "").strip().lower())
  marker = _RECENCY_MARKERS.search(" " + text + " ")
  if marker:
    return ShapeVerdict(True, f"recency marker '{marker.group(1).strip()}'")
  return ShapeVerdict(False, "")
