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

The long prompt texts, kept out of the control flow that uses them.

The answer instructions are FetchQuest's own, word for word, with two deliberate
deviations: the system prompt drops the description of the search and text-retrieval
tools, which TreeQuest does not expose (and which sent the model into unbounded
deliberation), and ``doc_id`` here is a source path rather than an integer index. The
one-cite-per-document sentence is TreeQuest's; the citation dedupe depends on it.
"""

from datetime import datetime

from treequest.prompting import DOMAIN, DUNNO, ORGANIZATION

#: Prefix marking a steering DIRECTIVE in working memory rather than a gathered fact. The
#: recency cap must never evict one: on a real run the evidence facts overflow the cap
#: several times over, and an evicted directive silently stops steering every decision.
PIN = "!! "

#: Tag identifying the residual-steering note, so it clears only its own line.
RESIDUAL_TAG = PIN + "FOCUS THE SEARCH."

#: Tag identifying the contrast-check directive, so it clears only its own line.
CONTRAST_TAG = PIN + "CONTRAST CHECK."


def system_prompt() -> str:
  """Build the agent's system prompt, stamped with today's date.

  Returns:
    The system prompt. It states the same role as FetchQuest's own, without the tool
    descriptions TreeQuest has no tools to back.
  """
  return (
    "You are a skilled information retrieval agent for "
    f"a {DOMAIN} document library maintained by {ORGANIZATION}.\n"
    "Your task is to research answers to user questions using the provided documents.\n"
    f"Today's date is {datetime.today().strftime('%Y-%m-%d')}. "
    "Use only information grounded in the indexed corpus."
  )


ANSWER_INSTRUCTIONS = (
  "Answer user questions ONLY using the information in the files. Do not use any prior "
  "knowledge in your answers. Do not speculate. If the answer cannot be found in the "
  f"files, say '{DUNNO}'\n"
  "If possible, return only exact, direct quotes of the most relevant passages from "
  "the documents followed by a reference to the respective document ID. Format your "
  "reference as [doc_id], where `doc_id` is the document ID.\nCite each document only "
  "ONCE: group all information drawn from the same document together and put a single "
  "reference to that document, formatted as [doc_id], at the END of that information. "
  "Do NOT repeat the same [doc_id] after every sentence.\nKeep your answers to 100 "
  "words or less. If a question requires a more detailed answer which cannot be "
  "expressed in less than one hundred words, refer the user to the relevant primary "
  "source documents instead, citing their document IDs following the above formatting "
  "guideline. For example, if a user asks for a specific SOP, simply provide the name "
  "of the SOP document and cite its document ID.\n"
)

# ---------------------------------------------------------------------------
# working-memory directives (read by the ranker and every evidence judge, never by the
# answer prompt - so these steer NAVIGATION, not the answer)
# ---------------------------------------------------------------------------

DEFINITION_DIRECTIVE = (
  "DEFINITION QUESTION: this asks what a named entity IS / DOES / is FOR. Its answer "
  "is the document whose OWN SUBJECT is that entity -- one that defines, specifies, "
  "governs or overviews it -- NOT a record that merely USES the entity while "
  "transacting other business (an incident, deviation, change request, meeting minute, "
  "ticket or completed form). A record describes only the facet its own business "
  "turned on, so it yields a true but narrow statement that reads like a definition "
  "and is not one. Two consequences for scoring: (1) a name repeating the entity means "
  "the document MENTIONS it, not that it is ABOUT it -- a title built around an event, "
  "a date, a form type or a problem is a RECORD of something that involved the entity: "
  "score it LOW; a title built around the entity itself, or around the process that "
  "specifies it, is the definition: score it HIGH. (2) The entity's name being ABSENT "
  "from a branch's listed names is NOT evidence that the entity is undocumented there "
  "-- governing documents are titled after their own process, so the one that defines "
  "the entity often never names it, and its whole branch looks silent. Do NOT floor a "
  "branch merely because nothing in it repeats the entity's name; score every branch "
  "on whether the KIND of documentation it holds would DEFINE such an entity, and "
  "prefer that over a branch that merely mentions the entity."
)

POLARITY_DIRECTIVE = (
  "This question asks whether a relationship holds ALWAYS / NECESSARILY / in every "
  "case, OR whether a specific item EXISTS / is available. The answer is in a "
  "GOVERNING document — the DEFINITION, PURPOSE or CONDITIONS of the process, or the "
  "INVENTORY / REGISTER / SCOPE / ITEM-LIST that would ENUMERATE the item's category — "
  "and is frequently NEGATIVE ('no, not always' / 'no, we do not have one'). Prefer "
  "such a governing/enumerating document and treat it as answer-bearing EVEN WHEN it "
  "does not name the specific item: an inventory or scope of the relevant category "
  "that omits the item is exactly what establishes a negative answer. Individual "
  "example records, or documents that merely happen not to mention the item, are NOT "
  "the answer — score them LOW; prefer the defining/enumerating document."
)

RECENCY_DIRECTIVE = (
  "This question asks for the CURRENT / LATEST / MOST RECENT state of something the "
  "corpus tracks over time (a version, revision, edition, status or effective value). "
  "Such values live in a SERIES -- a change log, version history, revision table or "
  "amendment list -- where each entry was current only until a later one superseded "
  "it. Do NOT stop at the first version-bearing entry you find: the answer is the MOST "
  "RECENT / highest entry. Prefer the document that RECORDS THE SERIES, read ALL of "
  "its entries before concluding, and treat any single mid-series value as provisional "
  "until you have confirmed no later entry exists."
)


def subject_directive(subject: str) -> str:
  """Build the pinned working-memory note anchoring the search on one named subject.

  Without it, every decision keys on the question's ATTRIBUTE words alone, and on a corpus
  recording the same attribute for many same-kind entities the search commits to whichever
  sibling states the attribute most prominently.

  Args:
    subject: The one specific thing the question asks about.

  Returns:
    The pinned directive text.
  """
  return PIN + (
    "SPECIFIC SUBJECT. The question asks about ONE particular thing: "
    + subject
    + " — that exact one, not its kind in general. The corpus very likely records the "
    "same KIND of fact for OTHER same-kind entities (other units, instruments, models, "
    "sites, versions) — those are DECOYS: the identical attribute of a DIFFERENT "
    "entity is not the answer, however exactly its wording matches the question. Score "
    "HIGH branches and documents identifiably about this subject (its name, model, "
    "location, or the system it serves); score LOW material stating the asked "
    "attribute for something else; and treat a value that cannot be tied to this "
    "subject as NOT settling the question."
  )


def residual_directive(parts: str) -> str:
  """Build the pinned note re-aiming the search at what the gate says is still open.

  Args:
    parts: The rendered list of unsettled parts.

  Returns:
    The pinned directive text.
  """
  return PIN + (
    "FOCUS THE SEARCH. Parts of the question the evidence does NOT yet settle: "
    + parts
    + ". The question's other parts are already settled by the facts below, so "
    "re-confirming them adds nothing. Score a candidate HIGH only if it plausibly "
    "settles one of the parts listed above; a candidate that only covers ground "
    "already settled scores LOW however topical it looks."
  )


def contrast_directive(sources: str) -> str:
  """Build the pinned note re-aiming ranking at the named subject during a contrast check.

  Without it the check is undermined the moment it starts: working memory is full of the
  CONTENT of the account already taken, so descending into the contrast target re-ranks
  its children against that account and walks straight back to more of the same.

  Args:
    sources: The rendered list of documents the held account came from.

  Returns:
    The pinned directive text.
  """
  return CONTRAST_TAG + (
    " The facts below already give ONE account of the answer, drawn from: "
    + sources
    + ". You are NOT gathering more of it — you are checking whether a DIFFERENT "
    "document states the asked fact for the exact subject the question names, more "
    "precisely or differently. So score a candidate HIGH only if it would be ABOUT "
    "that subject and could state that fact directly. Score LOW any candidate that is "
    "another record, form, template, instance or restatement of the account already "
    "held, however topical it looks — re-reading the same account cannot change it."
  )


# ---------------------------------------------------------------------------
# sufficiency-gate criteria
# ---------------------------------------------------------------------------

POLARITY_CRITERIA = (
  "POLARITY / UNIVERSAL / EXISTENCE QUESTION: this asks whether a relationship holds "
  "ALWAYS / NECESSARILY / in every case, whether something is required or automatic, "
  "OR whether a specific item EXISTS / is available. It is answered by the DEFINITION, "
  "PURPOSE or SCOPE of the entities involved and the CONDITIONS under which the "
  "relationship holds — a governing policy/procedure statement — or by the INVENTORY / "
  "REGISTER / SCOPE / ITEM-LIST that would ENUMERATE the item's category; and the "
  "correct answer is frequently NEGATIVE. Evidence that the subject serves a broader "
  "or different purpose, that the outcome arises only under specific conditions, OR a "
  "governing list/inventory/scope of the relevant category that does NOT include the "
  "item asked about, IS sufficient: it lets you answer (typically 'no'). Do NOT "
  "require an explicit sentence asserting the universal rule or the item's absence; a "
  "definitional, conditional, or enumerating basis for answering yes OR no is enough. "
  "But a set of individual EXAMPLES where the relationship happened to hold, or "
  "documents that merely fail to mention the item without being the governing "
  "list/definition for its category, do NOT by themselves settle the question: if all "
  "you have is that, answer insufficient so the search reaches the "
  "defining/enumerating document.\n"
)

STRICT_CRITERIA = (
  "STRICT MATCH REQUIRED: the evidence must state the SPECIFIC fact the question asks "
  "about, for the process/entity named. A fact about a DIFFERENT process does NOT "
  "count — e.g. if asked how often X assessments occur, a statement that a different "
  "process runs on some schedule is NOT sufficient. When the evidence merely gestures "
  "at the topic, answer insufficient — the search will continue to better "
  "candidates.\nFORM OF THE ANSWER — judge sufficiency against the question AS ASKED, "
  "not against a stricter form of answer you anticipated (an explicit rule sentence, a "
  "timing statement, a number). A question asking WHEN / WHY / HOW / UNDER WHAT "
  "CIRCUMSTANCES something happens is settled by a stated trigger, condition, or "
  "initiating process for the named subject ('done as part of Y', 'requires initiating "
  "Z'), or by a record documenting an actual instance of the event and its cause, even "
  "when no sentence restates that as an explicit rule. Do NOT answer insufficient "
  "merely because the answer is stated as a process rather than a rule.\nANSWER-SENSE "
  "MATCH — beyond the subject being right, the evidence must answer the particular "
  "SENSE the question asks: the specific quantity, measure, property, or requirement "
  "named — not merely one that shares the question's surface vocabulary. Corpora "
  "routinely state several DISTINCT facts using the SAME word, so a confidently-stated "
  "value that reuses that word can denote a DIFFERENT thing than the one asked (e.g. a "
  "count of items handled together vs. the material required per item; a schedule vs. "
  "a duration; a limit vs. a target). Such a near-homonym does NOT settle the "
  "question, however exactly its wording matches and however directly it appears to "
  "give 'a number'. When the only evidence you hold matches the question's words but "
  "plausibly answers a DIFFERENT sense than the one asked, answer insufficient and "
  "name the asked sense as missing, so the search continues to the document that "
  "addresses it directly.\nSUBJECT IDENTITY & CONFLICTING VALUES — when the question "
  "asks for an attribute of one specific named subject, check WHOSE attribute each "
  "piece of evidence states: corpora routinely record the same attribute for MANY "
  "same-kind entities (several units, instruments, models, sites, versions), and a "
  "value stated for a DIFFERENT entity of the same kind does NOT settle the question, "
  "however exactly it matches the asked attribute. If the evidence states SEVERAL "
  "DIFFERENT values for the very fact the question asks — the signature of values "
  "drawn from different entities or occasions — do NOT treat them as corroboration, "
  "and do NOT pick or average among them: that is sufficient ONLY if one of the values "
  "is tied by the evidence itself to the subject the question names; otherwise answer "
  "insufficient and name as missing the value stated FOR that subject (the document "
  "identifiably about it). Nor is AGREEMENT proof of identity: several pieces stating "
  "the SAME value corroborate each other only if at least one of them is identifiably "
  "about (or plausibly governs) the named subject — same-kind documents about sibling "
  "entities routinely repeat the same figure, so repetition alone adds nothing. A "
  "single consistent value from a document plausibly governing the named subject "
  "remains sufficient — do not manufacture a gap when nothing in the evidence points "
  "to a sibling entity.\nSCOPE INHERITANCE — do not manufacture a gap: a rule stated "
  "by the programme, policy or procedure that GOVERNS the thing being asked about DOES "
  "settle the question, even when it never repeats that thing's name. A governing "
  "document states its rule once; it does not restate it for each subject it covers. "
  "So if the evidence states the asked fact at the level that governs the subject, and "
  "nothing in the evidence places the subject outside that scope, that is SUFFICIENT — "
  "do NOT answer insufficient merely because the exact name is absent. Answer "
  "insufficient only when the stated rule's scope plainly does not reach the "
  "subject.\nSUPERSESSION — if two pieces of evidence DISAGREE, do not believe "
  "whichever you read last, nor whichever merely happens to name the subject. A "
  "statement that the general rule does not apply overrides that rule ONLY when it is "
  "unmistakably about the SAME subject the question asks about and actually says so; a "
  "passage that just mentions the subject nearby does not overturn a direct statement "
  "of the asked fact. Say in your reasoning which one governs.\nCOMPLETENESS REQUIRED: "
  "'sufficient' means the evidence answers the question FULLY, not partially. If the "
  "question asks what a process consists of — its components, elements, steps, "
  "requirements, or criteria — then evidence covering only SOME of them is NOT "
  "sufficient. Two specific traps:\n  - Evidence that assigns RESPONSIBILITIES (who "
  "performs or approves something) does not state what the process actually involves. "
  "If the question asks what is done and the evidence only says who does it, answer "
  "insufficient.\n  - Evidence drawn from a single section of a document that clearly "
  "has other relevant sections (a Procedure, its numbered steps, a Records or "
  "Reporting section) is usually partial. If the answer plausibly continues in a "
  "section you have not been given, answer insufficient.\nOnly say sufficient when you "
  "could write a COMPLETE answer naming every part the question asks for. A partial "
  "answer that omits one required element is INSUFFICIENT.\n"
)

PROVENANCE_CRITERIA = (
  "PROVENANCE -- WHOSE DESCRIPTION IS THIS? The question asks what a named entity IS / "
  "DOES / is FOR, so also judge WHERE each description comes from:\n  - AUTHORITATIVE: "
  "the document's own subject IS the entity, or a document that governs, specifies or "
  "overviews it describes it (an SOP, manual, specification, design or overview "
  "document). Even a brief purpose or scope sentence there SETTLES the question -- say "
  "sufficient. Do not demand more once you have this.\n  - INCIDENTAL: the document's "
  "own business is something ELSE (an incident, deviation, change request, meeting "
  "minute, ticket or completed form) and it describes the entity only in passing, in "
  "whatever terms mattered to THAT business. Such a description is real but PARTIAL BY "
  "CONSTRUCTION -- it names the facet that record turned on, not what the entity is "
  "for -- and several such records agreeing does NOT fix this, since they are all "
  "downstream of the same narrow use. If EVERY description you hold is incidental, "
  "answer INSUFFICIENT so the search reaches the document that is actually about the "
  "entity.\n"
  "Report which case you are in as \"basis\": 'subject' if any description comes from a "
  "document that governs or is about the entity, 'incidental' if all of them are "
  "passing mentions inside documents about other business, 'none' if nothing describes "
  "it.\n"
)

RECENCY_CRITERIA = (
  "RECENCY / LATEST-STATE -- the question asks for the CURRENT / LATEST / MOST RECENT "
  "/ in-effect value of something the corpus tracks OVER TIME (a version, revision, "
  "edition, status, or effective figure). Such values live in documents that record a "
  "SERIES of them -- change logs, version histories, revision tables, amendment lists "
  "-- where each entry was current only until a later one superseded it. A value you "
  "have read is the answer ONLY when the evidence itself establishes it is the MOST "
  "RECENT: it is explicitly marked current / effective / in-force, or it is the "
  "newest-dated or highest-numbered entry AND nothing indicates a later or higher "
  "entry that you have not read (including unread sections of the same document). If "
  "the value you hold is one entry drawn from such a series and later or other entries "
  "exist or plausibly remain unread, do NOT assume the one you happened to read is the "
  "current one: answer insufficient and name as missing 'confirm the most recent entry "
  "across the full series'. When the value you hold is a single value NOT part of any "
  "time-series -- nothing indicates other versions/revisions exist -- it stays "
  "sufficient; do not manufacture a gap.\n"
)

UNREAD_SECTIONS_NOTE = (
  "\nThese exist and are unread. If any of them, judging by its heading, plausibly "
  "states a part of the answer that the evidence above does not — the procedural "
  "steps, the records kept, what staff must do and when — then the evidence you have "
  "is INCOMPLETE: answer insufficient so that section gets read.\nThis is a reason to "
  "keep looking ONLY while a part of the question is genuinely unanswered. If the "
  "evidence above already states the fact the question asks for, an unread heading is "
  "not by itself a gap — say sufficient.\n\n"
)
