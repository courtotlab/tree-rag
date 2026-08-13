#!/usr/bin/env python
# coding: utf-8

# # Sequential benchmark — greedy-teleport TreeRAG vs. qms_search over the whole question set
# 
# Runs the TreeRAG traversal on every question in `questions.json`, one after another. Per question it prints the correct answer TEXT (not the option letter), the qms_search files read (full paths resolved via the `searches` field), the full TreeRAG traversal (each descend with relevance scores, each teleport with its ranked frontier), then BOTH full answers. It grades both with the same LLM judge and appends to `benchmark_report_6.json`.
# 
# Ranking scores on **real content, not just summaries**: every file/section candidate shows a `text:` excerpt of its actual words alongside its summary, so a section that literally answers the question can't score near-zero just because its summary is lossy or its name uses different words (e.g. an 'Ungraded PT Challenges' section for a question about *failed* challenges); the prompt also tells the ranker to match on meaning and treat domain synonyms as matches. Folders still route on the names they **contain**: every interior candidate is scored with a `contains:` line listing the names of the files/subsections inside it, so a folder holding the exactly-right file ranks like it even when its own summary is generic (the root-level failure where the correct folder sat 4th). Wide folders list names in query-similarity order (cached embeddings) so the relevant name is never clipped out.
# 
# Section `text:` excerpts are **relevance-ranked**: when a section is longer than the preview budget, the chunks shown are those most similar to the question (not the section's opening words), so a Procedure whose answer-bearing sentences sit past its preamble is scored on those sentences. Same-file candidates the sweep does not read are **deferred, not destroyed**, and the sufficiency gate is shown their headings: if it judges the evidence incomplete, the best deferred sections are read before the agent is allowed to teleport away.
# 
# After a take, the sweep finishes the **enclosing file**, not just the taken section: taking a *Responsibilities* section now still triages its sibling *Procedure* sections rather than only its own paragraphs. The sufficiency gate also demands **completeness** — evidence that names who is responsible but not what the procedure involves, or that comes from one section of a document with other relevant sections, is insufficient and the search continues.
# 
# Navigation: **no backtrack**. The agent descends the best-scored child at each level and pushes the runner-up siblings from every level onto one global frontier. On reaching a relevant leaf it makes one **granularity** decision: keep just that passage, or (if the answer depends on the surrounding section) take the parent section — climbing parent→grandparent→…→file while each step still needs the wider context. After every take it finishes the current file before anything else via an **intra-file triage sweep**: one **completeness-oriented** agentic pass over every remaining candidate (heading + inherited descent score + snippet). Within a document already chosen as relevant, sections that scored as high as the taken evidence are presumed part of the same answer, so an enumerated procedure or list spread across consecutive high-scoring sections is collected in full rather than stopping at the first passage; only the selected sections are read, each as a whole unit — a handful of calls even on a 100-chunk file, with nothing dismissed unseen. Only when the sufficiency gate then says the evidence still isn't enough does it **teleport** to the highest-scored node left on the global frontier (which can be a top-level folder, so a wrong first choice is fully recoverable). Bounded by `MAX_FILES` and `MAX_STEPS`. Both answers are graded by a **single joint judge call** for consistency, and each document is cited only once.
# 
# Answer assembly follows a strict budget order \u2014 **elements first, brevity second, quotes last**: when the answer will not fit in 100 words the model is told to shorten or drop quotations, never to drop a required element. A compression pass then trims a bloated repaired answer back toward the house limit, and is rejected if it loses an element. Answer assembly also ends with a **coverage gate**: the drafted answer is checked back against the retrieved evidence for required elements it failed to state (the two-part process where only one part was reported), and one repair call adds them, explicitly allowed to exceed the 100-word house limit because completeness outranks brevity.
# 
# Whenever qms beats TreeRAG, a diagnosis runs: it locates where qms's file sat in our ranking/frontier and proposes one concrete fix (navigation vs. answer-assembly). The final cell aggregates these across all qms-wins so you can implement the most common fixes first. Saved after every question (atomic replace); re-run to resume.
# 

# In[1]:


pass  # notebook pip magic removed; deps supplied by the runner


# In[2]:


import os, json, re, time, math, hashlib, textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter
import ollama
import numpy as np
import pandas as pd

OLLAMA_URL  = "http://localhost:11528"
AGENT_MODEL = "gpt-oss:120b"
JUDGE_MODEL = "gpt-oss:120b"
EMBED_MODEL = "nomic-embed-text"
CACHE_DIR   = Path("tree_cache"); TREE_FILE = CACHE_DIR / "corpus_tree.json"
QUESTIONS_FILE = "questions_public.json"
QMS_ANSWERS_FILE = "qms_answers_public.json"  # public flat-hybrid answers
REPORT_FILE = __import__("os").environ.get("IMPROVE_REPORT_PATH") or "results/treequest_report.json"
DOSSIER_FILE = "results/treequest_failure_dossiers.json"
                                               # outperforms treerag: traversal event sequence with scores,
                                               # warnings, timings, lengths — no LLM diagnosis (the in-loop
                                               # diagnoser confabulated mechanisms), no document names, no
                                               # document/answer text. Consumed by the external improvement
                                               # agent (see improve_loop.sh).
DOSSIER_INCLUDE_QUESTION = False               # keep question text out of dossiers by default (deidentified);
                                               # flip to True if you want the agent to see question wording

# ---- agent knobs (set for thoroughness on a single question, not speed) ----
STRATEGY      = "cluster"   # virtual subfolders so a wide folder becomes <= MAX_BRANCH groups
GROUP_SUMMARY = "llm"       # real semantic summary per group; a far better routing signal than a name list
MAX_BRANCH    = 6           # (retained) options shown per decision
SCORE_BATCH   = 5           # children scored per LLM call; small so the JSON never truncates
SCORE_PREVIEW = 700         # chars of a candidate's ACTUAL text shown to the ranker, in ADDITION to
                            #   its summary. Root cause of 'right section scored ~0.1': the ranker
                            #   routed on name+summary only, and a section's summary is lossy — an
                            #   'Ungraded PT Challenges' section that literally says what to do for a
                            #   failed challenge can read as irrelevant once compressed to a summary,
                            #   and its name doesn't surface-match 'failed'. Showing the real words
                            #   lets the model score on content. Folders still route on names, not text.
PREVIEW_SCAN_CAP = 60       # max chunks walked to build one preview, so this stays cheap on big folders
PREVIEW_RELEVANCE = True    # build a candidate's `text:` excerpt from the chunks most similar to the
                            #   QUESTION rather than the section's opening words. Root cause of the
                            #   'Procedure scored 0.50' miss: the answer-bearing sentences (sign in on
                            #   receipt / check out on removal) sat deeper than the first 700 chars, so
                            #   the ranker judged the section on its preamble. Embeddings are cached.
DEFERRED_MAX_READS = 4      # when the sufficiency gate says 'not enough', how many deferred same-file
                            #   sections may be pulled back and read before teleporting away.
MAX_STEPS     = 40          # total node visits (hard cap so teleporting can never loop forever)
MAX_FILES     = 8           # stop once this many evidence files are collected (agent may stop sooner)
MAX_EVIDENCE  = 50          # cap on evidence pieces fed to the final answer
# ---- SYNTHESIS MODE (enumeration questions: "what are ALL of our X") ----
# A vast minority of questions ask to ENUMERATE a class of items spread across the corpus
# ("what are our validated clinical assays", "list all SOPs for X"). The default agent -- greedy
# descent file->section->paragraph, then finish-the-file -- is the wrong shape here: it drills into
# deep per-assay validation reports (rejecting non-answer paragraphs while the assay name sits in the
# Scope/Conclusion), which cost ~800s and still missed assays. When a per-question heuristic detects
# an enumeration, synthesis mode does the OPPOSITE of drilling: it descends through FOLDERS normally
# but, on reaching a FILE, takes the WHOLE document as one (clipped) unit -- no section/paragraph
# descent -- and moves straight on to the next DISTINCT document until a small file budget is met.
# A handful of whole documents (a validation report names its assay in its Scope; an activity-menu
# doc lists them all) covers an enumeration far faster than drilling one report to the floor. It
# touches ONLY traversal control flow; the answer prompt (INSTRUCTIONS) is unchanged, so this is not
# a prompt-side intervention and does not reintroduce the removed completeness ladder or coverage gate.
SYNTH_MAX_FILES = 4         # breadth budget: gather this many DISTINCT documents (each taken whole),
                            #   or until the frontier runs dry, then answer. No sufficiency gate in
                            #   synthesis -- the gate under-counts enumerations and stopped short.
SYNTH_FILE_CHARS = 8000     # clip each whole-document synthesis unit to this many chars, so 3-4 whole
                            #   reports stay a lean answer prompt; the assay identity sits near the top
SYNTH_BREADTH_WINDOW = 0.20 # RELEVANCE-GATED BREADTH. A fixed SYNTH_MAX_FILES cap answers an
                            #   enumeration with a PARTIAL list whenever the corpus spreads the class
                            #   across more documents than the cap: the frontier still holds documents
                            #   scoring like the ones already taken, each likely a DISTINCT item, yet
                            #   the search stops. So past SYNTH_MAX_FILES the search keeps gathering as
                            #   long as an unexplored NEW-source frontier document scores within this
                            #   window of the WEAKEST document already accepted -- i.e. its item is as
                            #   relevant as the ones kept -- hard-bounded by MAX_FILES and MAX_STEPS so
                            #   it cannot run away. When the remaining candidates all score well below
                            #   the accepted ones, the enumeration is genuinely complete and it stops at
                            #   SYNTH_MAX_FILES exactly as before, so small enumerations are unaffected.
# COVERAGE_GATE has been REMOVED, not switched off. The gate re-read the evidence, enumerated the
# elements the answer "should" contain, and rewrote the answer to add any it had missed; a
# compression pass then tried and failed to shrink the result. qms_search has no such stage -- it
# drafts once from its retrieved text and stops. Keeping the gate meant TreeRAG's answers were the
# product of retrieval PLUS a repair loop while qms's were the product of retrieval alone, so the
# benchmark compared post-processing rather than evidence quality. It was also one-directional
# (added elements, never dropped out-of-scope ones), which is what pushed answers past 200 words.
# There is no flag to restore it: `git checkout` the v25 notebook if you want the old behaviour.
ANSWER_MAX_WORDS = 100      # the house limit, as stated verbatim in qms's INSTRUCTIONS. Nothing in
                            #   the pipeline enforces it programmatically -- and nothing in qms's
                            #   pipeline does either. The prompt is the only enforcement, on both
                            #   sides. Retained for the report's answer_long_* diagnostic warning.
NOISE_FLOOR   = 0.20        # the one deliberate threshold, adopted because nearly every diagnosis of
                            #   a qms win pointed at time wasted on clearly-irrelevant nodes: candidates
                            #   scoring below this are NOT placed on the active frontier or shown at
                            #   triage. Recall is preserved — they go to a RESERVE tier that is only
                            #   drawn from when the active frontier is exhausted and the evidence is
                            #   still insufficient, so nothing is ever unreachable, just deprioritized.
NOSIGNAL_ABORT = 2          # NO-SIGNAL DESCENT ABORT. At a FOLDER routing level, a best-child score
                            #   below NOISE_FLOOR is the ranker declaring IGNORANCE ("nothing listed
                            #   here relates"), so the descent taken on it is a guess. One guess is
                            #   allowed -- a level deeper the ranker sees richer names/text and may
                            #   recover a real signal -- but after this many CONSECUTIVE all-floor
                            #   folder rankings on the same line the guess is falsified: drilling on
                            #   only re-derives "nothing here" at higher cost. The line is abandoned
                            #   (children stay reachable on frontier/reserve) and the search teleports
                            #   to the best above-floor candidate elsewhere. Folders only: INSIDE a
                            #   document, low section scores can be summary-lossiness, and reads /
                            #   sweeps are the mechanism there.
CLUSTER_DELTA = 0.15        # intra-file cluster auto-include. The model's opt-in triage kept returning
                            #   'nothing' even when a document already chosen as relevant held a whole
                            #   cluster of 0.90+ paragraphs (an enumerated procedure), so most of the
                            #   answer was left on the floor. Now: same-file candidates scoring within
                            #   CLUSTER_DELTA of the top remaining score are READ automatically (the
                            #   per-section take/skip still guards each against redundancy). The model's
                            #   opt-in is kept only for the lower, ambiguous tier below the cluster.
CLUSTER_HIGH  = 0.60        # a same-file candidate this relevant on its own is auto-read regardless of
                            #   the top, so a strong section is never dropped just because something else
                            #   scored even higher.
# ---- MULTI-PART (COMPOUND) QUESTIONS ------------------------------------------------------------
# A question can bundle several distinct asks into one stem ("how long must X be kept, must it say
# when it is corrected, and must it be issued for Y"). The pipeline had no notion of this and failed
# it in two general ways, both visible in the same run: the search settled the first part and then
# spent the rest of its budget re-drilling THAT part, and the answer -- handed a bundle in collection
# order with no record of which piece carried which ask -- stated two of the three and dropped the
# third, one that had in fact been retrieved. Both knobs below are inert on a single-ask question
# (which is most of them): the decomposition returns one part and every mechanism no-ops.
PARTS_MAX       = 4    # max distinct sub-questions the decomposition may name
PART_SELECT_MIN = 6    # only settle up the bundle when it is at least this big; a small bundle
                       #   cannot bury a part, so it is handed over untouched
PART_KEEP       = 2    # pieces led with per sub-question once the agent has placed them
# ---- NAME-MATCH BOOST (navigation) --------------------------------------------------------------
# Repeated qms-win pattern: the correct FILE/FOLDER was reached-or-reachable but ranked 2nd/3rd behind
# a decoy, and the greedy agent committed to the decoy while the gate accepted it. The tell was always
# a NAME the LLM under-weighted: a 'file a CAPA' question whose answer sits in 'Non-Conformance and CAPA
# Procedure.docx' (out-ranked by a KPI doc), a 'library prep' question losing to 'Deparaffinization',
# a 'cleaning checklist' question whose 'Laboratory Cleaning Checklists' subfolder ranked below
# 'Equipment Calibration'. So after the LLM scores, a small DETERMINISTIC nudge is added to any
# file/folder candidate whose NAME shares distinctive (stemmed) terms with the question. It is capped
# small so it only reorders NEAR-ties -- a clear content winner (gap > the cap) is never overturned --
# and it is gated by a proximity WINDOW so a low-scored name-decoy can't leap a confident winner.
# Set NAME_BOOST_MAX = 0.0 to disable and A/B it against a run with it on.
NAME_BOOST_STEP   = 0.06     # per distinctive stemmed token shared between question and candidate name
NAME_BOOST_MAX    = 0.12     # cap (2+ shared terms) -> only near-ties get reordered by name
NAME_BOOST_WINDOW = 0.25     # only candidates within this of the top RAW score are eligible for the nudge
# ---- REGION QUOTA ON AGENTIC SHORTLISTS (navigation) --------------------------------------------
# See _diverse_shortlist. Caps how many of ONE top-level region's entries may occupy the option list
# shown to an agentic teleport choice, so the remaining slots go to the best each OTHER region offers.
# Small enough that the committed region still leads the list with its strongest candidates (so a
# question the region really does answer is unaffected), large enough that it is never shut out.
# Set SHORTLIST_REGION_QUOTA >= ALT_SHORTLIST to disable and A/B it against a run with it on.
SHORTLIST_REGION_QUOTA = 3
THINKING      = False       # gpt-oss reasoning; the big latency lever, keep off for clean json
KEEP_ALIVE    = "30m"
print(f"agent {AGENT_MODEL}; sequential benchmark over all of {QUESTIONS_FILE}; max {MAX_STEPS} steps/question")


# In[ ]:


client = ollama.Client(host=OLLAMA_URL, timeout=1500)
LLM_LAST = {}   # metadata of the most recent llm() call: done_reason, content_len, thinking_len

def _model_names(r):
    raw = r.get("models", []) if hasattr(r, "get") else getattr(r, "models", [])
    out=[]
    for m in raw:
        n = getattr(m,"model",None) or getattr(m,"name",None)
        if n is None and isinstance(m,dict): n=m.get("model") or m.get("name")
        if n: out.append(n)
    return out

def _wait_until_ready():
    announced=False
    while True:
        try:
            names=_model_names(client.list())
            if any(AGENT_MODEL in n for n in names):
                print(f"ollama ok; {AGENT_MODEL} is loaded"); return
            reason=f"{AGENT_MODEL} not loaded yet"
        except Exception as e:
            reason=f"server unreachable; {type(e).__name__}: {e}"
        if not announced:
            print(f"waiting for ollama, {reason}; rechecking every 10s and wont stop"); announced=True
        time.sleep(10)

def llm(prompt, counter, num_predict=512, temperature=0, think=None, thinking_fallback=True, model=None):
    # thinking_fallback: when the model routes everything into its 'thinking' field and leaves
    # content empty, returning the thinking text is useful for JSON-parsing callers (scores,
    # judges) but catastrophic for the ANSWER call — chain-of-thought would leak to the user.
    # Answer-producing callers pass thinking_fallback=False so an empty content stays empty and
    # the caller's retry machinery handles it instead.
    use_think = THINKING if think is None else think
    opts={"temperature":temperature, "num_predict": num_predict + 2048}
    attempt=0; pass_think=True
    while True:
        kw=dict(model=(model or AGENT_MODEL), messages=[{"role":"user","content":prompt}],
                options=opts, keep_alive=KEEP_ALIVE)
        if pass_think: kw["think"]=use_think
        try:
            r=client.chat(**kw)
            counter.calls+=1
            try: counter.in_tok  += int(r["prompt_eval_count"] or 0)
            except Exception: pass
            try: counter.out_tok += int(r["eval_count"] or 0)
            except Exception: pass
            txt=(r["message"]["content"] or "").strip()
            thk=""
            try: thk=(r["message"]["thinking"] or "").strip()
            except Exception: pass
            # metadata for the caller: was the call cut off by num_predict ("length"), and did the
            # model spend its budget reasoning? lets answer assembly detect reasoning-exhaustion
            # (content empty because ALL tokens went to the analysis channel) instead of misreading
            # it as "the model had nothing to say".
            try: LLM_LAST.update(done_reason=str(r.get("done_reason","") or ""),
                                 content_len=len(txt), thinking_len=len(thk))
            except Exception: pass
            if not txt and thinking_fallback:
                txt=thk
            return txt
        except TypeError:
            pass_think=False
        except Exception as e:
            attempt+=1
            if attempt==1 or attempt%5==0: print(f"[waiting for ollama] {type(e).__name__}: {e}; retrying")
            try: open("improve/run_logs/heartbeat.log","a").write(f"{time.strftime('%H:%M:%S')} OLLAMA-RETRY attempt {attempt} {type(e).__name__}\n")
            except Exception: pass
            time.sleep(min(60, 5*2**min(attempt-1,4)))

def embed(text, counter):
    try:
        r=client.embeddings(model=EMBED_MODEL, prompt=text or " ")
        return r["embedding"]
    except Exception:
        return None

_wait_until_ready()
print("llm and embed helpers ready")


# In[4]:


# ---- data types ----
@dataclass
class TreeNode:
    node_id:str; node_type:str; name:str; path:str; summary:str
    content:str=""; children:List["TreeNode"]=field(default_factory=list)
    metadata:Dict[str,Any]=field(default_factory=dict)

    @classmethod
    def from_dict(cls,d):
        n=cls(node_id=d["node_id"],node_type=d["node_type"],name=d["name"],
              path=d.get("path",""),summary=d.get("summary",""),
              content=d.get("content",""),metadata=d.get("metadata",{}))
        n.children=[cls.from_dict(c) for c in d.get("children",[])]
        return n

    def is_leaf(self): return self.node_type=="chunk"
    def count_leaves(self): return 1 if self.is_leaf() else sum(c.count_leaves() for c in self.children)


@dataclass
class Question:
    qid:str; stem:str; options:Dict[str,str]; answer:str; difficulty:str=""
    answers:List[str]=field(default_factory=list)   # one or more correct letters; answer stays as the first
    modified:bool=False   # set by resolve_referential_answers.py: a referential answer ("All of the
                          # above", "Both A and B", ...) was rewritten to concrete letters; such a
                          # question is reprocessed as if it had no benchmark entry (see the queue).

def _ans_list(q):
    return list(q.answers) if getattr(q,"answers",None) else ([q.answer] if q.answer else [])
def _opts_text(q):
    return "\n".join(f"{L}. {q.options[L]}" for L in "ABCDEF" if L in q.options)

# ---- REFERENTIAL OPTIONS: "All of the above", "Both A and B", "None of the above" ----
# The agent answers in prose and never sees the choices, so it can never say "all of the above".
# When the correct option's TEXT is a pointer to other options, grading against that literal text
# is meaningless: the judge must grade against the CONTENT the pointer stands for. These helpers
# resolve a referential option into the concrete options it actually asserts, so
#   "E. All of the above" (with A,B,C above)  ->  effective answer = A AND B AND C
#   "D. Both A and B"                          ->  effective answer = A AND B
# and the multi-answer rubric ("full credit only if the response conveys ALL of them") then applies
# automatically. "None of the above" cannot be expanded and is flagged for the judge instead.
_REF_FILLER = {"both","and","or","answers","answer","options","option","the","above","of",
               "choices","choice","only","these","are","correct","is","statements","all"}
_RE_ALL  = re.compile(r"^\W*(?:all|any)\s+of\s+(?:the\s+)?(?:above|these|options|other\s+options)\b.*$", re.I)
_RE_NONE = re.compile(r"^\W*none\s+of\s+(?:the\s+)?(?:above|these|options)\b.*$", re.I)

def _combo_letters(text, q, self_letter):
    """Letters named by a combo option ('Both A and B', 'Answers A and C'). [] if not a combo."""
    words = re.sub(r"[^A-Za-z ]", " ", str(text or "")).split()
    if not words: return []
    letters, leftover = [], []
    for w in words:
        if len(w) == 1 and w.upper() in q.options: letters.append(w.upper())
        elif w.lower() in _REF_FILLER: continue
        else: leftover.append(w)
    if leftover or len(letters) < 2: return []          # real prose -> not a combo option
    return [L for L in letters if L != self_letter]

def _is_referential(text, q, self_letter):
    t = str(text or "").strip()
    return bool(_RE_ALL.match(t) or _RE_NONE.match(t) or _combo_letters(t, q, self_letter))

def _expand_letter(L, q):
    """Expand ONE correct letter into the concrete options it asserts. [] if nothing to expand."""
    t = str(q.options.get(L, "")).strip()
    if _RE_NONE.match(t): return []                      # not expandable by construction
    if _RE_ALL.match(t):
        return [X for X in "ABCDEF" if X in q.options and X != L
                and not _is_referential(q.options[X], q, X)]
    return _combo_letters(t, q, L)

def _effective_letters(q):
    """The concrete options a correct prose answer must convey. 'E. All of the above' -> [A,B,C]."""
    out = []
    for L in _ans_list(q):
        for X in (_expand_letter(L, q) or [L]):
            if X not in out: out.append(X)
    return out

def _is_multi(q):
    """True when the answer requires conveying more than one concrete option (drives the rubric)."""
    return len(_effective_letters(q)) > 1

def _gold_letters_str(q):
    al, eff = _ans_list(q), _effective_letters(q)
    s = ", ".join(al)
    if eff and eff != al: s += f"  (which stands for: {', '.join(eff)})"
    return s

def _gold_text(q):
    """Ground truth shown to the judge, with referential options RESOLVED to their content."""
    al = _ans_list(q)
    if not al: return ""
    eff = _effective_letters(q)
    lines = [f"{L}. {q.options.get(L,'')}" for L in al]
    if any(_RE_NONE.match(str(q.options.get(L,"")).strip()) for L in al):
        lines += ["",
                  "NOTE: the correct option states that NONE of the listed options is correct. The "
                  "student could not see the choices, so a correct response is one whose stated facts "
                  "are inconsistent with every listed option (or that correctly reports the true fact); "
                  "a response that affirms any listed option is WRONG."]
    elif eff != al:
        lines += ["",
                  "NOTE: the correct option above is a POINTER to other options, not content. The "
                  "student answered in prose and could never write 'all of the above', so grade the "
                  "response against the CONTENT it stands for. To be fully correct, the response must "
                  "convey ALL of the following:"]
        lines += [f"{X}. {q.options.get(X,'')}" for X in eff]
        lines += ["Award proportional partial credit for conveying only some of them."]
    return "\n".join(lines)


class Counters:
    def __init__(self): self.in_tok=0; self.out_tok=0; self.calls=0

def clip(t,n):
    t=re.sub(r"\s+"," ",t or "").strip()
    return t if len(t)<=n else t[:n]+" …"

def full(t):                              # normalise whitespace but NEVER truncate
    t=(t or "").replace("\r\n","\n")
    t=re.sub(r"[ \t]+"," ",t)
    t=re.sub(r"\n{3,}","\n\n",t)
    return t.strip()

# index parent pointers so a section can be reassembled from its chunks
_NODES={}; _PARENT={}
def index_tree(root):
    _NODES.clear(); _PARENT.clear(); st=[(root,None)]
    while st:
        n,p=st.pop(); _NODES[n.node_id]=n
        if p is not None: _PARENT[n.node_id]=p
        for c in n.children: st.append((c,n.node_id))
    return _NODES,_PARENT

def all_chunks(node):
    out=[]; st=[node]
    while st:
        n=st.pop()
        if n.is_leaf(): out.append(n)
        else: st.extend(reversed(n.children))
    return out

# a whole document/section collapsed into one evidence item carrying its full text
def whole_unit(node):
    chunks=all_chunks(node)
    body=full("\n\n".join((c.content or c.summary or "") for c in chunks))
    src=next((c.metadata.get("source_file") for c in chunks if c.metadata.get("source_file")),None) or node.path or node.name
    md=dict(node.metadata); md["source_file"]=src
    return replace(node, content=body, metadata=md)

print("data types ready")


# In[5]:


# ---- ranking over REAL child summaries: score each child 0-1 for relevance to the question ----
# Scores order the descent winner AND seed the teleport frontier. No vgroups / group summaries /
# embeddings. The all-0.5 (or one-high-rest-flat) bug came from ONE thing: with many children the
# model's JSON score object was truncated by num_predict, so most indices went unscored and fell to
# a default. Fix: score in small fixed batches of SCORE_BATCH (=5) with a generous token budget, so
# every response is tiny and complete. Each item is scored exactly once, in its own small batch.
_nav_cache = {}

def get_nav_children(node, counter):
    return list(node.children)   # real children only, no virtualisation

# ---- CONTENTS-AWARE FOLDER SCORING ----
# Root-level failure this fixes: the ranker only ever saw `name — summary` per candidate, and a
# folder's LLM summary is lossy — "Quality SOPs and Worksheets" scored 0.71 (rank 4) off a generic
# summary while the exactly-right file sat inside it, invisible. Now every interior candidate also
# shows a `contains:` line listing the names of its children (files/subsections), and the scoring
# prompt says to treat those names as a first-class routing signal. SOP/worksheet file names are
# usually dead giveaways, so the folder that holds the right file now scores like it. No extra chat
# calls. When a folder is too wide for every name to fit the character budget, the listed names are
# ordered by embedding similarity to the question (embeddings cached across questions) so the
# relevant name is never the one that gets clipped out; if the embed endpoint is down we fall back
# to natural order.
CONTAINS_CHARS = 1500  # char budget for one candidate's contains-line. Raised from 600: mid-size
                       #   folders (Quality SOPs, Technical SOPs) now list ALL their file names, so
                       #   the exactly-right file (e.g. 'Proficiency Testing Program.docx',
                       #   'Initial Sample QC - Qubit.docx') is never the name that gets clipped out.

_name_emb_cache = {}
def _emb_cached(text, counter):
    v = _name_emb_cache.get(text)
    if v is None:
        v = embed(text, counter)
        if v is not None: _name_emb_cache[text] = v
    return v

def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b) / d

def _contains_line(query, c, counter):
    """One extra line for an interior candidate: 'contains (N items): name; name; ...'.
    Leaves/chunks return ''. Query-similarity-ordered only when the full list overflows the budget,
    so small folders stay deterministic and cost nothing."""
    kids = list(getattr(c, "children", None) or [])
    if not kids: return ""
    if all(k.is_leaf() for k in kids): return ""   # para/chunk children carry no routing signal
    names = [str(k.name or "?") for k in kids]
    if sum(len(n) + 2 for n in names) > CONTAINS_CHARS:
        # Order so query-relevant names SURVIVE the clip. PRIMARY signal is LEXICAL overlap with the
        # question -- deterministic, free, and it nails exact hits an embedding can miss (an 'Initial
        # Sample QC - Qubit' file for a Qubit question). Embedding similarity is a SECONDARY signal for
        # synonyms (ILC ~ Proficiency Testing) when the endpoint is up; natural order is the final
        # tiebreak. Was embedding-ONLY: it floated nothing when embed was down and ignored exact
        # lexical matches -- the two ways the right filename got clipped in the field.
        qset = set(_toks(query))
        try: qv = _emb_cached(query, counter)
        except Exception: qv = None
        def _rank_key(n):
            lex = sum(1 for w in set(_toks(n)) if w in qset)
            emb = 0.0
            if qv is not None:
                try:
                    nv = _emb_cached(n, counter)
                    if nv is not None: emb = _cos(qv, nv)
                except Exception: emb = 0.0
            return (lex, emb)
        names = sorted(names, key=_rank_key, reverse=True)
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > CONTAINS_CHARS: break
        shown.append(n); used += len(n) + 2
    extra = len(kids) - len(shown)
    line = "; ".join(shown) + (f"; (+{extra} more)" if extra > 0 else "")
    return f"\n      contains ({len(kids)} item{'s' if len(kids) != 1 else ''}): {line}"

# ---- CONTENT-AWARE SCORING (fixes: correct section scored ~0.1 because only its summary was seen) ----
# The ranker used to see only `name — summary`. A section's summary is a lossy routing gist: a section
# that literally answers the question can score near-zero when its summary drops the answer-bearing
# sentences and its name uses different words than the question ('Ungraded' vs 'failed'). So every
# single-document candidate (a file or, crucially, a section) now also shows a `text:` excerpt of its
# ACTUAL words. Folders keep routing on the `contains:` names, not text (a folder's "text" would just
# be one document's intro). Preview assembly is bounded (early-stop by chars and by PREVIEW_SCAN_CAP
# chunks) so it stays cheap even when a candidate is a huge top-level folder.
def _iter_chunks_bounded(node, cap):
    st = [node]; k = 0
    while st and k < cap:
        nd = st.pop()
        if nd.is_leaf(): yield nd; k += 1
        else: st.extend(reversed(nd.children))

# LEXICAL relevance for previews \u2014 NO embeddings, NO network. Embedding chunks here was a
# catastrophic mistake: _content_preview runs for every candidate at every ranking step, so per-chunk
# embed() calls turned one question into thousands of Ollama round-trips (1277s and a thrashed search).
# A section preview is only a routing hint, so a cheap bag-of-words overlap with the question is more
# than enough to float the answer-bearing paragraphs ('signed into RAMEN', 'checked out of RAMEN') to
# the top when the section overflows the budget. Pure Python, deterministic, effectively free.
_STOP = set("a an the of to in on for and or is are be as at by with from this that these those it its "
            "which what when how who whom where why any all each such into per via if then than also "
            "may must will shall should can could would about within between during under over".split())
def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 2 and w not in _STOP]

# ---- NAME-MATCH BOOST helpers (see NAME_BOOST_* in the knobs cell) ----
# Structural/format words carry no topical signal and would over-fire the boost, so they are stripped
# from the NAME side before matching (a doc's '... Procedure.docx' should not match every question).
_NAME_STOP = set("procedure procedures plan plans document documents sop sops worksheet worksheets "
                 "docx pdf doc form forms log logs record records file files".split())
def _stem(w):
    # tiny suffix stemmer so checklist~checklists, bench~benches, clean~cleaning match; deliberately
    # does NOT expand abbreviations (qc != quality control, prep != preparation) -- those stay misses.
    for suf in ("ings", "ing", "ies", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)] + ("y" if suf == "ies" else "")
    return w
def _stem_set(text):
    return {_stem(w) for w in _toks(text)}
def _name_match_boost(qstems, node):
    """Small additive nudge for a FILE/FOLDER whose name shares distinctive stemmed terms with the
    question. 0.0 for sections/paragraphs (their names are auto-generated; content-preview is the
    right signal there) and for the empty intersection."""
    if getattr(node, "node_type", None) not in ("folder", "file"):
        return 0.0
    nstems = {_stem(w) for w in _toks(node.name or "") if w not in _NAME_STOP}
    hits = len(qstems & nstems)
    return min(NAME_BOOST_MAX, NAME_BOOST_STEP * hits) if hits > 0 else 0.0

def _content_preview(node, limit, cap=None, query=None, counter=None):
    """Return (text, multi_doc). For a single-document node whose text overflows `limit`, the excerpt
    is the chunks with the highest lexical overlap with the QUESTION, restored to document order so it
    reads naturally \u2014 so a section that states the answer past its preamble is previewed on the answer,
    not the preamble. Folders short-circuit (no text preview; they route on contained names)."""
    cap = cap or PREVIEW_SCAN_CAP
    if node.is_leaf():
        return clip(full(node.content or node.summary), limit), False
    chunks, srcs = [], set()
    for i, ch in enumerate(_iter_chunks_bounded(node, cap)):
        sf = ch.metadata.get("source_file")
        if sf:
            srcs.add(sf)
            if len(srcs) > 1:
                return "", True                    # multi-doc folder: bail early, build nothing
        t = full(ch.content or "")
        if t: chunks.append((i, ch, t))
    if not chunks: return "", False

    picked = chunks
    overflows = sum(len(t) + 2 for _, _, t in chunks) > limit
    if PREVIEW_RELEVANCE and query and overflows and len(chunks) > 1:
        qset = set(_toks(query))
        if qset:
            scored = [(sum(1 for w in _toks(t) if w in qset), i, ch, t) for i, ch, t in chunks]
            scored.sort(key=lambda x: x[0], reverse=True)
            sel, used = [], 0
            for sc, i, ch, t in scored:
                if used >= limit: break
                if sc == 0 and used > 0: continue   # skip zero-overlap chunks once we have some content
                sel.append((i, ch, t)); used += len(t) + 2
            if sel:
                picked = sorted(sel, key=lambda x: x[0])   # back to document order

    buf, used = [], 0
    for _i, _ch, t in picked:
        if used >= limit: break
        buf.append(t); used += len(t) + 2
    return clip("\n".join(buf), limit), False

def _score_entry(i, c, query, counter):
    """One candidate's block for the ranker: name + summary + (for single-doc nodes) a text excerpt
    of the real content + (for folders) the contains-line of child names."""
    if c.is_leaf():
        prev, _ = _content_preview(c, SCORE_PREVIEW, query=query, counter=counter)
        return f"[{i}] {c.name} — {prev}"
    prev, multi = _content_preview(c, SCORE_PREVIEW, query=query, counter=counter)
    summ = clip(c.summary or "", 240)
    line = f"[{i}] {c.name}" + (f" — {summ}" if summ else "")
    if not multi and prev:                       # file/section: show its actual words
        line += f"\n      text: {prev}"
    line += _contains_line(query, c, counter)    # informative child names (folders); '' for para-only
    return line

def _parse_scores(raw, n):
    """Return {index: float} for indices 0..n-1 found in the model text, tolerant of shape."""
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    out = {}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            sc = obj.get("scores", obj) if isinstance(obj, dict) else obj
            if isinstance(sc, dict):
                for k, v in sc.items():
                    ks = str(k).strip()
                    if ks.isdigit():
                        try: out[int(ks)] = max(0.0, min(1.0, float(v)))
                        except (TypeError, ValueError): pass
            elif isinstance(sc, list):
                for i, v in enumerate(sc):
                    try: out[i] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError): pass
        except Exception:
            pass
    if not out:   # loose fallback: only accept "<index>": <0-1> pairs (index < n) from the scores region
        region = s
        mm = re.search(r'"scores"\s*:\s*\{(.*)', s, flags=re.S)
        if mm: region = mm.group(1)
        for k, v in re.findall(r'"(\d+)"\s*:\s*(0?\.\d+|[01](?:\.0+)?)', region):
            ki = int(k)
            if 0 <= ki < n: out[ki] = max(0.0, min(1.0, float(v)))
    return {i: out[i] for i in out if 0 <= i < n}

def _reasoning_of(raw):
    m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw or "")
    return m.group(1).strip() if m else ""

def _score_small_batch(query, candidates, memory, counter):
    """Score a batch of <= SCORE_BATCH candidates; guaranteed a real score for each (retry once,
    then a best-pick tie-break for any the model still refuses to score — never a flat default)."""
    n = len(candidates)
    lines = "\n".join(_score_entry(i, c, query, counter) for i, c in enumerate(candidates))
    mem = "\n".join("- " + m for m in memory) or "(empty)"
    keys = ", ".join(f'"{i}":<0-1>' for i in range(n))
    def _call(strict):
        note = ("" if not strict else
                f"\nIMPORTANT: score EVERY index 0..{n-1}. Output all {n} scores, no omissions.\n")
        prompt = (
            "You are navigating a document tree to answer a question. Rate how likely EACH entry "
            "below is to contain information that helps answer the question.\n"
            "Each entry may show up to three things: a SUMMARY (a lossy gist), a 'text:' excerpt (the "
            "section's ACTUAL words), and a 'contains:' line (names of the files/subsections inside). "
            "Weight them accordingly:\n"
            "- 'text:' is the strongest signal. If the actual words address the question, score HIGH "
            "even when the name and summary do not obviously match — a summary often drops the exact "
            "sentence that answers, and a section's name may use different words than the question.\n"
            "- 'contains:' names are a first-class routing signal for folders: a folder whose listed "
            "contents match what is asked should score HIGH even if its summary reads generic.\n"
            "- Match on MEANING, not shared keywords. Domain-equivalent wordings count: an "
            "'unsatisfactory', 'ungraded', or 'unacceptable' result IS a 'failed' one; 'corrective "
            "action' / 'follow-up' is 'what to do' after a problem; 'frequency' answers 'how often'. "
            "Judge whether the content addresses the question's INTENT.\n"
            "- Do not demand a particular FORM of answer. A question asking WHEN / WHY / HOW / UNDER "
            "WHAT CIRCUMSTANCES an event happens is answered not only by an explicit rule sentence "
            "but equally by the process, trigger, or form that INITIATES the event ('done as part of "
            "Y', 'requires initiating Z'), and by a record DOCUMENTING an actual instance of it (the "
            "circumstances in the record show when it happens). Score such content HIGH.\n"
            # SUBJECT-IDENTITY bullet: the recurring miss is the mirror image of keyword-matching --
            # the ranker floats every entry that matches the question's ATTRIBUTE words (the fact
            # being asked) and ignores its IDENTIFYING words (which one thing it is asked about), so
            # on a corpus that records the same attribute for many same-kind entities the search
            # commits to whichever sibling states the attribute most prominently.
            "- The question's IDENTIFYING words matter as much as its attribute words. When it asks "
            "a fact about one SPECIFIC named thing (a particular instrument, unit, system, model, "
            "site or document), an entry stating the SAME KIND of fact for a DIFFERENT thing of that "
            "kind is a decoy: score it LOW however exactly its wording matches the asked attribute, "
            "and score HIGH entries identifiably about the named thing itself (or the branch "
            "covering it), even when their wording matches the attribute less well.\n\n"
            f"QUESTION: {query}\n\n"
            f"WORKING MEMORY (facts gathered so far):\n{mem}\n\n"
            f"CANDIDATES:\n{lines}\n\n"
            "Give ONE short sentence of reasoning FIRST, then score each section from 0.0 (irrelevant) "
            "to 1.0 (almost certainly contains the answer). Use the full range; differentiate them — "
            "do NOT give everything the same middling number." + note +
            "\nReply with ONLY json:\n"
            '{"reasoning":"<1 short sentence>", "scores":{' + keys + "}}")
        # budget scales with batch size so the JSON is never truncated (>= 120 tokens/item + slack)
        raw = llm(prompt, counter, num_predict=max(256, 140 * n), temperature=0, think=False)
        return raw
    raw = _call(strict=False)
    scores = _parse_scores(raw, n); reasoning = _reasoning_of(raw)
    missing = [i for i in range(n) if i not in scores]
    if missing:
        raw2 = _call(strict=True)
        s2 = _parse_scores(raw2, n)
        for i, v in s2.items(): scores.setdefault(i, v)
        if not reasoning: reasoning = _reasoning_of(raw2)
        missing = [i for i in range(n) if i not in scores]
    if missing:   # separate leftovers by real signal, just under the lowest real score (never a flat 0.5)
        base = min(scores.values(), default=0.5)
        rem = list(missing); val = max(0.0, base - 0.01)
        while rem:
            bi = _pick_best_index(query, [candidates[i] for i in rem], memory, counter)
            gi = rem.pop(bi if 0 <= bi < len(rem) else 0)
            scores[gi] = max(0.0, val); val = max(0.0, val - 0.02)
    return [(candidates[i], scores[i], reasoning) for i in range(n)]

def _pick_best_index(query, candidates, memory, counter):
    if len(candidates) == 1: return 0
    lines = "\n".join(_score_entry(i, c, query, counter) for i, c in enumerate(candidates))
    prompt = ("Which ONE of these sections is MOST likely to contain information answering the "
              f"question?\n\nQUESTION: {query}\n\nSECTIONS:\n{lines}\n\n"
              'Reply ONLY json: {"choice":<index>}')
    raw = llm(prompt, counter, num_predict=128, temperature=0, think=False)
    m = re.search(r'"?choice"?\s*[:=]\s*(\d+)', raw or "")
    if m:
        ci = int(m.group(1))
        if 0 <= ci < len(candidates): return ci
    return 0

def rank_children(query, node, candidates, memory, counter):
    """Score EVERY real child in fixed batches of SCORE_BATCH, returning [(child, score, reasoning)]
    sorted high->low. Small batches keep each JSON response complete, so no child defaults to 0.5."""
    pool = list(candidates)
    if not pool: return []
    results = []
    for i in range(0, len(pool), SCORE_BATCH):
        results.extend(_score_small_batch(query, pool[i:i+SCORE_BATCH], memory, counter))
    # NAME-MATCH BOOST: nudge file/folder candidates whose name matches the question, but ONLY within a
    # proximity window of the top raw score (so a low-scored name-decoy can't overtake a confident
    # winner), clamped to 1.0. Ties are broken by the boost itself (stronger name match first) then the
    # raw score, so the reorder is deterministic. Disabled cleanly when NAME_BOOST_MAX == 0.
    if NAME_BOOST_MAX > 0 and results:
        qstems = _stem_set(query)
        top_raw = max(sc for _, sc, _ in results)
        adj = []
        for c, sc, rsn in results:
            b = _name_match_boost(qstems, c) if sc >= top_raw - NAME_BOOST_WINDOW else 0.0
            adj.append((c, min(1.0, sc + b), rsn, b, sc))
        adj.sort(key=lambda t: (t[1], t[3], t[4]), reverse=True)
        return [(c, s, r) for c, s, r, _b, _raw in adj]
    results.sort(key=lambda t: t[1], reverse=True)
    return results

def best_child(query, node, candidates, memory, counter):
    r = rank_children(query, node, candidates, memory, counter)
    return r[0][0] if r else None

print(f"ranking ready (batches of {SCORE_BATCH}; content-aware section + folder scoring; complete scores)")


# In[6]:


# ---- the agent: greedy descent with a teleport frontier (no backtrack) ----
# The agent descends the single best-ranked child at each level. Every runner-up sibling it
# passes is pushed onto ONE global frontier with its relevance score, each entry carrying the
# node itself so a teleport RE-ENTERS full descent there (so it can jump back to a top-level
# folder if that's the best thing left). On reaching a file it reads it and decides, in its own
# words, whether to TAKE it as evidence or reject it as irrelevant. Rejecting a file is the ONLY
# thing that triggers a teleport: jump to the highest-scoring node left on the frontier. It ends
# when the agent says it has ANSWERED / enough, or on the MAX_FILES / MAX_STEPS caps.
DOMAIN="public news"
ORGANIZATION="the MultiHop-RAG public corpus"
DUNNO="Insufficient information."
SYSTEM_PROMPT = (
  "You are a skilled information retrieval agent for "
  f"a {DOMAIN} document library for the "
  f"{ORGANIZATION}.\n"
  "Your task is to research answers to user questions using the provided documents.\n"
  f"Today's date is {datetime.today().strftime('%Y-%m-%d')}. "
  "Use only the supplied public corpus; do not rely on prior knowledge."
)
# INSTRUCTIONS: verbatim from qms_search's prompting.py, so the ONLY difference between the two
# systems under test is the EVIDENCE each one brings to the answer. The prompt is a controlled
# variable. Three deviations, each forced and each noted:
#   1. SYSTEM_PROMPT (above) is NOT qms's. qms's describes a search function and a text-retrieval
#      function; TreeRAG exposes neither, and the tool-shaped prompt sent gpt-oss into unbounded
#      deliberation that never emitted an answer. Ours states the same role without the tools.
#   2. [doc_id] here is the full source-file path. qms emits integer indices and rewrites them into
#      a reference table in format_llm_references(); we have no such post-pass. Cosmetic.
#   3. The one-cite-per-doc sentence is ours, not qms's. _dedupe_citations() below depends on it.
#      This is a real, known confound on citation style; it does not affect answer content.
# Everything else is qms's text, word for word, including the 100-word limit and the instruction
# to prefer exact direct quotes. The completeness ladder, the scope preamble and the soft word
# target that earlier revisions added here are REMOVED: they were prompt-side interventions that
# qms does not have, and they made the comparison measure prompt engineering rather than retrieval.
INSTRUCTIONS = (
  "Answer user questions ONLY using the information in the "
  "files. Do not use any prior knowledge in your answers. "
  "Do not speculate. If the answer cannot be found in the "
  f"files, say '{DUNNO}'\n"
  "If possible, return only exact, direct quotes of the most "
  "relevant passages from the documents followed by a "
  "reference to the respective document ID. Format your "
  "reference as [doc_id], where `doc_id` is the document ID.\n"
  "Cite each document only ONCE: group all information drawn from the same document together and "
  "put a single reference to that document, formatted as [doc_id], at the END of that information. "
  "Do NOT repeat the same [doc_id] after every sentence.\n"
  "Keep your answers to 100 words or less. If a question "
  "requires a more detailed answer which cannot be expressed "
  "in less than one hundred words, refer the user to the "
  "relevant primary source documents instead, citing their "
  "document IDs following the above formatting guideline. "
  "For example, if a user asks for a specific SOP, simply "
  "provide the name of the SOP document and cite its "
  "document ID.\n"
)

# Prefix marking a steering DIRECTIVE in working memory rather than a gathered fact. The recency cap
# must never evict one: on any real run the evidence facts alone overflow `cap` several times over, and
# an evicted directive silently stops steering every decision that reads memory.
_PIN = "!! "
def _add_memory(mem,fact,cap=20):
    fact=clip(fact,500)
    if fact and fact.lower() not in ("","none","n/a") and fact not in mem:
        mem.append(fact)
        if len(mem) > cap:
            pinned = [m for m in mem if m.startswith(_PIN)]
            facts  = [m for m in mem if not m.startswith(_PIN)]
            del facts[:-max(1, cap - len(pinned))]
            mem[:] = pinned + facts

def _short(n): return (n.name or "?")[:22]

# The document/file level of the tree. Different corpora name this node type differently ("file" in
# some builds, "document" in others); treating BOTH as the file level keeps every file-level branch
# (synthesis whole-take, whole-file judging) working regardless of the label. Without this, a synthesis
# branch gated on == "file" silently no-ops on a "document"-typed tree and falls back to section
# drilling -- the exact depth behaviour synthesis exists to avoid.
_FILE_TYPES = ("file", "document")
def _is_file_node(node):
    return getattr(node, "node_type", None) in _FILE_TYPES

# --- the agent reads a reached FILE and decides, in its own words, take vs reject ---
def _judge_file(query, node, memory, counter, full_content=False):
    # Topical keep/skip: does this document contribute REAL answer-bearing content (even partial,
    # like a definition or one required fact)? 'reject' only if it contributes nothing an answer
    # could be built on (boilerplate / off-topic / mentions the words but holds no actual info).
    # NOTE: the "is what I have ENOUGH to answer?" question is handled separately, at the moment the
    # agent tries to STOP — see the sufficiency gate in run_agent — not here.
    # SYNTHESIS whole-file judging: a file node's own .content is usually empty and its summary is a
    # lossy gist, so judge on the assembled document text (clipped). Only for file nodes in synthesis;
    # the normal leaf path is untouched, so this cannot regress default navigation.
    if full_content and _is_file_node(node):
        body = clip(full(whole_unit(node).content or node.summary), 3000)
    else:
        body = clip(full(node.content or node.summary), 3000)
    mem = "\n".join("- " + m for m in memory) or "(empty)"
    prompt = (
        "You have navigated to a specific document and can now read it. Decide, based ONLY on its "
        "content, whether it contributes real answer-bearing information toward the question.\n\n"
        f"QUESTION: {query}\n\n"
        f"WORKING MEMORY (facts gathered so far):\n{mem}\n\n"
        f"DOCUMENT: {node.name}\nCONTENT:\n{body}\n\n"
        # ANSWER-FORM RIGIDITY GUARD. The judge's recurring failure mode is rewriting the question
        # into a stricter one — deciding in advance what SHAPE the answer must take (an explicit rule,
        # a timing statement, a number) and rejecting every passage that answers the question's actual
        # intent in the corpus's own form. Corpora rarely state an answer as a single rule sentence:
        # they state the PROCESS that performs the event, cross-reference the procedure that governs
        # it, or hold a RECORD of an instance of it. The tell of this failure is a reject whose own
        # reasoning restates answer-bearing content ("it only says X is initiated via Y, without
        # stating when...") — so the judge is also told to check its reasoning before rejecting.
        "Judge the content against the question AS ASKED — do not silently substitute a stricter "
        "question that demands a specific FORM of answer (an explicit rule, timing, or number). "
        "For a question about when / why / how / under what circumstances something happens, ALL of "
        "these are real answer-bearing content:\n"
        "- a TRIGGER, CONDITION, or INITIATING PROCESS: the asked-about event is performed or "
        "initiated through some process or form ('X is changed by initiating Y', 'as part of Z') — "
        "that process IS the answer to when/how/under-what-circumstances it happens;\n"
        "- a cross-reference naming WHICH procedure, form, or record governs or performs the event;\n"
        "- a concrete RECORD or completed form documenting an actual INSTANCE of the event — the "
        "circumstances recorded in it are evidence of when and why the event occurs.\n\n"
        # SUBJECT-IDENTITY GUARD (the opposite trap). The recurring failure: the question asks a fact
        # about ONE specific entity, the corpus records the same attribute for MANY same-kind
        # entities, and the judge takes any document stating the right KIND of fact regardless of
        # WHOSE fact it is — so the search accumulates conflicting values from sibling entities and
        # the answer blends them. Wrong-entity matches must be rejected (which also lets reject-decay
        # push the search out of the sibling cluster), and an unattributed value must be flagged so
        # the sufficiency gate can see it is not yet tied to the asked subject.
        "SUBJECT IDENTITY — the opposite trap: do not accept a document merely because it states the "
        "RIGHT KIND of fact for the WRONG thing. When the question asks a fact about one specific "
        "named subject, first ask WHOSE fact this document states — corpora routinely record the "
        "same attribute for MANY same-kind entities (other units, instruments, models, sites, "
        "versions):\n"
        "- if the document identifiably concerns a DIFFERENT entity of the same kind, 'reject' — the "
        "identical attribute of a sibling entity is NOT the answer, however exactly it matches the "
        "asked attribute — and put WHICH entity it does describe in 'remember', so the search can "
        "tell the siblings apart;\n"
        "- if it states the asked fact without identifying which entity the value belongs to, you "
        "may 'take' it, but say in 'remember' that the value is not yet tied to the asked subject.\n\n"
        "Give ONE sentence of reasoning FIRST, then decide:\n"
        "- 'take' if it holds real answer-bearing content toward the question — even a partial piece "
        "(a definition, an example, one required fact) counts.\n"
        "- 'reject' if it contributes nothing an answer could be built on (boilerplate, off-topic, or "
        "merely mentions the right words without actual information). Being in the right file or using "
        "the right terms is NOT enough.\n"
        "Before deciding 'reject', re-read your reasoning sentence: if it says the document 'only' "
        "states some fact ABOUT the asked-about event (what initiates it, which process handles it, "
        "an instance of it happening), then that fact IS answer-bearing — decide 'take' and put it "
        "in 'remember'.\n"
        "Reply ONLY json:\n"
        '{"reasoning":"<1 sentence, before deciding>", "remember":"<a useful answer-bearing fact to keep, else empty>", '
        '"decision":"take|reject"}')
    raw = llm(prompt, counter, num_predict=384, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$","",(raw or "").strip(),flags=re.M).strip()
    d = {}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try: d = json.loads(m.group(0))
        except Exception: d = {}
    dec = str(d.get("decision","")).lower().strip()
    if dec not in ("take","reject"): dec = "take"   # when unsure, keep — the stop-gate will verify sufficiency
    return {"decision":dec,
            "reasoning":str(d.get("reasoning","")).strip(),
            "remember":str(d.get("remember","")).strip()}

# ---- helpers to reason about where a chunk lives in the tree ----
def _enclosing_file(node):
    """The FILE node that contains `node` — for a chunk, a section, or a file itself.
    THE BUG THIS FIXES: `_intra_file_sweep` is documented as 'finish the current file before any
    teleport', but when the agent TOOK an interior section the caller passed that *section* as the
    file. The sweep then ranked only that section's own paragraphs and pulled only frontier entries
    beneath it, so sibling sections of the same document (e.g. Procedure §2 and §6 next to a taken
    Responsibilities section) were never triaged, never auto-included, and the sufficiency gate then
    stopped on a partial answer. Never returns a folder: sweeping a folder would drag in other
    documents, which is the teleport's job, not the sweep's."""
    if _is_file_node(node): return node          # 'file' OR 'document' — see _FILE_TYPES
    cand = _file_of(node.node_id)
    if cand is None or cand.node_type == "folder" or cand.node_id == node.node_id:
        return node                      # no file ancestor (or it resolved upward to a folder)
    return cand

def _file_of(chunk_id):
    """Walk parent pointers up from a chunk to the enclosing FILE node (node_type == 'file' if the
    tree marks it, else the highest ancestor that still maps to one source_file). Falls back to the
    chunk's immediate parent, else the chunk itself."""
    src = _NODES.get(chunk_id)
    src_file = (src.metadata.get("source_file") if src else None)
    node_id = chunk_id; best = None
    while node_id in _PARENT:
        pid = _PARENT[node_id]; parent = _NODES.get(pid)
        if parent is None: break
        if _is_file_node(parent):            # 'file' OR 'document' — see _FILE_TYPES. Without this the
            return parent                    # walk relies on the source_file fallback below, which on a
                                             # single-document FOLDER keeps climbing past the document and
                                             # resolves to the folder — whereupon _enclosing_file gives up
                                             # and returns the chunk, silently disabling every file-level
                                             # mechanism (sweeps, granularity ceiling) for that document.
        # otherwise track the highest ancestor whose chunks all share this chunk's source_file
        if src_file:
            pchunks = all_chunks(parent)
            if pchunks and all((c.metadata.get("source_file")==src_file) for c in pchunks):
                best = parent
        node_id = pid
    return best or (_NODES.get(_PARENT.get(chunk_id, chunk_id)) or _NODES.get(chunk_id))

def _single_source(node):
    """True if every chunk under `node` comes from ONE source document — i.e. taking the whole node
    as a unit will not pull in a *different* file. This is the ceiling for granularity escalation:
    we may climb up to and including the file node, but never into a folder that mixes documents."""
    srcs = {c.metadata.get("source_file") for c in all_chunks(node) if c.metadata.get("source_file")}
    return len(srcs) <= 1

# ---- GRANULARITY FIX (parent-take, replaces the old sibling sweep) ----
# When the agent reaches a relevant leaf it makes ONE agentic decision: is the answer fully
# contained in THIS passage, or does it depend on the surrounding section? If it depends on the
# surrounding section we take the PARENT — the whole section (e.g. the entire 1-9 block) as one
# unit, so a detail the passage omits (the classic "... on the Quality SPN" line that sits a few
# sentences away) is carried along. This is the common case and directly fixes answers that were
# missing a required phrase because only the single highest-scoring sentence was fed to the answerer.
#
# What level of parent? The DIRECT parent by default. But the same question is then asked again of
# that section vs ITS parent: if the section still depends on the wider section/document, we climb
# again (parent -> grandparent -> ...). So it can reach the grandparent or the whole file, but only
# while each step independently says the answer still depends on the wider context. The climb is
# capped at the file boundary (`_single_source`) so it never grabs sibling documents.
def _scope_choice(query, passage_node, section_node, memory, counter):
    """Ask: does the answer live entirely in `passage_node`, or does it need the wider `section_node`?
    Returns 'self' (keep the passage as-is) or 'wider' (climb to the section)."""
    passage = clip(full(passage_node.content or passage_node.summary), 1600)
    section = clip(full(whole_unit(section_node).content), 3500)
    mem = "\n".join("- " + m for m in memory) or "(empty)"
    prompt = (
        "You have found a passage relevant to the question. Decide the right amount of surrounding "
        "context to keep as evidence.\n\n"
        f"QUESTION: {query}\n\n"
        f"WORKING MEMORY (facts already gathered):\n{mem}\n\n"
        f"PASSAGE (the single best-matching part):\n{passage}\n\n"
        f"SURROUNDING SECTION (the passage is one part of this larger section):\n{section}\n\n"
        "Decide:\n"
        "- 'self'  : the PASSAGE on its own fully contains the answer; the rest of the section adds "
        "nothing the answer needs.\n"
        "- 'wider' : the answer DEPENDS ON the surrounding section — a required detail, qualifier, "
        "definition, version, location, or reference the passage omits sits elsewhere in this section. "
        "Keep the whole section so that detail is not lost.\n"
        "Prefer 'self' unless the section clearly carries a needed detail the passage is missing.\n"
        'Reply ONLY json: {"reasoning":"<1 sentence>", "scope":"self|wider"}')
    raw = llm(prompt, counter, num_predict=320, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$","",(raw or "").strip(),flags=re.M).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    scope, reason = "self", ""
    if m:
        try:
            d = json.loads(m.group(0))
            scope = str(d.get("scope","self")).lower().strip()
            reason = str(d.get("reasoning","")).strip()
        except Exception: pass
    if scope not in ("self","wider"):
        mm = re.search(r'"?scope"?\s*[:=]\s*"?(self|wider)"?', s, re.I)
        scope = (mm.group(1).lower() if mm else "self")
    return scope, reason

def _granularity_unit(query, leaf, memory, counter, ink):
    """From a taken leaf, climb to the right granularity: keep escalating to the parent while the
    agent says the answer depends on the wider section, capped at the file (single-source) boundary.
    Returns (node_to_take, whole_flag, trace_list). node may be the leaf itself (whole=False) or an
    ancestor section/file (whole=True). trace_list records each escalation decision."""
    node = leaf
    escalations = []
    while True:
        pid = _PARENT.get(node.node_id)
        parent = _NODES.get(pid) if pid else None
        # stop at the file boundary: never climb into a node that mixes multiple documents,
        # and never climb above the root
        if parent is None or parent.node_type == "root" or not _single_source(parent):
            break
        scope, reason = _scope_choice(query, node, parent, memory, counter)
        escalations.append({"from": node.name, "to": parent.name, "scope": scope, "reasoning": reason})
        if scope != "wider":
            break
        ink("\n        ^ granularity: answer depends on surrounding section -> take parent '%s'" % parent.name)
        node = parent
        # if we've reached the whole file, there's nothing wider left within one document
        if node.node_type == "file":
            break
    took_whole = node.node_id != leaf.node_id
    return node, took_whole, escalations

# ---- INTRA-FILE TRIAGE SWEEP: finish the current file BEFORE the global frontier, efficiently ----
# Intra-file-first is preserved: after a take, the rest of the current file is dealt with COMPLETELY
# before any teleport. What changed is HOW. The old sweep exploded (~110 LLM calls / 824s on one
# file): it expanded low-scored sections into per-paragraph reads and asked continue/stop after every
# single passage with a prompt biased toward continuing — so it ground through 0.01-scored paragraphs
# one at a time. The new sweep spends its calls where the agent says they matter:
#
#   1. TRIAGE (1-2 calls total): EVERY same-file candidate — inherited from the global frontier with
#      the descent score it already earned, plus any unscored file children ranked once — is shown as
#      heading + score + snippet. The triage prompt is COMPLETENESS-oriented, not miserly: within a
#      document already chosen as relevant, sections scoring as high as the taken evidence are presumed
#      part of the same answer, so when the answer is a multi-step procedure / enumerated list spread
#      across consecutive high-scoring sections the agent takes the whole cluster instead of only the
#      first passage. Selectivity is reserved for OTHER documents (the frontier), not the current one.
#      (Historical note: the old sweep was correct to fear per-paragraph explosion, but it over-corrected
#      into taking one passage and leaving the rest of a relevant section behind — the failure this fixes.)
#      Original description continues — heading + score + first-line snippet. The agent selects the FEW sections that could plausibly
#      hold the missing information. It is told explicitly that it is budgeting limited reading time
#      and that the global search continues after this file regardless, so long-shots should not be
#      selected. A 0.01 paragraph is skipped because the agent, seeing its score and heading, does
#      not pick it — an informed agentic decision, not a numeric threshold.
#   2. READ (1 call per selection, in score order): each selection is read as a WHOLE UNIT — a
#      section's full assembled text, never a paragraph-by-paragraph expansion, which is what made
#      the old sweep explode — and the agent takes or skips it. The reading list was fixed by the
#      agent's own triage, so there is no per-passage continue loop left to grind.
#
# Dismissal stays informed: every candidate was seen at triage (heading + score + snippet), and the
# unselected ones are marked visited — the file is FINISHED when the sweep returns, by construction.
TRIAGE_BATCH = 25   # heading+score+snippet lines per triage call; short lines keep the JSON complete

def _node_under(node_id, root_id):
    nid = node_id
    while True:
        if nid == root_id: return True
        if nid not in _PARENT: return False
        nid = _PARENT[nid]

def _triage_select(query, cands_scored, evidence, memory, counter, anchor=None):
    """ONE agentic pass over every remaining same-file candidate (heading + score + snippet).
    Returns the sub-list the agent wants read in full. cands_scored: [(node, score)] sorted desc.

    `anchor` is the score the search committed to for the evidence already kept. The prompt asks the
    agent to select sections "scoring roughly as high as the evidence you just kept" — but that number
    was never shown to it, so the only scale it had was the remaining sections' scores compared to each
    other. Relative to a residue, the residue's own top always looks high, which is the same
    re-normalisation that made the cluster band fire on leftovers. Showing the anchor lets the agent
    make the comparison the prompt already asks of it, and lets it conclude that NOTHING here belongs."""
    have = clip("\n\n".join(full(e.content or e.summary) for e in evidence), 2200) or "(nothing yet)"
    selected = []
    for bs in range(0, len(cands_scored), TRIAGE_BATCH):
        batch = cands_scored[bs:bs+TRIAGE_BATCH]
        lines = "\n".join(f"[{i}] ({sc:.2f}) {n.name} — {clip(full(n.content or n.summary),110)}"
                           for i, (n, sc) in enumerate(batch))
        top_kept = max((sc for _, sc in cands_scored), default=0.0)
        prompt = (
            "You just took evidence from a document you deliberately navigated to. Below is everything "
            "ELSE in that SAME document, each with the relevance score it earned during navigation and "
            "its first line. Your job now is to FINISH this document: select every section that "
            "plausibly belongs to the answer before the search moves on to other documents.\n\n"
            f"QUESTION: {query}\n\n"
            f"EVIDENCE ALREADY KEPT:\n{have}\n\n"
            f"REMAINING SECTIONS OF THIS DOCUMENT (score — heading — first line):\n{lines}\n\n"
            "How to choose:\n"
            "- Every section here is inside a document already judged relevant, and each score is how "
            "relevant navigation found it. Sections scoring roughly as high as the evidence you just "
            "kept are very likely part of the SAME answer — ESPECIALLY when the answer is a multi-step "
            "procedure, an enumerated list, or a set of components / requirements / criteria that runs "
            "across several consecutive sections. When a cluster of high-scoring sections sits together, "
            "that is the signature of an answer spanning the whole section: select them ALL, not just "
            "the first one you already took.\n"
            "- Reading further WITHIN this one document is cheap; the limited-time budget you are "
            "protecting applies to OTHER documents, which are handled separately. So err toward "
            "inclusion here rather than leaving relevant sections behind.\n"
            "- Skip a section ONLY when it is clearly off-topic, boilerplate, or fully redundant with "
            "evidence already kept. A near-floor score with an unrelated heading is a skip.\n"
            "- A section about a NEIGHBOURING subject is not part of this answer. Documents routinely "
            "cover several sibling subjects (different assays, programmes, instruments, processes); a "
            "section stating the asked fact — or stating that it does not apply — for a DIFFERENT one "
            "of them does not answer the question and must not be selected, however similar it reads. "
            "Select it only if it is about the subject the question actually names.\n"
            + (f"(For reference: the evidence already kept was ranked {anchor:.2f}, and the highest "
               f"remaining score here is {top_kept:.2f}.) Judge the remaining sections against the "
               "evidence you kept, NOT against each other — the strong sections of this document are "
               "already taken, so what is left may be nothing but its weakest material, and the top of "
               "that residue is not thereby relevant. Select generously when sections score close to "
               "the kept evidence; select NOTHING when they all sit far below it and the question is "
               "already answered.\n" if anchor else
               f"(For reference, the highest remaining score is {top_kept:.2f}.) Select generously when "
               "high-scoring sections cluster; select little only when the remainder is clearly "
               "unrelated.\n")
            + 'Reply ONLY json: {"reasoning":"<1 sentence>", "read":[<indices>]}')
        raw = llm(prompt, counter, num_predict=max(220, 12*len(batch)), temperature=0, think=False)
        s = re.sub(r"^```(?:json)?|```$","",(raw or "").strip(),flags=re.M).strip()
        idxs = []
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                for v in (d.get("read") or []):
                    try: iv = int(v)
                    except (TypeError, ValueError): continue
                    if 0 <= iv < len(batch): idxs.append(iv)
            except Exception: pass
        for iv in sorted(set(idxs)):
            selected.append(batch[iv])
    return selected

def _read_selection(query, node, evidence, memory, counter, fallback=False, below_bar=False):
    """Read one triage selection as a whole unit; decide take/skip. Returns (decision, adds, why).

    `fallback=True` marks a piece the navigator itself ranked BELOW the bar and deferred, now being
    read only because the sufficiency gate called the kept evidence incomplete.
    `below_bar=True` marks any OTHER piece ranked materially below the evidence already kept. Both
    are the same tier and get the same stricter bar than "does it add information": see BELOW-THE-BAR
    ADMISSION below. They differ only in what an UNPARSED response defaults to."""
    unit = node if node.is_leaf() else whole_unit(node)
    body = clip(full(unit.content or unit.summary), 3000)
    have = clip("\n\n".join(full(e.content or e.summary) for e in evidence), 2400) or "(nothing yet)"
    # BELOW-THE-BAR ADMISSION. The take/skip criterion was purely additive — "does it ADD distinct
    # information" — and never asked whether the section CONTRADICTS what is already held, or whether
    # it is even ABOUT the same subject. So a low-ranked passage that co-mentioned the question's
    # subject always read as "adding" something, was admitted, and silently overturned a direct
    # statement of the answer the search had already found. That is the whole risk of reading on past a
    # satisfied question: the material still on the floor is, by the navigator's own ranking, the
    # weakest in the document, and the piece most likely to mention the subject without governing it.
    # Naming the subject is not answering about it.
    #
    # THE HOLE THIS CLOSES: the tier is defined by the SCORE, but it used to be triggered by the CODE
    # PATH — only the deferred pool (`fallback`) was ever guarded. The intra-file sweep admits material
    # from exactly the same tier (it reads the residue of a document whose answer is already taken) and
    # reached the bundle completely unguarded, so the guard was absent precisely where the sweep's
    # cluster auto-include had just widened admission. A section ranked far below the evidence in hand
    # is the same tier however it got here, and earns its place the same way.
    tier = (("\nIMPORTANT — this section was ranked BELOW the evidence already kept" +
             (", and is being read only because that evidence was judged incomplete."
              if fallback else
              ", which already states an answer to the question.") +
             " It earns a place only by supplying what is actually missing. Two traps:\n"
             "- Mentioning the question's subject is NOT answering about it. If it names the subject "
             "near the topic but does not state the asked fact, 'skip'.\n"
             "- If it CONTRADICTS evidence already kept (e.g. asserts the general rule does not apply, "
             "or that something is handled differently), take it ONLY if it is unmistakably about the "
             "SAME subject the question asks about and genuinely states an exception that overrides. If "
             "it concerns a neighbouring subject, a different programme, or a special case, 'skip' — a "
             "lower-ranked passage must not overturn a direct statement of the answer already in hand, "
             "and a fact stated about a NEIGHBOURING subject is not a fact about the one asked for.\n")
            if (fallback or below_bar) else "")
    prompt = (
        "You selected this section of the document as worth reading. Having read it, decide whether "
        "it ADDS distinct information the answer needs — a required detail, frequency, qualifier, "
        "definition, version, location, rule, OR a further step / component / criterion of the same "
        "procedure or list the question is asking for. A distinct item of an enumerated answer counts "
        "as adding information even if the kept evidence already has other items. Only 'skip' true "
        "repeats or genuinely off-topic text.\n" + tier + "\n"
        f"QUESTION: {query}\n\n"
        f"EVIDENCE ALREADY KEPT:\n{have}\n\n"
        f"SECTION ({node.name}):\n{body}\n\n"
        'Reply ONLY json: {"reasoning":"<1 sentence>", "decision":"take|skip", '
        '"adds":"<if take: what it adds, <=12 words>"}')
    # BUDGET: 320 was too small. `body` is up to 3000 chars and `have` up to 2400, so gpt-oss
    # reasons over ~4-6KB inside a 320-token budget, exhausts it in the analysis channel, and returns
    # empty/truncated content. `"decision"` is the LAST key in the schema, so truncation destroys it
    # first. Same mechanism that made the sufficiency gate return a bogus False at num_predict=256.
    raw = llm(prompt, counter, num_predict=2048, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$","",(raw or "").strip(),flags=re.M).strip()
    # DEFAULT ON FAILURE: `parsed` tracks whether the model actually rendered a decision. A parse
    # failure is NOT a "skip" — this node was already scored highly by the ranker and selected by
    # triage/cluster-auto-include, so two prior signals say it is relevant. Defaulting an
    # unparseable response to "skip" silently discarded the single most relevant section in the
    # document (0.94 "6. Inventory Check-Out and Expiration"), which is why the check-out fact was
    # missing from the answer. `_judge_file` already defaults to "take" when unsure; this now
    # matches. The signature of the old failure was a skip with an EMPTY reasoning string.
    dec, adds, why, parsed = "", "", "", False
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            dec = str(d.get("decision","")).lower().strip()
            adds = str(d.get("adds","")).strip()
            why = str(d.get("reasoning","")).strip()
            parsed = dec in ("take","skip")
        except Exception: pass
    if not parsed:   # salvage a decision from a truncated / non-JSON response before giving up
        mm = re.search(r'"?decision"?\s*[:=]\s*"?(take|skip)"?', s, re.I)
        if mm:
            dec, parsed = mm.group(1).lower(), True
    if not parsed:
        # ...but that rationale inverts on the FALLBACK tier: there the ranker scored the section LOW
        # and triage explicitly declined it, so both prior signals say the opposite. Defaulting it to
        # "take" would admit the weakest material in the document — the tier most likely to merely
        # co-mention the subject or contradict the answer already held — with no agentic decision made
        # at all. Unparseable there means the override was never justified, so it does not happen.
        dec = "skip" if fallback else "take"
        why = ("[unparsed model response; not admitted — a below-bar section must be affirmatively "
               "justified to override kept evidence]" if fallback else
               "[unparsed model response; kept because the ranker scored this section relevant]")
    return dec, adds, why

def _intra_file_sweep(query, file_node, visited, seen, evidence, memory, counter, ink, collect, frontier, reserve,
                      deferred=None, breadth=False, anchor=None):
    # `anchor`: the relevance score the search actually COMMITTED to on the line that reached this file
    # (the descent/teleport score behind the evidence just taken). It is the reference the cluster band
    # and the below-the-bar guard are measured against; None reproduces the old residue-relative
    # behaviour exactly.
    fid = file_node.node_id
    tr = []
    # gather every same-file candidate: frontier entries keep their descent scores; unscored file
    # children get ranked once (scores are used only to order and inform triage, never as a cutoff)
    local, keep = [], []
    for f in frontier:
        (local if _node_under(f["node"].node_id, fid) else keep).append(f)
    frontier[:] = keep
    keep_r = []
    for f in reserve:
        (local if _node_under(f["node"].node_id, fid) else keep_r).append(f)
    reserve[:] = keep_r
    if breadth:
        # SYNTHESIS / BREADTH: the granularity take already captured this file's relevant unit
        # (climbing to the whole report when it names several assays, so multi-assay files stay
        # intact as ONE piece). Do NOT deep-read the rest of the file -- the cluster auto-include is
        # exactly the drilling this mode exists to avoid; in the depth run it read 45 sections of one
        # report and 36 of the next, hitting the 50-piece cap inside two files and starving breadth.
        # So the sweep is evict-only here: drop this file's own sections from the frontier/reserve and
        # block re-entry, so the next teleport lands in a DIFFERENT document and breadth actually
        # happens. It reads nothing and makes no LLM call.
        for f in local: visited.add(f["node"].node_id)
        for c in file_node.children: visited.add(c.node_id)
        ink("\n        [synthesis] kept the top unit; evict-only sweep (dropped %d same-file frontier item(s)), on to the next file" % len(local))
        tr.append({"breadth_evict_only": file_node.name, "evicted_same_file": len(local)})
        return tr
    cands = [(f["node"], f["score"]) for f in local]
    queued = {n.node_id for n, _ in cands}
    fresh = [c for c in file_node.children
             if c.node_id not in visited and c.node_id not in seen and c.node_id not in queued]
    if fresh:
        for c, sc, _ in rank_children(query, file_node, fresh, memory, counter):
            cands.append((c, sc)); queued.add(c.node_id)
    cands = [(n, sc) for n, sc in cands if n.node_id not in visited and n.node_id not in seen]
    if not cands: return tr
    # sub-floor same-file candidates are auto-dismissed here: the file is being finished now, and
    # listing 0.0x noise at triage is exactly the time sink the floor exists to remove
    low = [(n, sc) for n, sc in cands if sc < NOISE_FLOOR]
    for n, _ in low: visited.add(n.node_id)
    cands = [(n, sc) for n, sc in cands if sc >= NOISE_FLOOR]
    if low:
        tr.append({"floor_dismissed": len(low)})
        ink("\n        intra-file: %d sub-floor candidate(s) dismissed without reading" % len(low))
    if not cands: return tr
    cands.sort(key=lambda t: t[1], reverse=True)
    tr.append({"triaged": [{"node": n.name, "score": round(sc,3)} for n, sc in cands]})

    # CLUSTER AUTO-INCLUDE (the fix): the agentic opt-in below reliably returned "nothing" even on a
    # dense 0.90+ cluster — an enumerated procedure spread across paragraphs of the very file we chose
    # — dropping most of the answer. So the top cluster is now READ automatically: any same-file
    # candidate scoring within CLUSTER_DELTA of the reference score below, or >= CLUSTER_HIGH on its
    # own, is read without waiting for the model to pick it. Recall widens without dupes because each
    # one still passes through _read_selection, which only TAKES what adds a distinct step/fact. The
    # model's opt-in triage is preserved for the lower, genuinely-ambiguous tier beneath the cluster.
    top = cands[0][1] if cands else 0.0
    # CO-EQUAL TO WHAT WE TOOK — not to what is LEFT. This band is the mechanism's whole definition of
    # a "cluster", and it was anchored on `top`, the top of the RESIDUE. But by the time the sweep runs
    # every strong section is already taken and marked `seen`, so `top` is whatever the document's
    # weakest material happens to score, and the band RE-NORMALISES onto it: after a 0.95 take, a floor
    # of 0.49/0.47/0.45 leftovers all sit within CLUSTER_DELTA of 0.49 and are auto-read as though they
    # were a co-equal cluster. The threshold could never say "nothing here is co-equal", because it
    # measured the residue against itself — so a document's weakest sections entered the bundle unasked
    # on EVERY sweep, and a passage about a NEIGHBOURING subject (a different assay, programme or
    # special case) then displaced the answer already in hand. CLUSTER_DELTA is documented as catching
    # sections "scoring as high as the taken evidence", so the band is anchored on the score the search
    # actually committed to on this line. max() keeps it monotone: whenever the residue outscores the
    # anchor this is byte-identical to the old behaviour, so no genuine 0.90+ cluster stops auto-reading.
    ref = max(top, anchor or 0.0)
    auto = [(n, sc) for n, sc in cands if sc >= ref - CLUSTER_DELTA or sc >= CLUSTER_HIGH]
    auto_ids = {n.node_id for n, _ in auto}
    ask = [(n, sc) for n, sc in cands if n.node_id not in auto_ids]
    selected = _triage_select(query, ask, evidence, memory, counter, anchor=ref) if ask else []
    read_list = auto + [(n, sc) for n, sc in selected]
    tr.append({"auto_selected": [{"node": n.name, "score": round(sc,3)} for n, sc in auto],
               "selected":      [{"node": n.name, "score": round(sc,3)} for n, sc in selected],
               "anchor": round(ref, 3)})
    ink("\n        intra-file triage: %d candidate(s); %d auto-read (cluster >= %.2f of committed %.2f), %d more picked by agent"
        % (len(cands), len(auto), (ref - CLUSTER_DELTA), ref, len(selected)))

    # Candidates neither auto-included nor picked at triage are marked visited so nothing teleports
    # back into this file mid-search. But they are NOT destroyed: each is DEFERRED with its score, so
    # that if the sufficiency gate later says the evidence is incomplete, the best of them can be
    # pulled back and read before we ever leave this document. (Previously they were dropped on the
    # floor: a 'Procedure' section scoring 0.50 next to a 0.85 'Responsibilities' became permanently
    # unreachable, and the answer it held was lost even though the gate could still have objected.)
    read_ids = {n.node_id for n, _ in read_list}
    n_def = 0
    for n, sc in cands:
        if n.node_id not in read_ids:
            visited.add(n.node_id)
            if deferred is not None and sc >= NOISE_FLOOR:
                deferred.append((n, sc)); n_def += 1
    if n_def:
        tr.append({"deferred": n_def})
        ink("\n        %d unread candidate(s) deferred (recoverable if the evidence proves incomplete)" % n_def)

    for n, sc in sorted(read_list, key=lambda t: t[1], reverse=True):
        if len(evidence) >= MAX_EVIDENCE: break
        if n.node_id in visited or n.node_id in seen: continue
        visited.add(n.node_id)
        auto_tag = " (auto)" if n.node_id in auto_ids else ""
        # A candidate that is NOT co-equal to the score we committed to is the same tier as a deferred
        # one — the weakest material in a document whose answer is already held — so it gets the same
        # subject-scope/contradiction guard rather than the purely-additive criterion. Only meaningful
        # once there IS an account it could displace, hence the `evidence` test.
        below = bool(evidence) and sc < ref - CLUSTER_DELTA
        dec, adds, why = _read_selection(query, n, evidence, memory, counter, below_bar=below)
        if dec == "take" and collect(n, whole=not n.is_leaf()):
            tr.append({"read": n.name, "score": round(sc,3), "decision": "take", "adds": adds, "auto": n.node_id in auto_ids})
            ink("\n        + intra-file take%s (%.2f): %s — %s" % (auto_tag, sc, n.name, clip(adds,80)))
        else:
            tr.append({"read": n.name, "score": round(sc,3), "decision": "skip", "reasoning": why, "auto": n.node_id in auto_ids})
            ink("\n        - intra-file skip%s (%.2f): %s — %s" % (auto_tag, sc, n.name, clip(why,80) or "(no reasoning returned)"))
    return tr

# ---- sufficiency gate: before the agent is allowed to STOP and answer, verify the evidence it has
# would actually produce an answer and not the DUNNO string. If it WOULD be DUNNO, we do not stop — we
# keep searching (intra-file first, then teleport). This is the only thing between a premature
# "could not be found" and continuing to look. It fires ONLY when the agent is about to answer;
# bounded by MAX_FILES / MAX_STEPS so search can't run forever, and DUNNO is emitted only once every
# avenue is exhausted.
GATE_LAST = {}   # structured output of the most recent sufficiency verdict. `missing` is the agent's
                 # own list of the parts the evidence still does not settle — see RESIDUAL STEERING.

def _evidence_would_answer(query, evidence, counter, unread=None, polarity=False, definitional=False,
                           recency=False):
    GATE_LAST.clear()
    if not evidence: return False, "no evidence yet"
    groups, order = {}, []
    for e in evidence[:MAX_EVIDENCE]:
        k = e.metadata.get("source_file") or e.path or e.name
        if k not in groups: groups[k] = []; order.append(k)
        groups[k].append(full(e.content or e.summary))
    docs = "\n\n".join(f"[{k}]\n" + "\n".join(groups[k]) for k in order)
    # For a POLARITY/UNIVERSAL question the default strict-match + completeness criteria are actively
    # wrong: they demand an explicit restatement of the universal claim (which, when the true answer is
    # negative, never exists in the corpus) and treat a governing one-liner as "partial", so the gate is
    # never satisfied and the search grinds through example records to the step cap and emits DUNNO. Swap
    # in criteria that recognise a definition/scope/condition -- or any basis for a NEGATIVE answer -- as
    # sufficient, while still refusing to count a pile of examples as universality.
    if polarity:
        criteria = (
            "POLARITY / UNIVERSAL / EXISTENCE QUESTION: this asks whether a relationship holds ALWAYS / "
            "NECESSARILY / in every case, whether something is required or automatic, OR whether a "
            "specific item EXISTS / is available. It is answered by the DEFINITION, PURPOSE or SCOPE of "
            "the entities involved and the CONDITIONS under which the relationship holds — a governing "
            "policy/procedure statement — or by the INVENTORY / REGISTER / SCOPE / ITEM-LIST that would "
            "ENUMERATE the item's category; and the correct answer is frequently NEGATIVE. Evidence "
            "that the subject serves a broader or different purpose, that the outcome arises only under "
            "specific conditions, OR a governing list/inventory/scope of the relevant category that "
            "does NOT include the item asked about, IS sufficient: it lets you answer (typically 'no'). "
            "Do NOT require an explicit sentence asserting the universal rule or the item's absence; a "
            "definitional, conditional, or enumerating basis for answering yes OR no is enough. But a "
            "set of individual EXAMPLES where the relationship happened to hold, or documents that "
            "merely fail to mention the item without being the governing list/definition for its "
            "category, do NOT by themselves settle the question: if all you have is that, answer "
            "insufficient so the search reaches the defining/enumerating document.\n")
    else:
        criteria = (
            "STRICT MATCH REQUIRED: the evidence must state the SPECIFIC fact the question asks about, for "
            "the process/entity named. A fact about a DIFFERENT process does NOT count — e.g. if asked how "
            "often X assessments occur, a statement that a different process runs on some schedule is NOT "
            "sufficient. When the evidence merely gestures at the topic, answer insufficient — the search "
            "will continue to better candidates.\n"
            # FORM OF THE ANSWER (companion to the same guard in _judge_file). 'Strict match' is about
            # the SUBJECT of the fact, not its grammatical FORM: a gate that demands an explicit rule
            # sentence ('amendments are made when ...') declares a gap the corpus never fills — these
            # corpora state such answers as the PROCESS that performs the event or as a RECORD of an
            # instance of it — and the endless 'not enough' then drives the search into weaker decoys.
            "FORM OF THE ANSWER — judge sufficiency against the question AS ASKED, not against a "
            "stricter form of answer you anticipated (an explicit rule sentence, a timing statement, a "
            "number). A question asking WHEN / WHY / HOW / UNDER WHAT CIRCUMSTANCES something happens "
            "is settled by a stated trigger, condition, or initiating process for the named subject "
            "('done as part of Y', 'requires initiating Z'), or by a record documenting an actual "
            "instance of the event and its cause, even when no sentence restates that as an explicit "
            "rule. Do NOT answer insufficient merely because the answer is stated as a process rather "
            "than a rule.\n"
            # ANSWER-SENSE MATCH (companion to STRICT MATCH, from the other direction). STRICT MATCH
            # guards against the SUBJECT being wrong; this guards against the MEASURE being wrong while
            # the subject is right. The blind spot: a single confidently-stated value that reuses the
            # question's own vocabulary reads as a direct answer, so the gate says ENOUGH on the first
            # such hit -- even when that value denotes a DIFFERENT quantity than the one asked (a count
            # of items handled together vs. the input required per item, a schedule vs. a duration, an
            # upper limit vs. an intended target). Because 'sufficient' clears the stall counter, the
            # search then stops before ever reaching the document that states the asked measure -- which
            # frequently sits, unreached, on the frontier. Requiring the SENSE to match, not just the
            # words, is what keeps the search alive to reach it.
            "ANSWER-SENSE MATCH — beyond the subject being right, the evidence must answer the "
            "particular SENSE the question asks: the specific quantity, measure, property, or "
            "requirement named — not merely one that shares the question's surface vocabulary. Corpora "
            "routinely state several DISTINCT facts using the SAME word, so a confidently-stated value "
            "that reuses that word can denote a DIFFERENT thing than the one asked (e.g. a count of "
            "items handled together vs. the material required per item; a schedule vs. a duration; a "
            "limit vs. a target). Such a near-homonym does NOT settle the question, however exactly its "
            "wording matches and however directly it appears to give 'a number'. When the only evidence "
            "you hold matches the question's words but plausibly answers a DIFFERENT sense than the one "
            "asked, answer insufficient and name the asked sense as missing, so the search continues to "
            "the document that addresses it directly.\n"
            # SUBJECT IDENTITY & CONFLICTING VALUES. The gate's blind spot behind blended answers: on
            # an attribute-of-one-named-thing question it only ever asked whether A value of the asked
            # attribute existed, never WHOSE value each piece states — so the first sibling entity's
            # value read as ENOUGH, further conflicting values from other siblings still read as
            # ENOUGH, and the search stopped without ever reaching the document about the asked
            # subject. Conflict across pieces is the observable signature of that entity confusion,
            # so it is what triggers 'insufficient' — a single consistent value stays sufficient, so
            # single-entity corpora cannot regress into manufactured gaps.
            "SUBJECT IDENTITY & CONFLICTING VALUES — when the question asks for an attribute of one "
            "specific named subject, check WHOSE attribute each piece of evidence states: corpora "
            "routinely record the same attribute for MANY same-kind entities (several units, "
            "instruments, models, sites, versions), and a value stated for a DIFFERENT entity of the "
            "same kind does NOT settle the question, however exactly it matches the asked attribute. "
            "If the evidence states SEVERAL DIFFERENT values for the very fact the question asks — "
            "the signature of values drawn from different entities or occasions — do NOT treat them "
            "as corroboration, and do NOT pick or average among them: that is sufficient ONLY if one "
            "of the values is tied by the evidence itself to the subject the question names; "
            "otherwise answer insufficient and name as missing the value stated FOR that subject "
            "(the document identifiably about it). Nor is AGREEMENT proof of identity: several "
            "pieces stating the SAME value corroborate each other only if at least one of them is "
            "identifiably about (or plausibly governs) the named subject — same-kind documents "
            "about sibling entities routinely repeat the same figure, so repetition alone adds "
            "nothing. A single consistent value from a document "
            "plausibly governing the named subject remains sufficient — do not manufacture a gap "
            "when nothing in the evidence points to a sibling entity.\n"
            # SCOPE INHERITANCE (fixes the manufactured gap). 'Strict match' used to demand that the
            # evidence restate the asked fact FOR THE NAMED ENTITY, and treated a broader statement as a
            # non-answer. But a governing programme/policy states its rule ONCE and it governs every
            # subject under it — restating it per subject is not how these corpora are written. So on any
            # question whose answer is a governing rule, the gate declared a gap the corpus cannot fill,
            # the search kept digging, and the only thing left to find was text that merely CO-MENTIONS the
            # entity (a scope note, an exception, an example). That co-mention then displaced the direct
            # answer already in hand. Inheritance is the general principle the gate was missing: a rule
            # covers the cases within its scope whether or not it names them.
            "SCOPE INHERITANCE — do not manufacture a gap: a rule stated by the programme, policy or "
            "procedure that GOVERNS the thing being asked about DOES settle the question, even when it "
            "never repeats that thing's name. A governing document states its rule once; it does not "
            "restate it for each subject it covers. So if the evidence states the asked fact at the level "
            "that governs the subject, and nothing in the evidence places the subject outside that scope, "
            "that is SUFFICIENT — do NOT answer insufficient merely because the exact name is absent. "
            "Answer insufficient only when the stated rule's scope plainly does not reach the subject.\n"
            # SUPERSESSION. Nothing previously adjudicated disagreement: the gate simply re-read the pile
            # and flipped to 'sufficient' on the strength of whatever arrived last, so a late scrap could
            # silently overturn a direct statement of the answer.
            "SUPERSESSION — if two pieces of evidence DISAGREE, do not believe whichever you read last, "
            "nor whichever merely happens to name the subject. A statement that the general rule does not "
            "apply overrides that rule ONLY when it is unmistakably about the SAME subject the question "
            "asks about and actually says so; a passage that just mentions the subject nearby does not "
            "overturn a direct statement of the asked fact. Say in your reasoning which one governs.\n"
            "COMPLETENESS REQUIRED: 'sufficient' means the evidence answers the question FULLY, not "
            "partially. If the question asks what a process consists of — its components, elements, steps, "
            "requirements, or criteria — then evidence covering only SOME of them is NOT sufficient. Two "
            "specific traps:\n"
            "  - Evidence that assigns RESPONSIBILITIES (who performs or approves something) does not state "
            "what the process actually involves. If the question asks what is done and the evidence only "
            "says who does it, answer insufficient.\n"
            "  - Evidence drawn from a single section of a document that clearly has other relevant "
            "sections (a Procedure, its numbered steps, a Records or Reporting section) is usually partial. "
            "If the answer plausibly continues in a section you have not been given, answer insufficient.\n"
            "Only say sufficient when you could write a COMPLETE answer naming every part the question asks "
            "for. A partial answer that omits one required element is INSUFFICIENT.\n")
    # ENTITY-DEFINITION PROVENANCE. Every criterion above asks whether the evidence STATES the asked
    # fact; none asks whether the document had any business stating it. On a "what is X" question that
    # gap IS the failure: a record that merely uses X describes the one facet its own incident turned
    # on, which reads as a serviceable definition, so the gate says ENOUGH on the first mention it
    # meets -- and because a sufficient verdict clears the stall counter, the breadth escape that could
    # have reached the region holding the actual definition never fires. Provenance is the general
    # principle the gate was missing: a description is only as authoritative as the document's business
    # in giving it. Note this is a RETRIEVAL decision (keep searching or stop), not an answer-side one.
    if definitional and not polarity:
        criteria += (
            "PROVENANCE -- WHOSE DESCRIPTION IS THIS? The question asks what a named entity IS / DOES / "
            "is FOR, so also judge WHERE each description comes from:\n"
            "  - AUTHORITATIVE: the document's own subject IS the entity, or a document that governs, "
            "specifies or overviews it describes it (an SOP, manual, specification, design or overview "
            "document). Even a brief purpose or scope sentence there SETTLES the question -- say "
            "sufficient. Do not demand more once you have this.\n"
            "  - INCIDENTAL: the document's own business is something ELSE (an incident, deviation, "
            "change request, meeting minute, ticket or completed form) and it describes the entity only "
            "in passing, in whatever terms mattered to THAT business. Such a description is real but "
            "PARTIAL BY CONSTRUCTION -- it names the facet that record turned on, not what the entity "
            "is for -- and several such records agreeing does NOT fix this, since they are all "
            "downstream of the same narrow use. If EVERY description you hold is incidental, answer "
            "INSUFFICIENT so the search reaches the document that is actually about the entity.\n"
            "Report which case you are in as \"basis\": 'subject' if any description comes from a "
            "document that governs or is about the entity, 'incidental' if all of them are passing "
            "mentions inside documents about other business, 'none' if nothing describes it.\n")
    # RECENCY / LATEST-STATE. The plain STRICT-MATCH criteria ask only whether A value of the asked
    # attribute is stated -- never whether it is the CURRENT one. On a "what is the current/latest X"
    # question that gap is the failure: the value the search happens to read first reads as a direct
    # answer, the gate says ENOUGH, and it stops before reaching the LATER entry that supersedes it.
    # These values live in documents that record a SERIES over time (change logs, version histories,
    # revision tables), where every entry was current only until the next one replaced it, so a single
    # mid-series value is not the answer -- the MOST RECENT entry is. This makes the gate hold out for
    # the whole series (driving the intra-file/deferred reads that finish the log) rather than accept
    # the first entry; it stays satisfiable and cannot manufacture a gap when no series exists.
    if recency and not polarity:
        criteria += (
            "RECENCY / LATEST-STATE -- the question asks for the CURRENT / LATEST / MOST RECENT / "
            "in-effect value of something the corpus tracks OVER TIME (a version, revision, edition, "
            "status, or effective figure). Such values live in documents that record a SERIES of them "
            "-- change logs, version histories, revision tables, amendment lists -- where each entry "
            "was current only until a later one superseded it. A value you have read is the answer ONLY "
            "when the evidence itself establishes it is the MOST RECENT: it is explicitly marked "
            "current / effective / in-force, or it is the newest-dated or highest-numbered entry AND "
            "nothing indicates a later or higher entry that you have not read (including unread "
            "sections of the same document). If the value you hold is one entry drawn from such a "
            "series and later or other entries exist or plausibly remain unread, do NOT assume the one "
            "you happened to read is the current one: answer insufficient and name as missing 'confirm "
            "the most recent entry across the full series'. When the value you hold is a single value "
            "NOT part of any time-series -- nothing indicates other versions/revisions exist -- it "
            "stays sufficient; do not manufacture a gap.\n")
    prompt = (
        "Decide whether the evidence below is SUFFICIENT to actually answer the question. Simulate "
        "answering using ONLY this evidence: if you can state a real, substantive answer, say yes; if "
        f"you would have to reply '{DUNNO}' or could only give a vague/partial non-answer, say no.\n"
        + criteria +
        f"\nQUESTION: {query}\n\nEVIDENCE:\n{docs}\n\n"
        + (("SECTIONS OF THE SAME DOCUMENT YOU HAVE NOT BEEN GIVEN:\n"
            + "\n".join(f"  - {nm}" for nm in unread)
            + "\nThese exist and are unread. If any of them, judging by its heading, plausibly states "
              "a part of the answer that the evidence above does not — the procedural steps, the "
              "records kept, what staff must do and when — then the evidence you have is INCOMPLETE: "
              "answer insufficient so that section gets read.\n"
              # ...but this nudge is one-directional, and it was firing on questions the evidence had
              # ALREADY answered: an unread heading is always 'plausible', so the gate kept saying
              # 'not enough' and the recovery loop kept admitting weaker and weaker material until
              # something contradicted the answer already held. An unread section is a reason to keep
              # looking only while something the question actually asks for is still missing.
              "This is a reason to keep looking ONLY while a part of the question is genuinely "
              "unanswered. If the evidence above already states the fact the question asks for, an "
              "unread heading is not by itself a gap — say sufficient.\n\n") if unread else "")
        + "Give ONE sentence of reasoning FIRST (state which parts of the question the evidence covers "
        "and which, if any, are still missing), then answer. Reply ONLY json:\n"
        # the basis key is requested ONLY on the shape that uses it, so every other question's gate
        # call is byte-identical to before and cannot regress.
        '{"reasoning":"<1 sentence>", '
        + ('"basis":"subject|incidental|none", ' if (definitional and not polarity) else "")
        + '"missing":["<part still unanswered>"], "sufficient":true|false}')
    raw = llm(prompt, counter, num_predict=2048, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$","",(raw or "").strip(),flags=re.M).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            suff = bool(d.get("sufficient", False))
            miss = [str(x).strip() for x in (d.get("missing") or []) if str(x).strip()]
            # a model that names a missing part but still ticks sufficient=true is contradicting
            # itself; believe the missing list and keep searching.
            if suff and miss: suff = False
            # ...and the same applies to a model that reports an all-incidental basis and still ticks
            # sufficient: it has just said every description it holds comes from a document with no
            # business defining the entity. Believe the basis, exactly as we believe the missing list.
            basis = str(d.get("basis", "")).strip().lower()
            GATE_LAST["basis"] = basis
            if definitional and basis == "incidental":
                suff = False
                if not miss:
                    miss = ["a document whose own subject is this entity: what it is and what it is for"]
            GATE_LAST["missing"] = miss
            why = str(d.get("reasoning","")).strip()
            if miss: why += f"  [still missing: {'; '.join(miss)}]"
            return suff, why
        except Exception: pass
    mm = re.search(r'"?sufficient"?\s*[:=]\s*(true|false)', s, re.I)
    return (bool(mm and mm.group(1).lower()=="true"), "")

# ---- QUESTION DECOMPOSITION: what distinct things is this stem actually asking? ------------------
def _decompose_parts(query, counter):
    """The DISTINCT sub-questions a complete answer must settle, plus the question's SPECIFIC SUBJECT
    (the one particular thing whose fact is asked, '' when the question is about a category/process in
    general). Returns (parts, subject). ONE llm call — it replaces the old key-concepts call rather
    than adding to it, so this is cost-neutral; the subject rides along on the same call. Most stems
    ask exactly one thing and return one part, which makes every compound mechanism below a no-op; a
    stem that genuinely asks several things returns one part per ask. Agentic by construction: the
    model decides where the seams are, so nothing here is tied to any corpus or question wording."""
    raw = llm(
        "Break the question into the DISTINCT sub-questions a complete answer must settle — one entry "
        "per separate thing being asked. MOST questions ask exactly ONE thing; return a single entry "
        "then. Split ONLY where the stem really does ask separate things, each with its own answer "
        "(usually joined by 'and' / 'or' / commas). Never split one ask into restatements of itself.\n"
        "Also name the question's SPECIFIC SUBJECT: the ONE particular, individuated thing (a named "
        "instrument, unit, system, assay, site, document, version) whose fact is being asked, with "
        "every identifying qualifier the stem gives (its name, model, location, the system it "
        "serves). Leave it \"\" when the question asks about a category, process, policy or rule in "
        "general rather than one specific instance that could be confused with same-kind siblings.\n\n"
        f"QUESTION: {query}\n\n"
        f'Reply ONLY json: {{"parts":["<sub-question, <=12 words>", ...], '
        f'"subject":"<the one specific thing asked about, else empty>"}}  (at most {PARTS_MAX} parts)',
        counter, num_predict=320, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    out, subject = [], ""
    if m:
        try:
            d = json.loads(m.group(0))
            for p in (d.get("parts") or []):
                p = re.sub(r"\s+", " ", str(p)).strip(" .")
                if p and p not in out: out.append(p)
            subject = re.sub(r"\s+", " ", str(d.get("subject") or "")).strip(" .")
        except Exception: pass
    return out[:PARTS_MAX], subject

# ---- RESIDUAL STEERING (navigation) -------------------------------------------------------------
# The sufficiency gate already reports, in its own words, WHICH parts of the question the evidence does
# not yet settle — and that list was thrown away, read only for its truthiness. Meanwhile rank_children
# goes on scoring every candidate against the WHOLE original stem. On a question that asks several
# things, the part that is already answered therefore keeps attracting the search: it is still most of
# the stem's wording, so the documents that match it still score highest, and the search re-drills
# settled ground (in the run this fixes: a second and a third retention document, long after retention
# was settled) while the parts still open steer nothing. The remedy re-aims the search at the RESIDUAL
# question — the gate's own missing-parts list is pinned into working memory, which the ranker and
# every evidence judge already read, so a candidate is scored on whether it settles what is still OPEN
# rather than on similarity to a stem that is mostly answered. This is the agent's own verdict feeding
# the agent's next decision: no index, no similarity search, no corpus keywords. It no-ops on a
# single-part question, where the gate names no parts and the residual is simply the whole question.
# The residual note is the ONLY pinned entry _set_residual owns, but it used to clear EVERY _PIN-
# prefixed entry — so any other standing directive pinned into memory (a question-shape directive that
# must steer every decision for the whole run) was silently deleted the first time the gate reported,
# and every later ranking decision lost it. It now clears only its own line, identified by its own tag.
# Nothing else changes: the residual note already begins with this exact text.
_RESID_TAG = _PIN + "FOCUS THE SEARCH."
def _set_residual(memory, missing):
    memory[:] = [m for m in memory if not m.startswith(_RESID_TAG)]
    if not missing: return
    parts = "; ".join("(%d) %s" % (i, clip(p, 120)) for i, p in enumerate(missing[:PARTS_MAX], 1))
    memory.insert(0, _PIN + (
        "FOCUS THE SEARCH. Parts of the question the evidence does NOT yet settle: " + parts + ". "
        "The question's other parts are already settled by the facts below, so re-confirming them adds "
        "nothing. Score a candidate HIGH only if it plausibly settles one of the parts listed above; a "
        "candidate that only covers ground already settled scores LOW however topical it looks."))

# ---- PART-COVERAGE SELECTION: the retrieval hand-off --------------------------------------------
# The other half of the compound-question failure, and the half that actually loses the mark. The
# search settles every part — but each part is settled in a DIFFERENT document at a different step, and
# the bundle handed over is whatever accumulated, in COLLECTION order: dozens of pieces, many of them
# whole sections climbed for context, several of them further copies of a part settled long ago, and
# the piece that settles the LAST part sitting at the bottom because it was found last. Nothing records
# which piece carries which part, so the answer — written once, to the house word limit — states the
# parts it meets first and drops the rest, and retrieval is marked down for a part it did retrieve.
# So the last act of RETRIEVAL is to settle up: the agent says which collected piece settles which
# part, and the bundle is rebuilt part by part, each part leading with the piece that settles it, the
# redundant bulk behind them dropped. This is evidence SELECTION — the same job qms's top-k does when
# it hands its chunks over — and NOT an intervention on the answer: the answer prompt and its single
# call are untouched, and nothing re-reads, re-grades or rewrites what the model then writes.
# Conservative by construction: it runs only on a genuinely multi-part question with a bundle big
# enough to bury a part; pieces are DROPPED only when the agent placed at least one against EVERY part
# (a bundle it could not fully account for is reordered but kept whole, since an unplaced piece may be
# the one carrying the open part); and any parse failure hands the bundle back untouched.
def _select_by_part_coverage(query, parts, evidence, counter):
    """Returns (evidence, per_part). per_part is None when the bundle was handed back untouched."""
    lines = "\n".join(
        "[%d] (%s) %s" % (i, clip(str(e.metadata.get("source_file") or e.path or e.name), 60),
                          clip(full(e.content or e.summary), 300))
        for i, e in enumerate(evidence))
    plist = "\n".join("%d. %s" % (i, p) for i, p in enumerate(parts, 1))
    raw = llm(
        "The search is finished. Below are the PARTS of the question and every piece of evidence that "
        "was collected for it. Say which pieces settle which part.\n\n"
        f"QUESTION: {query}\n\nPARTS:\n{plist}\n\nEVIDENCE PIECES:\n{lines}\n\n"
        f"For each part, list the piece indices that STATE its answer, best first, at most {PART_KEEP} "
        "per part. A piece that merely mentions the topic without stating the answer is NOT a match — "
        "leave it out. One piece may serve several parts. Leave a part's list empty only if nothing "
        "here settles it.\n"
        'Reply ONLY json: {"coverage":{"1":[<indices>], ...}}',
        counter, num_predict=512, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return evidence, None
    try:
        cov = (json.loads(m.group(0)).get("coverage") or {})
    except Exception:
        return evidence, None
    if not isinstance(cov, dict): return evidence, None
    picked, used, per_part = [], set(), {}
    for pi in range(1, len(parts) + 1):
        got = []
        raw_idx = cov.get(str(pi)) or cov.get(pi) or []
        if not isinstance(raw_idx, list): raw_idx = []
        for v in raw_idx[:PART_KEEP]:
            try: j = int(v)
            except (TypeError, ValueError): continue
            if not (0 <= j < len(evidence)): continue
            got.append(j)
            if j not in used: used.add(j); picked.append(j)
        per_part[pi] = got
    if not picked: return evidence, None
    if not all(per_part[pi] for pi in per_part):
        picked += [j for j in range(len(evidence)) if j not in used]
    return [evidence[j] for j in picked], per_part

# ---- SYNTHESIS DETECTION: enumerate-a-category questions ("what are ALL of our X") ----
# A vast minority of questions ask to ENUMERATE a class of items spread across the corpus
# ("what are our validated clinical assays", "list all SOPs", "which instruments are accredited").
# These want BREADTH, not the default DEPTH. Detection is a deterministic lexical heuristic on the
# QUESTION STEM only -- NO llm call, so the benchmark stays reproducible and cost-neutral against
# qms_search and nothing is added to the hot path. It is deliberately CONSERVATIVE: a scoped
# "steps/parts OF <one procedure>" question is single-document and is excluded, and the enumeration
# must be over a plural corpus-class noun, so a false positive is rare (and only mildly costly -- a
# couple of extra relevant files rather than a wrong answer).
_SYNTH_CAT_PL = (r"assays|sops|procedures|documents|policies|instruments|tests|methods|reagents|"
                 r"controls|records|forms|templates|systems|registers|kits|panels|workflows|"
                 r"processes|standards|certifications|accreditations|guidelines|manuals|worksheets|"
                 r"logs|assets|analyzers|platforms|analytes|markers")
_SYNTH_CAT_SG = (r"assay|sop|procedure|document|policy|instrument|test|method|reagent|control|"
                 r"record|form|template|system|register|kit|panel|workflow|process|standard|"
                 r"certification|accreditation|guideline|manual|worksheet|log|asset|analyzer|"
                 r"platform|analyte|marker")
_SYNTH_SINGLE_SCOPE = re.compile(
    r"\b(steps?|stages?|phases?|parts?|components?|sections?|fields?|elements?)\s+(of|in|for|to|within)\b", re.I)
_SYNTH_PATTERNS = [
  (r"(?:^\s*(?:list|enumerate)\b|\b(?:list|enumerate)\s+(?:all|the|our|every|each|of)\b)", "list/enumerate"),
  (r"\ball of (?:our|the|your|oicr'?s)\b", "'all of our/the'"),
  (r"\bevery\s+(?:%s)\b" % _SYNTH_CAT_SG, "'every <category>'"),
  (r"\bwhat are (?:all |our |the )+.*\b(?:%s)\b" % _SYNTH_CAT_PL, "'what are all/our <categories>'"),
  (r"\bwhat\b.*\b(?:%s)\b.*\b(?:do we|does oicr|are (?:there|available|validated|approved|in use|current))\b" % _SYNTH_CAT_PL, "'what <categories> do we have / are validated'"),
  (r"\bwhich\b.*\b(?:%s)\b.*\b(?:validated|approved|active|available|in use|accredited|certified|current)\b" % _SYNTH_CAT_PL, "'which <categories> ... validated'"),
  (r"\bwhich (?:validated|approved|current|available|accredited|certified)\b.*\b(?:%s)\b" % _SYNTH_CAT_PL, "'which validated/... <categories>'"),
]
_SYNTH_COMPILED = [(re.compile(p, re.I), why) for p, why in _SYNTH_PATTERNS]

def _is_synthesis_question(stem):
    """(is_synthesis, why). True when the stem asks to enumerate a corpus-wide class of items rather
    than a single fact or the parts of one named thing. Pure regex, deterministic, no LLM call."""
    s = " " + re.sub(r"\s+", " ", (stem or "").strip().lower()) + " "
    if _SYNTH_SINGLE_SCOPE.search(s):
        return False, ""                      # "steps/parts OF <one thing>" -> single document
    for rx, why in _SYNTH_COMPILED:
        if rx.search(s):
            return True, why
    return False, ""

# ---- POLARITY / UNIVERSALITY DETECTION: "does X always lead to Y", "is Y required for every X" ----
# A distinct question shape the default pipeline handles badly: it asks whether a RELATIONSHIP holds
# UNIVERSALLY or NECESSARILY (always / automatically / in every case / is-required), and its correct
# answer is frequently NEGATIVE. Two confirmation biases combine to fail it: (1) the ranker floats
# individual EXAMPLE records that mention both entities together — decoys, since one instance neither
# proves nor disproves a universal rule — over the governing procedure that DEFINES the subject and the
# CONDITIONS for the outcome; and (2) the sufficiency gate demands an explicit restatement of the
# universal claim, which for a false universal never exists, so it grinds through instances to the step
# cap and answers DUNNO. Detection is a deterministic lexical test on the STEM only (no LLM, no corpus
# keywords), mirroring _is_synthesis_question. It only relaxes the gate and adds a navigation note, so a
# false positive is cheap: a governing definition/condition answers most questions anyway.
_POLARITY_MARKERS = re.compile(
    r"\b(always|automatically|necessarily|invariably|inevitably|guarantee[ds]?|mandatory|"
    r"obligatory|without exception|in (?:all|every) cases?|every ?time|each time|whenever|"
    r"in all instances)\b", re.I)
# a yes/no polar frame led by an auxiliary (does/is/must/... — NOT a wh-word) carrying a necessity modal
_POLARITY_MODAL = re.compile(
    r"^(?:does|do|is|are|can|will|must|should|would|has|have|need)\b.*\b(?:must|required|mandatory|"
    r"have to|need to|necessary)\b", re.I)
# EXISTENCE / AVAILABILITY frame ("is there a dedicated X", "do we have a Y", "does an X exist"). This
# is the SAME negative-answer shape: when the true answer is 'no', NO document positively names the
# item, so the ranker floats incidental topical matches, _judge_file rejects every file for "does not
# mention X", and the gate demands a confirmation that cannot exist — the search grinds to the step cap
# and answers DUNNO while the vector baseline states the negative. The answer instead lives in the
# governing INVENTORY / REGISTER / SCOPE / DEFINITION that WOULD enumerate the item's category. Only
# distinctive lead frames are matched, so an ordinary yes/no fact question does not trip it.
_EXISTENCE_FRAME = re.compile(
    r"^\W*(?:is|are)\s+there\b"
    r"|\bdo(?:es)?\s+(?:we|oicr|our\s+\w+|the\s+lab\w*)\s+(?:have|own|possess|maintain|keep|hold)\b"
    r"|\bexists?\b", re.I)

def _is_polarity_question(stem):
    """(is_polarity, why). True when the stem asks whether a relationship holds universally or
    necessarily, OR whether a specific item exists / is available — all answerable by a governing
    definition / condition / inventory, and all frequently NEGATIVE. Pure regex."""
    s = re.sub(r"\s+", " ", (stem or "").strip().lower())
    m = _POLARITY_MARKERS.search(" " + s + " ")
    if m: return True, "universal/necessity marker '%s'" % m.group(1).strip()
    if _POLARITY_MODAL.search(s): return True, "necessity-modal polar frame"
    if _EXISTENCE_FRAME.search(s): return True, "existence/availability frame"
    return False, ""

# ---- ENTITY-DEFINITION DETECTION: "what is X", "what does X do", "what is the purpose of X" ------
# A distinct question shape with its own systematic trap. It asks what a NAMED ENTITY is (a tool, a
# system, a role, a programme). Top-down routing sees only names, summaries and child-name lists, so
# the ONLY branches that can score above the noise floor are the ones whose NAMES repeat the entity --
# and in a quality-management corpus those are overwhelmingly TRANSACTIONAL records that merely USE the
# entity while transacting other business (an incident form 'Incorrect <X> Metrics', a change request
# about <X>). The document that DEFINES the entity is titled after its own process and never names the
# entity at all, so its entire region scores at the floor, is banished to reserve, and -- because the
# frontier never empties on a real corpus within MAX_STEPS -- is never reached.
#
# Both halves of the pipeline then confirm the mistake rather than catch it. The ranker reads "nothing
# here mentions X" as IRRELEVANCE when it is really IGNORANCE (it never saw any body text). And the
# sufficiency gate asks only whether a description EXISTS, never whether the document had any business
# giving one -- so the first record's passing description ("X reported the wrong coverage metric")
# reads as a perfectly good definition, the gate says ENOUGH, `stall` never rises, and the breadth
# escape that could have reached the banished region never fires. The answer then states what the
# entity did in one incident instead of what it is. This is precisely where a body-content baseline
# wins, so the remedy has to be a better DECISION, not a content index.
#
# Detection is a deterministic test on the STEM only (no LLM call, no corpus keywords), mirroring
# _is_synthesis_question / _is_polarity_question. The trailing-length bounds keep it to genuine
# name-a-thing stems: "what is <entity>" matches, "what is the retention period for records" does not.
_DEFN_PATTERNS = [
  (r"^\W*what\s+(?:is|are|was|were)\s+(?:(?:an?|the)\s+)?[^\s?]+(?:\s+[^\s?]+){0,2}\s*\??\s*$",
   "'what is <entity>'"),
  (r"^\W*what\s+do(?:es)?\b.{0,40}?\b(?:do|mean|stand for|refer to)\b", "'what does <entity> do/mean'"),
  (r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:purpose|role|function|use)\s+of\b", "'purpose/role of <entity>'"),
  (r"^\W*(?:describe|define|explain)\b", "'describe/define <entity>'"),
  (r"\bwhat\s+(?:is|are)\b.{0,40}?\bused\s+for\b", "'what is <entity> used for'"),
]
_DEFN_COMPILED = [(re.compile(p, re.I), why) for p, why in _DEFN_PATTERNS]

def _is_definitional_question(stem):
    """(is_definitional, why). True when the stem asks what a named entity IS / DOES / is FOR, rather
    than asking for a fact about a process. Pure regex, deterministic, no LLM call."""
    s = re.sub(r"\s+", " ", (stem or "").strip())
    for rx, why in _DEFN_COMPILED:
        if rx.search(s):
            return True, why
    return False, ""

# ---- RECENCY DETECTION: "what is the CURRENT / LATEST / MOST RECENT version / revision of X" ------
# A distinct question shape with its own systematic trap, mirroring the polarity/definitional detectors
# (pure regex on the STEM, no LLM call, no corpus keywords). It asks for the value of something the
# corpus tracks OVER TIME -- a version, revision, edition, status or effective figure -- whose answer is
# the MOST RECENT entry of a SERIES (a change log, version history, revision table). The default search
# is the wrong shape: it descends to the first version-bearing entry, the sufficiency gate rubber-stamps
# whatever value it states, and the search stops before reaching the LATER entry that supersedes it --
# exactly the failure a whole-document baseline avoids by reading every entry. The flag steers NAVIGATION
# (a working-memory directive: prefer the series document, read all its entries) and, load-bearingly,
# tightens the SUFFICIENCY GATE so a single mid-series value is not accepted as 'the current one' until
# the full series has been surveyed. It is a DECISION change, not a content index.
_RECENCY_MARKERS = re.compile(
    r"\b(current(?:ly)?|latest|most\s+recent|newest|up[\s-]?to[\s-]?date|"
    r"in\s+effect|in\s+force|presently|now\s+in\s+use|as\s+of)\b", re.I)

def _is_recency_question(stem):
    """(is_recency, why). True when the stem asks for the CURRENT / LATEST / MOST RECENT state of a
    thing the corpus tracks over time -- a class whose answer is the maximum/most-recent entry of a
    series, not the first matching entry the search reaches. Pure regex, deterministic, no LLM call."""
    s = re.sub(r"\s+", " ", (stem or "").strip().lower())
    m = _RECENCY_MARKERS.search(" " + s + " ")
    if m: return True, "recency marker '%s'" % m.group(1).strip()
    return False, ""

# The standing directive for this shape. It goes in working memory, which the ranker and every
# evidence judge read but the final answer prompt never does -- so this steers NAVIGATION, not the
# answer. It is pinned (_PIN) because it must survive both the recency cap and the residual note for
# the whole run: it governs the root decision AND every decision after a breadth escape. Note what it
# does NOT say: it never claims a name-match is bad. A document titled after the entity itself IS the
# definition and still scores high. What it separates is titles built around an EVENT (a date, a form
# type, a problem, a request) -- records OF something that involved the entity -- from titles built
# around the entity or the process that specifies it.
DEFN_DIRECTIVE = (
  "DEFINITION QUESTION: this asks what a named entity IS / DOES / is FOR. Its answer is the document "
  "whose OWN SUBJECT is that entity -- one that defines, specifies, governs or overviews it -- NOT a "
  "record that merely USES the entity while transacting other business (an incident, deviation, change "
  "request, meeting minute, ticket or completed form). A record describes only the facet its own "
  "business turned on, so it yields a true but narrow statement that reads like a definition and is "
  "not one. Two consequences for scoring: (1) a name repeating the entity means the document MENTIONS "
  "it, not that it is ABOUT it -- a title built around an event, a date, a form type or a problem is a "
  "RECORD of something that involved the entity: score it LOW; a title built around the entity itself, "
  "or around the process that specifies it, is the definition: score it HIGH. (2) The entity's name "
  "being ABSENT from a branch's listed names is NOT evidence that the entity is undocumented there -- "
  "governing documents are titled after their own process, so the one that defines the entity often "
  "never names it, and its whole branch looks silent. Do NOT floor a branch merely because nothing in "
  "it repeats the entity's name; score every branch on whether the KIND of documentation it holds "
  "would DEFINE such an entity, and prefer that over a branch that merely mentions the entity.")

# How many times the provenance rule below may override an otherwise-sufficient verdict and push the
# search onward. It bounds the worst case -- an entity the corpus genuinely only ever mentions in
# passing -- so that after a fair look the rule stands down and the gate accepts the incidental
# description in hand rather than grinding to MAX_STEPS. Mirrors DEFERRED_MAX_READS in spirit.
DEFN_MAX_PUSH = 6

def _n_distinct_files(evidence):
    """How many DISTINCT source documents the collected evidence spans -- the breadth counter that
    synthesis mode budgets against (whereas len(evidence) counts individual pieces)."""
    return len({(e.metadata.get("source_file") or e.path or e.name) for e in evidence})

# ---- REJECT-DECAY (navigation): stop the search from grinding a barren decoy subtree ----
# Root failure this fixes: the score-priority teleport frontier let ONE subtree that scored high on
# spurious SURFACE overlap (a 'Laboratory Equipment' doc mentioning 'personal computers' for a
# 'personal device policy' question) monopolise the whole step budget — every rejected section read
# teleported straight to ANOTHER section of the same failed document, so the agent ground through one
# decoy while genuinely-unexplored top-level branches (and the low-scored-but-correct folder sitting
# in the reserve tier) were never reached within MAX_STEPS. The fix is general and corpus-agnostic:
# when a reached node is REJECTED, demote every remaining frontier/reserve entry under its enclosing
# FILE into the deprioritised reserve tier and multiply their scores by REJECT_DECAY. Repeated rejects
# in the same subtree COMPOUND the decay, so a subtree that keeps yielding nothing quickly sinks below
# fresh, never-tried branches and the search is pushed toward BREADTH. Nothing is ever removed, so
# recall is preserved — a demoted sibling stays reachable, just deprioritised behind untried territory.
REJECT_DECAY = 0.30
# ---- CLUSTER-REJECT DECAY (navigation): abandon a barren REGION, not just one file at a time ----
# Per-file REJECT_DECAY has a blind spot: when a corpus holds a large HOMOGENEOUS CLUSTER of
# look-alike documents that all score high on surface topicality (e.g. a whole folder of near-
# duplicate checklists for a "when is X done" question), rejecting one file only sinks THAT file's
# subtree — the frontier stays full of its equally-high-scoring siblings, so the agent teleports
# from one decoy to the next and exhausts MAX_STEPS inside the cluster while a lower-scored but
# CORRECT branch in a different part of the tree is never reached. The general fix: on every barren
# reject, bubble the reject up the ancestor FOLDERS; once any folder has yielded CLUSTER_REJECT_N
# barren files, demote that whole folder's remaining subtree at once. A region that keeps returning
# nothing thus sinks below never-tried branches as a WHOLE, pushing the search out to structurally-
# distant territory. Purely structural (no keywords, no per-file priorities) and recall-preserving:
# demoted entries move to the reserve tier and stay reachable once the frontier is dry.
CLUSTER_REJECT_N = 3

# ---- REJECT-SWEEP (navigation): read a reached file's OTHER sections before condemning it ----
# Companion to REJECT_DECAY, fixing its blind spot. Greedy descent commits to ONE section of a file
# (chosen on name/summary/preview), so when that single section is rejected the whole file is
# abandoned -- even though the answer may sit in a SIBLING section the ranker under-scored. This is
# the exact miss when a file is reached via a CONFIDENT folder-level score but its per-section scores
# are all low/uncertain (the ranker knew the file was right but not which section). A take already
# triggers a full intra-file triage sweep of the sibling sections; a reject did not, so exploration
# was asymmetric and correctly-navigated files were dropped on the first unlucky section pick. Now: on
# a reject, if the enclosing file was reached confidently (line score >= REJECT_SWEEP_MIN) and has
# unread sections, we run the SAME bounded triage sweep the take-path uses before demoting anything;
# any answer-bearing sibling is then collected. A genuinely barren decoy costs only one triage pass
# (its sections are skipped, not taken) and is demoted as before -- so decoy-avoidance is intact.
REJECT_SWEEP_MIN = 0.60   # a reject triggers a same-file sweep only when the file was reached this
                          #   confidently; weaker files are demoted immediately, as before

# ---- NEAR-TIE GUARD (navigation): don't answer from one document while an equally-ranked sibling
# document is still unread. Root failure this fixes: when the ranker returns a TIGHT CLUSTER of
# near-tied candidate DOCUMENTS (a 'how to report X' question whose folder holds a form-template doc
# at 0.97 and the actual procedure doc at 0.96), greedy descent commits to the single top one and the
# intra-file sweep + deferred recovery then drain every 'not enough' signal into MORE sections of that
# one file. If the top file is a plausible decoy, the sufficiency gate eventually accepts its
# partial/wrong content and the agent answers WITHOUT ever visiting the sibling the ranker judged
# almost exactly as relevant -- the sibling that actually held the answer. A near-tie is the ranker's
# own signal that it could not separate the right file from a decoy, so committing to one and stopping
# is premature. The guard: before the agent is allowed to STOP and answer, if the best UNEXPLORED
# frontier entry belongs to a DIFFERENT source document and scores within TIE_WINDOW of the score we
# committed to on the current line, visit it first (so it is actually read and compared) instead of
# answering. Purely relative (no keywords, no per-file priorities) and bounded by TIE_MAX_EXPLORE so a
# genuinely single-document question pays at most a couple of extra visits.
TIE_WINDOW      = 0.05   # an unexplored other-document frontier entry within this of the committed
                         #   line score counts as a near-tie the ranker could not confidently break
TIE_MAX_EXPLORE = 3      # max extra near-tied documents visited before answering (per question).
                         #   This is a CEILING, not a quota: a contrast excursion that comes back
                         #   BARREN (read, rejected, evidence unchanged) ends the search — it just
                         #   confirmed the committed account — so the full budget is spent only
                         #   while contrasts keep CHANGING the evidence (see FAILED CONTRAST IS
                         #   CONFIRMATION in run_agent).
# SINGLE-SOURCE DECOY GUARD: the failure this fixes is subtler than a near-tie. The ranker was
# CONFIDENT (one decoy at ~0.96, the true file far lower) but WRONG: it committed to a topically
# adjacent document that literally mentions the queried term, took a section that reads on-topic,
# and the sufficiency gate — which only ever sees the current document's evidence — rubber-stamped
# that single-source partial answer. A lone strong topical match is exactly where a decoy hides, so
# when EVERY piece of collected evidence comes from ONE document we widen the pre-answer check: the
# plausible runner-up OTHER-documents (not just 0.05 near-ties) each get one verification pass before
# we commit. Still purely relative to the committed score, above the noise floor, and hard-bounded by
# TIE_MAX_EXPLORE, so a genuinely single-document question pays only a couple of extra visits and no
# corpus keywords or per-file priorities are introduced.
SINGLE_SRC_WINDOW = 0.5  # width of that widened band, used ONLY while the answer is single-sourced

def _src_of(node):
    """Canonical source-document identity for a node — the same rule whole_unit() uses to name a
    document. Two nodes with the same _src_of belong to the same underlying file."""
    return (next((c.metadata.get("source_file") for c in all_chunks(node)
                  if c.metadata.get("source_file")), None) or node.path or node.name)

# ---- CONTRAST VERIFICATION (navigation): spend the pre-answer look on a DIFFERENT account, not on
# another copy of the one already held. ------------------------------------------------------------
# The guard above is the right idea executed with the wrong selector. It picks the alternative to
# verify by RAW SCORE — `next(f for f in frontier if different-source)` — but score is produced by the
# same ranker, reading the same question, under the same framing that produced the first commitment.
# So the highest-scoring unexplored candidate is almost always a SIBLING OF THE COMMITTED ACCOUNT, and
# the guard spends its whole budget CONFIRMING the hypothesis it was built to challenge.
#
# The signature (seen whole in one run): the search commits to the general CATEGORY a question falls
# under rather than the specific subject it names — a governing 'non-conformance' procedure for a
# question about one process's own failure handling — the gate accepts that single-sourced account,
# and the guard then verifies three more documents OF THAT SAME CATEGORY (its records folder, its
# forms folder, one of its form templates), each scoring at the top of the frontier precisely because
# it restates the committed framing. All three reject, add nothing, and exhaust TIE_MAX_EXPLORE. The
# document that addressed the named subject head-on sat mid-frontier, in the very folder the search
# had entered first, and was never once looked at — because the guard only ever considers ONE
# candidate, the top-scored one.
#
# Two changes, both decisions rather than substrate:
#   1. The guard shortlists EVERY eligible alternative (same eligibility band as before), not just the
#      top-scored one, so a mid-ranked candidate is at least visible to the decision.
#   2. The agent CHOOSES among them, on their own words and contents, asked for the one that would
#      CONTRADICT or SHARPEN the account in hand — and told explicitly that another record/form/copy
#      of the same account cannot change it however high it scores, so a lower-scored document about
#      the named subject is the better use of the one look. It may also answer -1 ("all of these would
#      just repeat what I hold"), which is a genuine agentic stop the score-only guard could not make.
# The chosen target is then actually TELEPORTED TO (see `forced_next`): the old guard picked `near`,
# used it only as permission to keep going, and then let score-priority send the search wherever it
# liked — usually straight back into the committed family.
# No index, no similarity search, no keywords: this reuses the same preview/summary/contains material
# the ranker already builds, and the LLM makes the call.
ALT_SHORTLIST = 8     # eligible alternatives shown to the contrast chooser (was: only the top one)
ALT_PREVIEW   = 400   # chars of a candidate's ACTUAL words shown, so the choice is made on content

def _alt_entry(i, f, query, counter):
    """One alternative's block for the contrast chooser: score + name + where it sits + its own words
    (and, for a folder, the names it contains) — the same signals the ranker routes on."""
    c = f["node"]
    line = "[%d] (score %.2f) %s" % (i, f["score"], c.name)
    par = _NODES.get(_PARENT.get(c.node_id, ""))
    if par is not None and par.node_type != "root":
        line += "   [inside: %s]" % clip(par.name, 40)
    summ = clip(c.summary or "", 200)
    if summ: line += "\n      summary: " + summ
    try:
        prev, multi = _content_preview(c, ALT_PREVIEW, query=query, counter=counter)
    except Exception:
        prev, multi = "", False
    if prev and not multi: line += "\n      text: " + prev
    cl = _contains_line(query, c, counter)
    if cl: line += "\n      " + clip(cl, 300)
    return line

def _choose_alternative(query, evidence, cands, memory, counter):
    """Which unexplored candidate would most CHANGE the answer? Returns (entry|None, why).
    None means the agent judged that every listed candidate would merely repeat the account in hand."""
    have_srcs, have = [], []
    for e in evidence:
        s = str(e.metadata.get("source_file") or e.path or e.name)
        if s not in have_srcs: have_srcs.append(s)
        have.append(full(e.content or e.summary))
    acct = clip("\n\n".join(have), 2000) or "(nothing)"
    lines = "\n".join(_alt_entry(i, f, query, counter) for i, f in enumerate(cands))
    prompt = (
        "You are about to answer the question using only the evidence below. Before you commit, you "
        "get ONE look at an unexplored document. Spend it on whichever would most likely CONTRADICT "
        "or SHARPEN what you have.\n\n"
        "Why this matters: the evidence you hold may not be the answer but a plausible NEIGHBOUR of "
        "it — the general category the question falls under, or a record, form or instance of that "
        "category. Such material reads like a complete answer, while the document that actually "
        "governs the SPECIFIC subject the question names states the fact more precisely, and "
        "sometimes differently. You cannot detect that from the evidence itself; only a different "
        "document can show it.\n\n"
        f"QUESTION: {query}\n\n"
        f"EVIDENCE YOU HOLD (from: {', '.join(clip(s, 60) for s in have_srcs)}):\n{acct}\n\n"
        f"UNEXPLORED CANDIDATES:\n{lines}\n\n"
        "Choose the ONE candidate most likely to change or sharpen your answer:\n"
        "- Prefer whatever is most likely to be ABOUT the exact subject or process the question names, "
        "and to state the asked fact for it directly.\n"
        "- Do NOT choose another document of the SAME KIND as one you already read — a further record, "
        "form, template, or another copy of the same procedure. More of the same account cannot change "
        "that account, however high its score. A LOWER-scored candidate that addresses the named "
        "subject head-on is a better use of this one look than a high-scored sibling of what you "
        "already have.\n"
        "- AGREEMENT IS NOT CONTRAST: a candidate whose shown text states the SAME value or fact you "
        "already hold cannot contradict or sharpen it — reading it would merely corroborate, and "
        "same-kind documents about sibling entities routinely repeat the same figures, so agreement "
        "proves nothing about the subject the question names. If your evidence's tie to that exact "
        "subject is unproven, spend the look on the candidate most likely to be about the NAMED "
        "subject itself — often from a part of the tree not yet searched — rather than on any "
        "restatement, however topical its wording.\n"
        "- Judge on each candidate's own words and contents where shown, NOT on its score. The scores "
        "were produced by the same reading of the question that led you to the evidence above, so they "
        "favour more of the same.\n"
        "Answer -1 ONLY if every candidate listed would merely repeat the account you already hold.\n"
        'Reply ONLY json: {"reasoning":"<1 short sentence>", "choice":<index or -1>}')
    raw = llm(prompt, counter, num_predict=512, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    ci, why = None, ""
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            why = str(d.get("reasoning", "")).strip()
            ci = int(d.get("choice"))
        except Exception:
            ci = None
    if ci is None:
        mm = re.search(r'"?choice"?\s*[:=]\s*(-?\d+)', s)
        if mm: ci = int(mm.group(1))
    if ci is None:
        # Unparsed: fall back to the OLD behaviour (check the top-scored alternative) rather than
        # silently committing to the account in hand — a missing decision must not become a stop.
        return cands[0], "[unparsed contrast response; checking the top-scored alternative]"
    if 0 <= ci < len(cands): return cands[ci], why
    return None, why or "no listed candidate could give a different account"

# The standing directive for a contrast check, pinned into working memory (read by the ranker and
# every evidence judge, never by the answer prompt). Without it the check is undermined the moment it
# starts: memory is full of the CONTENT of the account already taken, so descending into whatever the
# chooser picked re-ranks its children against that account and walks straight back to more of the
# same. This re-aims ranking at the named subject for the duration of the check. Cleared as soon as
# the check resolves. It owns its own tag, so it cannot be evicted by the recency cap and does not
# collide with the residual note.
_ALT_TAG = _PIN + "CONTRAST CHECK."
def _set_contrast(memory, evidence):
    memory[:] = [m for m in memory if not m.startswith(_ALT_TAG)]
    if not evidence: return
    srcs = []
    for e in evidence:
        s = str(e.metadata.get("source_file") or e.path or e.name)
        if s not in srcs: srcs.append(s)
    memory.insert(0, _ALT_TAG + (
        " The facts below already give ONE account of the answer, drawn from: "
        + "; ".join(clip(s, 60) for s in srcs[:4]) + ". You are NOT gathering more of it — you are "
        "checking whether a DIFFERENT document states the asked fact for the exact subject the "
        "question names, more precisely or differently. So score a candidate HIGH only if it would be "
        "ABOUT that subject and could state that fact directly. Score LOW any candidate that is "
        "another record, form, template, instance or restatement of the account already held, however "
        "topical it looks — re-reading the same account cannot change it."))

# ---- RESIDUAL-AIMED TELEPORT (navigation): choose the next jump against what is MISSING, not
# against a score computed before anything was known. ---------------------------------------------
# The failure class: the gate reports 'not enough' and names the gap in its own words — and the search
# answers that verdict by popping the highest-SCORED frontier entry. But every frontier score was
# produced by the ranker reading the WHOLE question BEFORE any evidence existed, so it ranks a
# candidate on how much it RESEMBLES THE QUESTION — and the candidates that resemble it most are the
# same kind of document whose partial account is ALREADY in hand. So each teleport lands on another
# sibling of what was just read, returns the same partial fact, and the gate names the SAME gap again;
# the search spends its entire budget re-confirming the half it has settled. Meanwhile the one
# candidate that would close the gap sits mid-frontier at a lower score — eligible at every jump,
# chosen at none — because a stale sort never revisits its own belief. The signature is a run of
# 'not enough' verdicts whose missing-part never changes while the frontier's top keeps churning.
#
# `_set_residual` already pins the gap into working memory, but memory only reaches the ranker when it
# scores the CHILDREN of a node the search descends into. Frontier entries keep the score they were
# born with and are never re-judged, so the teleport — the single most consequential navigation choice
# in the loop, and the only one made after the agent knows what it is missing — is not a decision at
# all. That is the gap this closes: the agent's own sufficiency verdict now feeds the agent's own next
# choice, instead of being spent solely on whether to keep going.
#
# The remedy: when the gate names a gap, the AGENT picks the jump. The top frontier entries are
# shortlisted with exactly the material the ranker already routes on (score, position, summary, the
# candidate's own words, the names it contains) alongside the gap and the documents already read, and
# the agent chooses the one that would STATE the missing fact — told explicitly that another document
# of the kind already read will return the same partial account, so a lower-scored candidate that
# speaks to the gap is the better jump. No index, no similarity search, no keywords, no per-file
# priorities: this reuses the same previews the ranker builds and the LLM makes the call. It no-ops
# when the gate names no gap (a settled or single-part question never reaches it) or when the frontier
# holds nothing to choose between, and ANY parse failure falls back to the top-scored entry — i.e.
# exactly the old score-priority behaviour.
RESID_SHORTLIST = 8   # frontier entries offered to the residual-aimed chooser

def _choose_by_residual(query, missing, evidence, cands, memory, counter):
    """Pick the frontier entry most likely to settle the parts the gate itself says are still MISSING.
    Returns (entry, why). Falls back to the top-scored entry when the model returns nothing usable, so
    a missing decision degrades to score-priority rather than to a bad jump."""
    srcs = []
    for e in evidence:
        s = str(e.metadata.get("source_file") or e.path or e.name)
        if s not in srcs: srcs.append(s)
    have = clip("\n\n".join(full(e.content or e.summary) for e in evidence), 1600) or "(nothing yet)"
    gap = "; ".join("(%d) %s" % (i, clip(p, 120)) for i, p in enumerate(missing[:PARTS_MAX], 1))
    lines = "\n".join(_alt_entry(i, f, query, counter) for i, f in enumerate(cands))
    prompt = (
        "You are navigating a document tree and have just judged the evidence you hold INCOMPLETE. "
        "Choose the ONE unexplored candidate to go to next.\n\n"
        f"QUESTION: {query}\n\n"
        f"STILL MISSING (your own verdict on the evidence below): {gap}\n\n"
        f"DOCUMENTS ALREADY READ: {', '.join(clip(s, 60) for s in srcs) or '(none)'}\n"
        f"WHAT THEY GAVE YOU:\n{have}\n\n"
        f"UNEXPLORED CANDIDATES:\n{lines}\n\n"
        "Choose on ONE criterion: which candidate is most likely to STATE the missing part above.\n"
        "- Judge on each candidate's own words, its contents, and what KIND of document it is — NOT on "
        "its score. The scores were produced by reading the WHOLE question before you had any "
        "evidence, so they rank candidates by how much they resemble the question, and the ones that "
        "resemble it most are the same kind of document that already gave you the part you HAVE. Going "
        "to another one returns the same partial account and leaves the same gap open.\n"
        "- The parts you already hold are settled; re-confirming them is worth nothing. A LOWER-scored "
        "candidate that plausibly states the MISSING part is a better jump than a high-scored sibling "
        "of what you have already read.\n"
        "- Prefer whatever would state the missing fact directly and specifically for the subject the "
        "question names — the document, table or specification that GOVERNS it — over one that "
        "discusses the topic generally around it. General overviews restate what you have; specific "
        "criteria, thresholds, values and conditions are what is missing.\n"
        'Reply ONLY json: {"reasoning":"<1 short sentence>", "choice":<index>}')
    raw = llm(prompt, counter, num_predict=512, temperature=0, think=False)
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    ci, why = None, ""
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            why = str(d.get("reasoning", "")).strip()
            ci = int(d.get("choice"))
        except Exception:
            ci = None
    if ci is None:
        mm = re.search(r'"?choice"?\s*[:=]\s*(-?\d+)', s)
        if mm: ci = int(mm.group(1))
    if ci is None or not (0 <= ci < len(cands)):
        return cands[0], "[unparsed residual choice; falling back to score-priority]"
    return cands[ci], why

# ---- LEXICAL CONTENT SEED (navigation recall for distinctive rare terms) ----
# The failure class this fixes: a question turns on a DISTINCTIVE term (e.g. a named stakeholder role)
# that appears in a document's BODY but in no folder/file NAME or SUMMARY. Top-down LLM routing only
# ever sees names + summaries + child-name lists, so it is structurally blind to that term: every
# folder scores a uniformly low ~0.05 ("none of these mention it"), greedy descent commits to a
# generic decoy, teleports chase equally-generic decoys, and the answer-bearing file is NEVER reached
# -- exactly where the vector baseline, which retrieves on body content, wins. The remedy restores a
# content-grounded recall signal WITHOUT embeddings (the code deliberately avoids per-chunk embed()
# for cost, see _content_preview): a one-pass, cached, IDF-weighted LEXICAL scan of every chunk's
# actual words. Files whose bodies contain the question's RARE terms are seeded onto the teleport
# frontier, so when top-down routing flounders the search can still reach them. It is purely additive
# (extra frontier entries the LLM must still confirm by reading), gated on a genuinely distinctive
# match, and capped in count and score, so confident LLM descents are never overridden and generic
# (all-common-term) questions seed nothing and behave exactly as before.
LEX_SEED_K       = 3      # max files seeded from the lexical scan
LEX_SEED_CAP     = 0.55   # a full distinctive-term match seeds at this frontier score (below a
                          #   confident LLM descent, well above the barren decoys the search grinds)
LEX_RARE_DF_FRAC = 0.15   # a query token is "distinctive" only if it occurs in <= this fraction of
                          #   documents; seeding fires only when a matched token is this rare, so
                          #   common-word questions add nothing

_LEX_INDEX = {"root": None}
def _build_lex_index(root):
    """One cached, LLM-free pass over the whole tree: per source document, the SET of body tokens and
    a frontier-seedable node for that document, plus corpus document-frequency per token (for IDF).
    Keyed by source_file, the same document identity _src_of/whole_unit use."""
    if _LEX_INDEX.get("root") is root:
        return _LEX_INDEX
    src_tokens, src_chunk = {}, {}
    for ch in all_chunks(root):
        src = ch.metadata.get("source_file") or ch.path
        if not src: continue
        if src not in src_tokens:
            src_tokens[src] = set(); src_chunk[src] = ch
        src_tokens[src].update(_toks(ch.content or ch.summary))
    df = Counter()
    for toks in src_tokens.values():
        for t in toks: df[t] += 1
    src_node = {}                                     # resolve the document node once per document
    for src, ch in src_chunk.items():
        f = _enclosing_file(ch)
        if f is not None: src_node[src] = f
    _LEX_INDEX.clear()
    _LEX_INDEX.update(root=root, src_tokens=src_tokens, src_node=src_node,
                      df=df, N=max(1, len(src_tokens)))
    return _LEX_INDEX

def _lexical_seed_files(query, root, k=LEX_SEED_K):
    """Return [(document_node, seed_score, reason)] for the files whose BODY best matches the
    question's RARE terms. [] when the question has no distinctive term or nothing matches -- so
    generic questions are untouched. Deterministic, no LLM/embeddings, one cached corpus pass."""
    idx = _build_lex_index(root)
    N, df = idx["N"], idx["df"]
    idf = {t: math.log(1.0 + N / df[t]) for t in set(_toks(query)) if df.get(t)}
    total = sum(idf.values())
    if total <= 0: return []
    rare_cut = max(1, int(N * LEX_RARE_DF_FRAC))
    scored = []
    for src, toks in idx["src_tokens"].items():
        node = idx["src_node"].get(src)
        if node is None: continue
        hit = set(idf) & toks
        # require at least one genuinely distinctive (rare) matched term, else this is generic overlap
        if not hit or not any(df[t] <= rare_cut for t in hit): continue
        coverage = sum(idf[t] for t in hit) / total
        scored.append((node, coverage))
    if not scored: return []
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for node, cov in scored[:k]:
        score = min(LEX_SEED_CAP, NOISE_FLOOR + cov * (LEX_SEED_CAP - NOISE_FLOOR))
        out.append((node, score, "lexical body match on distinctive term(s)"))
    return out

# ---- BREADTH ESCAPE (navigation): recover a top-level region a confident-but-wrong root decision banished ----
# Failure class: the answer-bearing file sits in a top-level region the root ranker scored BELOW the
# noise floor (a plausible-but-wrong routing heuristic sent the search into a sibling region), so that
# region goes to the reserve tier and is only ever drawn 'when the frontier is empty' -- which, on a
# large corpus within MAX_STEPS, never happens. Meanwhile the greedy line accumulates topically-adjacent
# MENTIONS (documents that USE the queried term but do not DEFINE/answer it): each is TAKEN, so
# reject-decay never fires to push the search out, yet the sufficiency gate keeps saying 'not enough'.
# The search grinds the decoy region to the step cap and the correct region is never reached -- exactly
# where the vector baseline, retrieving on body content, wins. The general remedy: treat a persistent
# 'not enough' (after evidence already spans several DISTINCT documents) as the signal that the explored
# regions do not hold the answer, and on the next teleport jump to the best-scoring frontier/reserve
# entry in a top-level region NOT yet explored. Purely structural, recall-preserving, and bounded by the
# finite number of regions; it fires only after the topical region has had a fair multi-document look.
STALL_TRIGGER  = 3   # consecutive 'not enough' verdicts (with multi-doc evidence) before a breadth escape
STALL_MIN_DOCS = 2   # ...and only once evidence spans at least this many distinct documents, so the
                     #   confident region is given a fair look and single-document questions never divert

def _region_of(node_id):
    """The top-level region (root-child ancestor) a node lives under -- the unit the breadth escape
    reasons about. A root child returns itself; the root returns itself."""
    nid = node_id
    while nid in _PARENT:
        pid = _PARENT[nid]; parent = _NODES.get(pid)
        if parent is None or parent.node_type == "root":
            return nid
        nid = pid
    return nid

# ---- REGION-DIVERSE SHORTLISTS (navigation): let an agentic teleport choice actually SEE the rest
# of the tree ---------------------------------------------------------------------------------------
# The failure class: the root ranks one region confidently (0.90 vs 0.45 for everything else), the
# search descends there, and NEVER LEAVES — every teleport lands on another sibling of what was just
# read, the whole step budget is spent inside the first pick, and a question whose answer is spread
# across other regions is answered from the one region that happens to hold part of it. The root's
# runner-ups sit on the frontier above the noise floor for the entire run, eligible at every jump,
# chosen at none. This is exactly where a content-retrieving baseline wins, so the remedy has to be a
# better DECISION, not an index.
#
# Why the escape hatches could not fix it themselves. Both genuinely agentic teleport choices — the
# contrast check and the residual-aimed teleport — are handed their options as a top-K-BY-SCORE slice
# of the frontier. But a frontier score is not comparable across the tree. A candidate inside the
# committed region was scored while the ranker was THERE: reading its text previews, with a working
# memory full of that region's account and (once a gate has run) a residual note drawn from it. A
# top-level sibling was scored cold, on its name and child-name list alone, before anything at all was
# known. The committed region's descendants therefore outscore every unexplored region almost by
# construction, and each descent pushes more of them on — faster than teleports consume them. So the
# top-K slice is always K siblings of what was just read, and the agent, whose judgement is fine, can
# only ever pick more of the same. Its option set was pre-filtered by the exact bias the decision
# exists to correct.
#
# The single-source decoy guard shows the bug in miniature: it deliberately widens its eligibility
# band to SINGLE_SRC_WINDOW because "a lone strong topical match is where a decoy hides" — and then
# clips the result to the top ALT_SHORTLIST by score, discarding every candidate the widened band was
# opened to admit. The band was doing its job; the truncation undid it silently.
#
# The fix changes WHICH OPTIONS THE AGENT SEES, and nothing else: no single top-level region may take
# more than SHORTLIST_REGION_QUOTA of a shortlist's slots, so the rest go to the best candidate each
# OTHER region has to offer. The agent still decides, from the same previews / summaries /
# contains-lines, under the same prompts — both of which already tell it to judge on content rather
# than score and that a lower-scored candidate speaking to the gap beats a high-scored sibling of what
# it has read. When the committed region IS right it keeps winning: its quota slots carry its
# strongest candidates and it still leads the list. Nothing here is scored, matched, indexed or
# ordered by content — the only input is the tree's own structure. Recall is untouched: no entry
# leaves the frontier, and the list is topped back up to k in score order, so with fewer than k
# regions to spread over this returns the old top-K exactly.
def _diverse_shortlist(entries, k, quota=SHORTLIST_REGION_QUOTA):
    """Score-ordered shortlist of <= k frontier entries in which no ONE top-level region may hold more
    than `quota` slots, so the options span the tree rather than one subtree. Any slots the quota
    leaves empty are filled back in by plain score order."""
    ranked = sorted(entries, key=lambda f: f["score"], reverse=True)
    out, used = [], Counter()
    for f in ranked:
        if len(out) >= k: break
        r = _region_of(f["node"].node_id)
        if used[r] >= quota: continue
        out.append(f); used[r] += 1
    if len(out) < k:            # fewer than k regions to spread over -> degrade to the old top-K
        chosen = {id(f) for f in out}
        for f in ranked:
            if len(out) >= k: break
            if id(f) not in chosen: out.append(f)
    return sorted(out, key=lambda f: f["score"], reverse=True)

def run_agent(q, counter, live=True):
    ink=(lambda s: print(s,end="",flush=True)) if live else (lambda s: None)
    nav_q=q.stem                                   # the agent only ever sees the question, never the choices
    memory=[]; evidence=[]; seen=set(); visited=set(); deferred=[]   # deferred: unread same-file candidates
    reject_folders=Counter()   # barren-file tally per ancestor folder (CLUSTER-REJECT DECAY)
    trail=["root"]; trace=[]; teleports=0; steps=0
    frontier=[]        # global priority queue: {"node","score","reason","from"} — runner-ups from every level
    reserve=[]         # sub-NOISE_FLOOR runner-ups: never explored while anything better exists, but kept
                       # so recall is preserved — drawn from only when the frontier is empty and evidence
                       # is still insufficient
    last_take_file=None  # the file of the most recent take; where intra-file exploration looks first
    descent_top=0.0      # NEAR-TIE GUARD: the best relevance score committed to on the CURRENT line
                         # (max descend score since the last teleport, seeded by a teleport's own
                         # target score). Frontier near-ties are measured against it.
    tie_explores=0       # how many other-documents we've already diverted to verify before answering
    forced_next=None     # CONTRAST VERIFICATION: the frontier entry the agent itself chose as the
                         # alternative worth checking. The old guard chose one and then let
                         # score-priority teleport somewhere else entirely — usually back into the
                         # family it had just committed to — so its choice never actually steered the
                         # search. The next teleport now goes HERE.
    contrast_active=False # a dispatched contrast excursion is still in flight: forced_next was
                         # honoured and no read has completed since. Cleared at the first gate
                         # evaluation after the excursion's read (see FAILED CONTRAST IS CONFIRMATION)
                         # and by the collapse abort in the descend section.
    contrast_ev_mark=0   # len(evidence) at dispatch — if the excursion changes nothing, the
                         # committed account has survived the adversarial check the agent itself
                         # designed, and the search answers instead of re-checking until the
                         # TIE_MAX_EXPLORE counter runs out.
    defn_pushes=0        # times the entity-definition provenance rule has overridden an otherwise-
                         # sufficient verdict; bounded by DEFN_MAX_PUSH so it cannot grind forever
    committed_scores={}  # src_file -> the frontier/descent score at which we committed to each DOCUMENT
                         # that produced evidence. The near-tie stop guard anchors on the WEAKEST of
                         # these (same scale as the frontier's folder/file scores) instead of the
                         # section-level descent_top, so a peer-scoring sibling document is never
                         # judged "too low" and skipped.
    line_entry_score=1.0 # score by which we reached the CURRENT line's document — a folder/file score,
                         # updated on each folder/file descent and on every teleport; recorded into
                         # committed_scores when this line yields evidence.
    stall=0              # BREADTH ESCAPE: run of 'not enough' verdicts while stuck in the current region
    entered_regions=set()# top-level regions (root children) the search has descended into or jumped to
    evidence_regions=set() # top-level regions evidence has come from. The pre-answer contrast check
                         # treats same-region evidence as ONE account (see single_account below):
                         # two documents from the same region restating the same fact are one route's
                         # findings, not independent verification.
    nosignal_run=0       # consecutive all-below-floor FOLDER rankings on the current line (see
                         # NOSIGNAL_ABORT); reset whenever a folder ranking finds real signal and on
                         # every teleport (a new line starts fresh)
    residual=[]          # RESIDUAL-AIMED TELEPORT: the gate's own list of parts the evidence does not
                         # yet settle, as of the most recent verdict. [] when the evidence is
                         # sufficient or no gate has run, which is what makes the mechanism no-op.

    def _collect_node(nd, whole, clip_chars=None):
        if nd.node_id in seen: return False
        item = whole_unit(nd) if whole else nd
        if clip_chars and item.content and len(item.content) > clip_chars:
            item = replace(item, content=clip(item.content, clip_chars))
        evidence.append(item); seen.add(nd.node_id)
        # remember the score by which we committed to THIS document (first take wins); the stop guard
        # uses the weakest such score as its "is an unread sibling still worth a look?" baseline.
        committed_scores.setdefault(_src_of(nd), line_entry_score)
        evidence_regions.add(_region_of(nd.node_id))
        # if we took a whole section/file, mark its descendant chunks as seen too so intra-file
        # exploration never re-collects text already contained in the unit we just took.
        if whole:
            for ch in all_chunks(nd): seen.add(ch.node_id)
        if item.content: _add_memory(memory, clip(item.content,600))
        return True

    def _push_frontier(items, src_name):
        # items: [(child, score, reason), ...] runner-ups to remember for a possible teleport.
        # Sub-floor candidates go to the reserve tier instead of the active frontier: they are the
        # noise the long runs were grinding through, but keeping them reachable preserves recall.
        for c, sc, rs in items:
            if c.node_id in visited: continue
            if any(f["node"].node_id==c.node_id for f in frontier) or \
               any(f["node"].node_id==c.node_id for f in reserve): continue
            entry = {"node":c,"score":sc,"reason":rs,"from":src_name}
            (frontier if sc >= NOISE_FLOOR else reserve).append(entry)

    def _penalize_subtree(root_id):
        # A reached node under `root_id` was just rejected. Demote the whole subtree so the search
        # stops re-entering it: move its frontier entries into reserve (decayed) and decay any of its
        # entries already in reserve. Compounds across consecutive rejects in the same subtree.
        moved = 0
        for f in reserve:
            if _node_under(f["node"].node_id, root_id): f["score"] *= REJECT_DECAY
        keep = []
        for f in frontier:
            if _node_under(f["node"].node_id, root_id):
                f["score"] *= REJECT_DECAY; reserve.append(f); moved += 1
            else:
                keep.append(f)
        frontier[:] = keep
        return moved

    def _pop_frontier():
        if frontier:
            frontier.sort(key=lambda f: f["score"], reverse=True)
            return frontier.pop(0)
        if reserve:   # last resort: every promising candidate is exhausted, dip below the floor
            reserve.sort(key=lambda f: f["score"], reverse=True)
            ink("\n    (active frontier exhausted; drawing from sub-floor reserve)")
            return reserve.pop(0)
        return None

    def _gate(_unread):
        """The sufficiency gate, plus the entity-definition provenance budget. `definitional` is passed
        through only while that rule still has pushes left: once it has forced the search onward
        DEFN_MAX_PUSH times and no document that is actually ABOUT the entity has turned up, the corpus
        evidently holds none within reach, so the rule stands down and the gate is allowed to accept the
        incidental description in hand rather than running to the step cap."""
        nonlocal defn_pushes
        d_on = definitional and defn_pushes < DEFN_MAX_PUSH
        s, w = _evidence_would_answer(nav_q, evidence, counter, unread=_unread, polarity=polarity,
                                      definitional=d_on, recency=recency)
        if d_on and not s and GATE_LAST.get("basis") == "incidental":
            defn_pushes += 1
        return s, w

    # one-time agentic query decomposition: the ranker scores nodes by name+summary, and generic
    # documents can outrank specific procedural ones when scored on gestalt similarity alone (a
    # recurring diagnosed failure). Extracting what the question asks into working memory — which the
    # ranker and every judge already see — makes scoring coverage-aware. This is the vector-free
    # version of the diagnosers' recurring 'multi-concept relevance' suggestion.
    # The call now returns the question's distinct PARTS rather than a flat concept list: on a stem
    # that asks one thing that is the same signal as before, and on a stem that asks several it is
    # additionally the ledger the residual steering and the part-coverage hand-off settle up against.
    # Still ONE call, so this is cost-neutral.
    parts, subject = [], ""
    try:
        parts, subject = _decompose_parts(nav_q, counter)
        if parts:
            _add_memory(memory, "The answer must cover: " + "; ".join(parts))
            ink("  parts: %s\n" % " | ".join(parts))
    except Exception: pass

    # SPECIFIC-SUBJECT ANCHOR: when the question asks a fact about ONE particular thing, pin its
    # identity into working memory — read by the ranker and every evidence judge, never by the answer
    # prompt. Without it every decision keys on the question's ATTRIBUTE words alone, and on a corpus
    # that records the same attribute for many same-kind entities (several backup units, instruments,
    # rooms, versions — each with its own value) the search commits to whichever sibling states the
    # attribute most prominently, takes conflicting values from several of them, and never reaches the
    # document about the entity actually named. Pinned (_PIN) so it survives the recency cap and
    # steers every ranking/judging decision for the whole run; the residual and contrast directives
    # clear only their own tags, so they cannot evict it. Inserted directly (not via _add_memory,
    # whose 500-char clip could truncate it). No-ops on questions about a category/process in general,
    # where the decomposition returns an empty subject.
    if subject:
        memory.insert(0, _PIN + (
            "SPECIFIC SUBJECT. The question asks about ONE particular thing: " + clip(subject, 160) +
            " — that exact one, not its kind in general. The corpus very likely records the same KIND "
            "of fact for OTHER same-kind entities (other units, instruments, models, sites, versions) "
            "— those are DECOYS: the identical attribute of a DIFFERENT entity is not the answer, "
            "however exactly its wording matches the question. Score HIGH branches and documents "
            "identifiably about this subject (its name, model, location, or the system it serves); "
            "score LOW material stating the asked attribute for something else; and treat a value "
            "that cannot be tied to this subject as NOT settling the question."))
        trace.append({"event": "subject_anchor", "subject": clip(subject, 160)})
        ink("  subject anchor: %s\n" % clip(subject, 80))

    # POLARITY / UNIVERSAL steering: when the question asks whether a relationship holds always /
    # necessarily, the answer lives in the governing DEFINITION/CONDITIONS, not in example records.
    # Note it in working memory (which the ranker and every evidence judge read, but NOT the final
    # answer prompt) so instance forms that merely mention both entities score DOWN and the defining
    # policy/procedure scores UP. The same flag also relaxes the sufficiency gate below.
    polarity, polarity_why = _is_polarity_question(nav_q)
    if polarity:
        _add_memory(memory,
            "This question asks whether a relationship holds ALWAYS / NECESSARILY / in every case, OR "
            "whether a specific item EXISTS / is available. The answer is in a GOVERNING document — the "
            "DEFINITION, PURPOSE or CONDITIONS of the process, or the INVENTORY / REGISTER / SCOPE / "
            "ITEM-LIST that would ENUMERATE the item's category — and is frequently NEGATIVE ('no, not "
            "always' / 'no, we do not have one'). Prefer such a governing/enumerating document and treat "
            "it as answer-bearing EVEN WHEN it does not name the specific item: an inventory or scope of "
            "the relevant category that omits the item is exactly what establishes a negative answer. "
            "Individual example records, or documents that merely happen not to mention the item, are "
            "NOT the answer — score them LOW; prefer the defining/enumerating document.")
        ink("  polarity/existence: negative-answer question — steering to governing definition/inventory (%s)\n" % polarity_why)

    # SYNTHESIS DETECTION: is this an ENUMERATE-a-category question? (breadth over depth)
    synthesis, synth_why = _is_synthesis_question(nav_q)
    trace.append({"event":"mode","synthesis":synthesis,"why":synth_why})
    if synthesis:
        ink("  mode: SYNTHESIS (breadth over depth; up to %d files) -- %s\n" % (SYNTH_MAX_FILES, synth_why))

    # ENTITY-DEFINITION steering: on a "what is X" question the only branches that CAN score above the
    # floor are the ones whose names repeat X -- and those are the records that USE X, not the document
    # that DEFINES it. So name-mention must not be read as aboutness, and its ABSENCE must not be read
    # as irrelevance: a branch scoring 0.05 because "nothing here mentions X" is the ranker reporting
    # IGNORANCE (it only ever saw names), not a finding. The directive is pinned into working memory --
    # read by the ranker and every evidence judge, never by the answer prompt -- so it steers the ROOT
    # decision, where the region holding the definition would otherwise be floored and banished to
    # reserve, and keeps steering after a breadth escape lands the search somewhere new. Deferred to
    # the other two shapes, which carry their own (compatible but differently-aimed) steering.
    definitional, defn_why = _is_definitional_question(nav_q)
    definitional = definitional and not (synthesis or polarity)
    trace.append({"event":"mode","definitional":definitional,"why":defn_why})
    if definitional:
        # inserted directly rather than via _add_memory, whose 500-char clip would truncate a directive
        # of this length into a fragment; _add_memory's cap preserves _PIN entries, so it survives.
        memory.insert(0, _PIN + DEFN_DIRECTIVE)
        ink("  entity-definition: prefer the document that is ABOUT the entity over records that "
            "merely mention it (%s)\n" % defn_why)

    # RECENCY / LATEST-STATE steering: when the question asks for the CURRENT / LATEST / MOST RECENT
    # value of something the corpus tracks over time (a version, revision, edition, effective figure),
    # the answer is the MOST-RECENT entry of a SERIES -- a change log, version history or revision
    # table -- not the first version-bearing entry the search happens to read. Left alone the search
    # finds one plausible value, the gate accepts it, and it stops before the later entry that
    # supersedes it (which a whole-document baseline sees). Note the shape in working memory (ranker +
    # evidence judges read it, the answer prompt does not) so the series-recording document is
    # preferred and read through, and -- the load-bearing half -- pass the flag to the sufficiency gate
    # (via _gate) so a single mid-series value is not accepted until the full series is surveyed.
    recency, recency_why = _is_recency_question(nav_q)
    if recency:
        _add_memory(memory,
            "This question asks for the CURRENT / LATEST / MOST RECENT state of something the corpus "
            "tracks over time (a version, revision, edition, status or effective value). Such values "
            "live in a SERIES -- a change log, version history, revision table or amendment list -- "
            "where each entry was current only until a later one superseded it. Do NOT stop at the "
            "first version-bearing entry you find: the answer is the MOST RECENT / highest entry. Prefer "
            "the document that RECORDS THE SERIES, read ALL of its entries before concluding, and treat "
            "any single mid-series value as provisional until you have confirmed no later entry exists.")
        ink("  recency/latest: answer is the most-recent entry of a series -- surveying the full log (%s)\n" % recency_why)
    # Evidence budget. SYNTHESIS bounds on total pieces (its breadth-stop caps distinct files inside
    # the loop). DEFAULT bounds on DISTINCT FILES (MAX_FILES) with a piece backstop (MAX_EVIDENCE).
    # THE BUG THIS FIXES: the default bound was `len(evidence) < MAX_FILES`, counting intra-file
    # cluster PIECES against a FILE budget -- so ONE file's sweep (13 auto-read safety sections -> 16
    # pieces > 8) tripped the cap and killed cross-file search the instant it teleported, with the
    # gate still saying 'not enough, other areas uncovered'. Files and pieces are now separate.

    # start the descent at root
    cur = ROOT; visited.add(ROOT.node_id)
    ink("  path: root")

    # LEXICAL CONTENT SEED: give the frontier a content-grounded entry point for distinctive rare
    # terms that surface in no folder/file name or summary (the recall miss where the answer-bearing
    # file is otherwise never reached). Purely additive -- the LLM still confirms every seed by
    # reading it -- and generic (all-common-term) questions seed nothing, so default behaviour is
    # unchanged. When top-down routing flounders (all scores near the floor), the first teleport now
    # lands on the lexically-matched document instead of the next generic decoy.
    try:
        seeds = _lexical_seed_files(nav_q, ROOT)
        if seeds:
            _push_frontier(seeds, "lexical-seed")
            trace.append({"event":"lexical_seed",
                          "files":[{"node":f.name,"score":round(sc,3)} for f,sc,_ in seeds]})
            ink("\n  lexical seed: %s" % ", ".join("%s(%.2f)" % (f.name, sc) for f,sc,_ in seeds))
    except Exception:
        pass

    while (steps < MAX_STEPS and len(evidence) < MAX_EVIDENCE
           and (synthesis or _n_distinct_files(evidence) < MAX_FILES)):
        steps += 1
        entered_regions.add(_region_of(cur.node_id))   # this line's top-level region is now explored
        kids = [c for c in get_nav_children(cur, counter) if c.node_id not in visited]

        # ---- reached a FILE / leaf: read it and let the agent decide take vs reject ----
        # SYNTHESIS: at a FILE node, do NOT drill its sections -- take the whole document as one unit
        # and move on. Per-section descent into deep validation reports is the drilling that burned
        # ~800s and rejected non-answer paragraphs while the assay name sat in the Scope/Conclusion.
        # Folders still descend normally to reach the files inside.
        synth_file = synthesis and _is_file_node(cur)
        if cur.is_leaf() or not kids or synth_file:
            fj = _judge_file(nav_q, cur, memory, counter, full_content=synth_file)
            step_rec = {"step":steps,"at":cur.name,"node_type":cur.node_type,"event":"read_file",
                        "decision":fj["decision"],"reasoning":fj["reasoning"]}
            if fj["remember"]: _add_memory(memory, fj["remember"])
            if fj["decision"]=="take":
                # GRANULARITY: keep the passage itself, or climb to the surrounding section/file if
                # the answer depends on it (parent -> grandparent -> ... up to the file boundary).
                if cur.is_leaf():
                    unit, took_whole, escalations = _granularity_unit(nav_q, cur, memory, counter, ink)
                    # CLUSTER-PRESERVING GRANULARITY (fixes the distributed/enumerated-answer miss):
                    # climbing to a whole unit collapses the section into ONE monolithic blob and marks
                    # every descendant `seen`, which buries co-equal answer-bearing passages and silences
                    # the intra-file cluster sweep. When the answer is spread across several comparably
                    # high-scoring sibling passages (a cabinet in one, chair-backs in another), the answer
                    # step -- told to quote the MOST relevant passage and cite each doc once -- then emits
                    # only the top one and drops the rest, exactly where vector search (separate chunks)
                    # wins. So if the unit we climbed to holds a CLUSTER of other strong passages on the
                    # frontier, keep the taken passage PLUS each of those as DISTINCT evidence items rather
                    # than one blob, so every fact is independently visible and quotable. This triggers
                    # only on the distributed-answer signature (took_whole AND >=1 strong sibling passage);
                    # the ordinary single-fact-needs-context case still collapses to the whole unit.
                    cluster = ([f for f in frontier
                                if f["score"] >= CLUSTER_HIGH
                                and getattr(f["node"], "node_type", None) not in ("folder", "file")
                                and f["node"].node_id not in seen
                                and _node_under(f["node"].node_id, unit.node_id)]
                               if took_whole else [])
                    if cluster:
                        _collect_node(cur, whole=False)
                        kept = 0
                        for f in cluster:
                            if _collect_node(f["node"], whole=not f["node"].is_leaf()): kept += 1
                        frontier[:] = [f for f in frontier if f["node"].node_id not in seen]
                        ink("\n        granularity: distributed answer -> kept %d distinct passage(s) instead of one blob" % (kept + 1))
                        step_rec["distributed_passages"] = kept + 1
                    else:
                        _collect_node(unit, whole=took_whole)
                    last_take_file = _enclosing_file(cur)
                    if escalations:
                        step_rec["granularity"] = escalations
                        step_rec["kept_unit"] = unit.name
                        step_rec["kept_scope"] = ("section/file" if took_whole else "passage")
                else:
                    # synthesis takes the whole document (clipped) so 3-4 files stay a lean answer prompt
                    _collect_node(cur, whole=True, clip_chars=(SYNTH_FILE_CHARS if synthesis else None))
                    # was `= cur`, which confined the sweep to the taken SECTION and hid its siblings
                    last_take_file = _enclosing_file(cur)
                trail.append(_short(cur)+"*"); ink("\n    step %d read %s -> TAKE — %s" % (steps, cur.name, clip(fj["reasoning"],90)))
                # INTRA-FILE TRIAGE SWEEP: the current file is FINISHED before any teleport.
                # One triage pass shows the agent every remaining candidate (heading + descent
                # score + snippet); only its selections are read, each as a whole unit. Efficient
                # (a handful of calls even on a 100-chunk file) yet nothing is dismissed unseen.
                # never sweep a folder: that would pull in other documents, which is the teleport's
                # job. Only a real file (or a lone section with no file ancestor) is swept.
                if last_take_file is not None and last_take_file.node_type == "folder":
                    last_take_file = None
                if last_take_file is not None and len(evidence) < MAX_EVIDENCE:
                    ink("\n        finishing file: %s" % last_take_file.name)
                    # anchor: the score this line committed to, so the sweep's cluster band is measured
                    # against the evidence just taken rather than against the residue left behind.
                    sweep_tr = _intra_file_sweep(nav_q, last_take_file, visited, seen,
                                                 evidence, memory, counter, ink, _collect_node, frontier, reserve,
                                                 deferred=deferred, breadth=synthesis, anchor=descent_top)
                    if sweep_tr:
                        step_rec["intra_file_sweep"] = sweep_tr
                        step_rec["swept_file"] = last_take_file.name
                # a folder that just produced real evidence is NOT barren: clear the reject tally on its
                # ancestor folders so later rejects of its other sections can't demote a fruitful region.
                _ta = _PARENT.get(cur.node_id)
                while _ta:
                    reject_folders.pop(_ta, None)
                    _ta = _PARENT.get(_ta)
            else:
                trail.append(_short(cur)+"✗"); ink("\n    step %d read %s -> reject — %s" % (steps, cur.name, clip(fj["reasoning"],90)))
                # REACHED-BUT-MIS-DRILLED FILE: greedy descent read only ONE section of this file, so a
                # reject of that single section is weak evidence that the WHOLE file lacks the answer --
                # a confidently-reached file whose sibling sections are still unread deserves the same
                # intra-file triage sweep a take gets. Sweep first; only condemn the subtree if the file
                # was reached weakly (a real decoy) or has nothing left to examine.
                rej_file = _enclosing_file(cur)
                ev_before = len(evidence)
                swept = False
                # `_is_file_node`, not `== "file"`: corpora label the document level either way (see
                # _FILE_TYPES), and on a 'document'-typed tree this gate was NEVER true — so the
                # reject-side sweep silently no-opped corpus-wide and greedy descent kept condemning a
                # confidently-reached document on one unlucky section pick, exactly the asymmetry
                # REJECT_SWEEP_MIN exists to remove. The take-side sweep does not gate on type, so only
                # the reject path was dead, which is why exploration stayed asymmetric.
                if (rej_file is not None and _is_file_node(rej_file)
                        and descent_top >= REJECT_SWEEP_MIN and len(evidence) < MAX_EVIDENCE):
                    has_more = (any(c.node_id not in visited and c.node_id not in seen
                                    for c in rej_file.children)
                                or any(_node_under(f["node"].node_id, rej_file.node_id)
                                       for f in frontier + reserve))
                    if has_more:
                        ink("\n        reached file scored %.2f but its section didn't answer; sweeping its other sections" % descent_top)
                        sweep_tr = _intra_file_sweep(nav_q, rej_file, visited, seen,
                                                     evidence, memory, counter, ink, _collect_node,
                                                     frontier, reserve, deferred=deferred, breadth=synthesis,
                                                     anchor=descent_top)
                        if sweep_tr:
                            step_rec["reject_sweep"] = sweep_tr
                            step_rec["swept_file"] = rej_file.name
                        swept = True
                # BARREN REJECT: if nothing was salvaged (whether or not we swept), demote this region.
                # DECOY-TRAP FIX: first the rejected node's own enclosing-file subtree; then, the general
                # CLUSTER fix, bubble the reject up the ancestor FOLDERS and, once any folder has yielded
                # CLUSTER_REJECT_N barren files, demote that whole folder's remaining subtree — so a
                # homogeneous cluster of look-alike decoys is abandoned as a whole and the search breaks
                # out to structurally-distant, never-tried branches instead of grinding the cluster.
                if len(evidence) == ev_before:
                    _demoted = _penalize_subtree((rej_file or cur).node_id)
                    anc = _PARENT.get((rej_file or cur).node_id)
                    while anc:
                        anode = _NODES.get(anc)
                        if anode is None or anode.node_type == "root": break
                        if anode.node_type == "folder":
                            reject_folders[anc] += 1
                            if reject_folders[anc] == CLUSTER_REJECT_N:
                                _cl = _penalize_subtree(anc)
                                if _cl:
                                    _demoted += _cl
                                    ink("\n        (cluster-reject: '%s' yielded %d barren files; demoted %d frontier item(s) to reserve)"
                                        % (clip(anode.name, 40), CLUSTER_REJECT_N, _cl))
                        anc = _PARENT.get(anc)
                    if _demoted:
                        ink("\n        (reject-decay: demoted %d same-subtree frontier item(s) to reserve)" % _demoted)
            trace.append(step_rec)

            # Consider stopping only when we have evidence. DEFAULT path: a sufficiency gate
            # decides (would this evidence answer, or be DUNNO?). SYNTHESIS path: no gate -- each file
            # is already taken WHOLE, so just gather DISTINCT documents up to the breadth cap.
            if evidence:
                if synthesis:
                    # BREADTH, GATE-FREE. The sufficiency gate under-counts an enumeration (in the
                    # depth run it declared ENOUGH after 2 of 4 assays), so trusting it stops short of
                    # the full list. Gather distinct whole documents (or until the frontier is dry),
                    # then answer. Also removes one slow gate call per file.
                    n_files = _n_distinct_files(evidence)
                    # RELEVANCE-GATED BREADTH (fix): a fixed SYNTH_MAX_FILES cap answers with a PARTIAL
                    # list whenever the class is spread across more documents than the cap. So the base
                    # cap is only a FLOOR: past it, keep gathering while the best unexplored NEW-source
                    # frontier document scores within SYNTH_BREADTH_WINDOW of the WEAKEST document already
                    # accepted -- that unread document is as relevant as the ones kept, so for an
                    # enumeration it is very likely another DISTINCT item. When no such comparably-relevant
                    # document remains, the list is complete and we stop at the base cap exactly as before.
                    # Hard-bounded by MAX_FILES (breadth ceiling) and MAX_STEPS so it can never run away.
                    evid_srcs = {e.metadata.get("source_file") or e.path or e.name for e in evidence}
                    best_new = next((f for f in sorted(frontier, key=lambda f: f["score"], reverse=True)
                                     if _src_of(f["node"]) not in evid_srcs), None)
                    base = min(committed_scores.values()) if committed_scores else 0.0
                    strong_left = (best_new is not None
                                   and best_new["score"] >= base - SYNTH_BREADTH_WINDOW
                                   and best_new["score"] >= NOISE_FLOOR)
                    trace.append({"step":steps,"event":"synthesis_breadth","n_files":n_files,
                                  "n_evidence":len(evidence),"strong_left":bool(strong_left),
                                  "best_new":(round(best_new["score"],3) if best_new else None)})
                    if n_files >= MAX_FILES:
                        ink(" ⇒ synthesis: %d document(s) — breadth ceiling reached — answer" % n_files); break
                    if n_files >= SYNTH_MAX_FILES and not strong_left:
                        ink(" ⇒ synthesis: %d distinct document(s), no comparably-relevant document left — answer" % n_files); break
                    if not frontier and not reserve:
                        ink(" ⇒ synthesis: nothing left after %d document(s) — answer" % n_files); break
                    ink(" ⇒ synthesis: %d document(s) so far; teleporting to the next one" % n_files)
                    # fall through to teleport for the next distinct document
                else:
                    _unread = [f"{n.name}  (relevance {sc:.2f})" for n, sc in
                               sorted(deferred, key=lambda t: t[1], reverse=True)[:8] if sc >= 0.40]
                    suff, suff_reason = _gate(_unread)
                    # A CONTRAST EXCURSION ENDS at the first completed read after dispatch — this
                    # gate call always follows one. Note that it ended; whether it CONFIRMED the
                    # committed account is decided below by whether it changed the evidence.
                    was_contrast = contrast_active
                    contrast_active = False
                    trace.append({"step":steps,"event":"sufficiency_check","sufficient":suff,
                                  "reasoning":suff_reason,"n_evidence":len(evidence),
                                  "basis":GATE_LAST.get("basis"),
                                  "unread_same_file":len(_unread)})
                    ink("\n    sufficiency: %s — %s" % ("ENOUGH" if suff else "not enough", clip(suff_reason,90)))
                    # RESIDUAL STEERING: re-aim ranking/judging at the parts the gate says are still
                    # open, so the search stops chasing the parts it has already settled.
                    residual = [] if suff else (GATE_LAST.get("missing") or [])
                    _set_residual(memory, [] if suff else GATE_LAST.get("missing"))
                    # A contrast directive from an earlier check is stale once the gate reports the
                    # evidence INCOMPLETE: the job is back to filling the gap the residual names, not
                    # to seeking a rival account. Drop it so the two directives never steer at once.
                    if not suff: _set_contrast(memory, None)

                    # RECOVER DEFERRED SECTIONS BEFORE LEAVING THE DOCUMENT (default path only): read
                    # the best unread same-file sections before teleporting away.
                    _drained = 0
                    while (not suff) and deferred and _drained < DEFERRED_MAX_READS and len(evidence) < MAX_EVIDENCE:
                        deferred.sort(key=lambda t: t[1], reverse=True)
                        dn, dsc = deferred.pop(0)
                        _drained += 1
                        # fallback=True: this section was ranked below the bar and deferred; it is
                        # being read only because the gate called the kept evidence incomplete, so it
                        # must supply what is missing rather than merely co-mention the subject or
                        # contradict what is already held.
                        ddec, dadds, dwhy = _read_selection(nav_q, dn, evidence, memory, counter,
                                                            fallback=True)
                        if ddec == "take" and _collect_node(dn, whole=not dn.is_leaf()):
                            ink("\n    + recovered deferred (%.2f): %s — %s" % (dsc, dn.name, clip(dadds,70)))
                            trace.append({"step":steps,"event":"deferred_take","node":dn.name,
                                          "score":round(dsc,3),"adds":dadds})
                        else:
                            ink("\n    - deferred rejected (%.2f): %s — %s" % (dsc, dn.name, clip(dwhy,70)))
                            trace.append({"step":steps,"event":"deferred_skip","node":dn.name,
                                          "score":round(dsc,3),"reasoning":dwhy})
                            continue
                        _unread = [f"{n.name}  (relevance {sc:.2f})" for n, sc in
                                   sorted(deferred, key=lambda t: t[1], reverse=True)]
                        suff, suff_reason = _gate(_unread)
                        trace.append({"step":steps,"event":"sufficiency_check","sufficient":suff,
                                      "reasoning":suff_reason,"n_evidence":len(evidence),
                                      "basis":GATE_LAST.get("basis"),
                                      "after_deferred":True})
                        ink("\n    sufficiency: %s — %s" % ("ENOUGH" if suff else "not enough", clip(suff_reason,90)))
                        residual = [] if suff else (GATE_LAST.get("missing") or [])
                        _set_residual(memory, [] if suff else GATE_LAST.get("missing"))

                    # BREADTH-ESCAPE stall signal: an insufficiency verdict AFTER evidence already spans
                    # several distinct documents means this region is not yielding the answer; count it so
                    # a persistent stall can trigger a jump to an unexplored region at teleport time.
                    # THE RESET ON `suff` IS REMOVED, and that is the point. A sufficient verdict has
                    # exactly two outcomes: it ENDS the search (the contrast check finds nothing worth
                    # contrasting -> `stop` -> break, and the counter is moot), or the contrast check
                    # OVERRIDES it and the search carries on. Only the second case ever reached this
                    # reset, and there it is backwards — the search has not answered, is still inside
                    # the same region, and the evidence that it is stuck there is precisely what gets
                    # thrown away. So a gate that oscillates enough / not-enough — the signature of a
                    # region that keeps ALMOST answering, which is the same region monopoly the
                    # shortlist fix above addresses — could never accumulate STALL_TRIGGER, and the
                    # breadth escape was dead on exactly the runs that need it. `stall` is now cleared
                    # only where progress is real: when a breadth escape actually relocates the search.
                    if not suff and (_n_distinct_files(evidence) >= STALL_MIN_DOCS
                                     or (definitional and GATE_LAST.get("basis") == "incidental")):
                        # PROVENANCE STALL: "every description I hold is incidental" is a verdict about
                        # the KIND of document this region holds, not about how much of it has been
                        # read. Reading further along a trail of mentions cannot produce the document
                        # that defines the entity, because that document does not mention the entity by
                        # name -- that is exactly why the trail does not lead there. So it counts as a
                        # stall immediately, without waiting for STALL_MIN_DOCS distinct documents, and
                        # the breadth escape gets the search out to territory the mention-trail never
                        # reaches (it draws from reserve, where the floored region was banished).
                        stall += 1

                    # PRE-ANSWER CONTRAST CHECK (was: NEAR-TIE GUARD): even when the gate says ENOUGH,
                    # do not answer from the current document(s) while a comparably-ranked DIFFERENT
                    # document sits unread on the frontier -- the ranker could not tell the right file
                    # from a decoy, so check the alternative before committing. Eligibility is as
                    # before (relative to the score we committed to, bounded by TIE_MAX_EXPLORE, no
                    # keywords or file priorities); WHAT the check looks at is what changed -- see
                    # CONTRAST VERIFICATION above.
                    stop = False
                    if suff:
                        # A FAILED CONTRAST IS CONFIRMATION (stop logic). The excursion just ended
                        # was dispatched by the agent's OWN choice of the alternative most likely to
                        # CONTRADICT or SHARPEN the account in hand; it went there, read it, and it
                        # changed nothing. That is the strongest stop signal this loop ever gets —
                        # the committed account survived an adversarial check designed to break it.
                        # Re-running the check until TIE_MAX_EXPLORE is exhausted treats that
                        # confirmation as if it were doubt: each extra excursion burns several steps
                        # of the budget on progressively weaker candidates, and every extra
                        # below-bar read is a fresh chance for a neighbouring-subject passage to
                        # displace an already-complete answer. So a further alternative is
                        # considered ONLY while contrasts keep changing the evidence; a barren one
                        # ends the search. A wrong single-source account is unaffected: the rival
                        # that corrects it TAKES something, evidence changes, and the next check
                        # still fires.
                        confirmed = was_contrast and len(evidence) == contrast_ev_mark
                        near = None; alt_why = ""
                        if not confirmed and tie_explores < TIE_MAX_EXPLORE and frontier:
                            frontier.sort(key=lambda f: f["score"], reverse=True)
                            evid_srcs = {e.metadata.get("source_file") or e.path or e.name for e in evidence}
                            # ELIGIBILITY is unchanged: an unexplored candidate from a DIFFERENT source
                            # document than anything collected, within the band around the score we
                            # committed to. Near-ties always qualify; when the answer rests on ONE
                            # document, the wider band qualifies too (a lone strong match is where a
                            # decoy hides). Anchor on the WEAKEST document we actually accepted as
                            # evidence, on the frontier's own folder/file scale -- NOT the section-level
                            # descent_top, which runs higher and made a genuine peer sibling look too
                            # low, stopping the search while a distributed answer's other documents sat
                            # unread.
                            base = min(committed_scores.values()) if committed_scores else descent_top
                            # SINGLE ACCOUNT, not merely single source. Two documents that sit in the
                            # SAME top-level region and restate the same fact are ONE account reached
                            # by one route, not independent verification. The observed failure: the
                            # first contrast excursion took an AGREEING same-folder sibling of the
                            # committed document, which flipped the old single_source test off — the
                            # band narrowed and the banished-region admission below shut down — so two
                            # mutually-corroborating decoys confirmed each other while the never-
                            # entered region holding the right document was never offered again.
                            # Region identity is the tree's own structure; no content is matched.
                            single_account = len(evid_srcs) == 1 or len(evidence_regions) <= 1
                            window = SINGLE_SRC_WINDOW if single_account else TIE_WINDOW
                            # WHAT CHANGED: every eligible alternative is shortlisted, not just the
                            # top-scored one, and the AGENT picks which to check -- on the candidates'
                            # own words, asked for the one that would CONTRADICT or SHARPEN the account
                            # in hand. Score-priority alone spent this budget on siblings of the
                            # committed account (they score highest precisely because they restate its
                            # framing) while the document addressing the named subject sat mid-frontier,
                            # eligible but never looked at, because only ONE candidate was ever considered.
                            # ...and the shortlist is REGION-DIVERSE, which is what makes the widened
                            # band above mean anything. Eligibility is unchanged; the truncation is
                            # what changed. `[:ALT_SHORTLIST]` by score kept the top eight scorers,
                            # which are always eight descendants of the region already committed to —
                            # so the band opened deliberately wide to admit a lower-scored rival
                            # account had every one of those rivals clipped straight back out, and the
                            # check verified the committed region against more of itself.
                            eligible = [f for f in frontier
                                        if _src_of(f["node"]) not in evid_srcs
                                        and f["score"] >= base - window
                                        and f["score"] >= NOISE_FLOOR]
                            # BANISHED-REGION ADMISSION (the single-source stop). When the whole account
                            # rests on ONE document reached inside ONE confidently-picked region, the
                            # correct answer may live in a top-level region the ROOT ranker floored and
                            # never entered -- its low score is the ranker's IGNORANCE of body content
                            # (it saw only names/summaries/child-names), not a finding, the same
                            # rationale the breadth escape acts on. But the breadth escape only fires on
                            # a STALL of 'not enough' verdicts, and here the gate said ENOUGH on the
                            # first take, so it never ran; and the score-window eligibility above can
                            # never reach a floored region, so the contrast check verifies the committed
                            # region against more of ITSELF and confirms the wrong account. The fix: on
                            # a single-source stop, also offer the best-scoring frontier/reserve entry
                            # from each top-level region NOT yet entered, regardless of its score, so a
                            # banished-but-correct region gets ONE look before the agent commits. Still
                            # fully agentic -- the agent chooses among them (and may answer -1) and reads
                            # what it picks; recall-preserving; no index/similarity/keywords; and bounded
                            # by TIE_MAX_EXPLORE with a barren excursion ending the search, so a
                            # genuinely single-region question pays at most a couple of extra visits.
                            if single_account:
                                have_ids = {id(f) for f in eligible}
                                best_by_region = {}
                                for f in frontier + reserve:
                                    if _src_of(f["node"]) in evid_srcs: continue
                                    r = _region_of(f["node"].node_id)
                                    if r in entered_regions: continue
                                    cur_best = best_by_region.get(r)
                                    if cur_best is None or f["score"] > cur_best["score"]:
                                        best_by_region[r] = f
                                for f in best_by_region.values():
                                    if id(f) not in have_ids:
                                        eligible.append(f); have_ids.add(id(f))
                            elig = _diverse_shortlist(eligible, ALT_SHORTLIST)
                            if elig:
                                near, alt_why = _choose_alternative(nav_q, evidence, elig, memory, counter)
                        if near is None:
                            _set_contrast(memory, None)   # nothing left to contrast against
                            if confirmed:
                                ink(" ⇒ answer — contrast excursion changed nothing; committed account confirmed")
                                trace.append({"step":steps,"event":"contrast_confirmed",
                                              "n_evidence":len(evidence)})
                            else:
                                ink(" ⇒ answer%s" % (" — %s" % clip(alt_why, 80) if alt_why else ""))
                            stop = True
                        else:
                            tie_explores += 1
                            forced_next = near            # honour the choice: teleport THERE, not to
                                                          #   whatever score-priority prefers
                            contrast_active = True        # excursion in flight until its first read
                            contrast_ev_mark = len(evidence)
                            _set_contrast(memory, evidence)
                            ink("\n    contrast check: %s (%.2f) may give a different account than the "
                                "evidence held -- verifying before answering — %s"
                                % (near["node"].name, near["score"], clip(alt_why, 80)))
                            trace.append({"step":steps,"event":"contrast_explore","node":near["node"].name,
                                          "score":round(near["score"],3),"committed":round(descent_top,3),
                                          "n_eligible":len(elig),"reasoning":alt_why})
                    elif not frontier and not reserve:
                        ink(" ⇒ frontier and reserve exhausted; answering with what we have"); stop = True
                    if stop: break
                    # else: insufficient, or a near-tie to verify -> fall through and teleport onward

            # TELEPORT to the best node left anywhere on the global frontier. BREADTH ESCAPE: when the
            # sufficiency gate has repeatedly reported 'not enough' after evidence was gathered from
            # several distinct documents, the greedy line is stuck in a topically-adjacent DECOY region
            # (its high-scoring frontier is all same-region mentions) while the answer sits in a top-level
            # region the confident root decision scored below the noise floor and banished to reserve --
            # unreachable within the step budget because the frontier never empties. So on a stall, jump
            # to the best frontier/reserve entry in an UNEXPLORED top-level region, giving the low-scored-
            # but-correct branch the chance pure score-priority never would. General and recall-preserving:
            # no keywords/priorities, bounded by the finite number of regions, and it only fires after the
            # topical region has already been given a fair, multi-document look.
            nxt = None
            # CONTRAST VERIFICATION: the agent named the alternative worth checking; go there. Without
            # this the choice was advisory only — the pre-answer guard granted one more teleport and
            # score-priority then decided the destination, which is exactly the bias the check exists
            # to break. Falls through to the normal rules once the check has been dispatched.
            if forced_next is not None:
                for _i, _f in enumerate(frontier):
                    if _f is forced_next: del frontier[_i]; break
                nxt = forced_next; forced_next = None
            if nxt is None and (not synthesis) and stall >= STALL_TRIGGER:
                unentered = [(f, lst) for lst in (frontier, reserve) for f in lst
                             if _region_of(f["node"].node_id) not in entered_regions]
                if unentered:
                    esc, home = max(unentered, key=lambda t: t[0]["score"])
                    for i, f in enumerate(home):
                        if f is esc: del home[i]; break
                    stall = 0; nxt = esc
                    ink("\n    breadth escape: topical search stalled -> unexplored region %s (%.2f)"
                        % (esc["node"].name, esc["score"]))
                    trace.append({"step":steps,"event":"breadth_escape",
                                  "node":esc["node"].name,"score":round(esc["score"],3)})
            # RESIDUAL-AIMED TELEPORT: the gate just named what the evidence is still missing, so let
            # the agent choose the jump against THAT rather than against scores struck before any of it
            # was known (see the header above _choose_by_residual). Sits below the two structural
            # overrides — a contrast target the agent already named, and the breadth escape's jump to a
            # never-tried region — and above plain score-priority, which remains the fallback whenever
            # the gate names no gap, the frontier offers no real choice, or the model returns nothing
            # usable. Synthesis is excluded: it runs no gate, so it has no residual by construction.
            if nxt is None and residual and not synthesis:
                # REGION-DIVERSE POOL. The prompt tells the agent that a lower-scored candidate which
                # speaks to the gap beats a high-scored sibling of what it has already read — but a
                # plain top-RESID_SHORTLIST-by-score pool contains ONLY high-scored siblings of what it
                # has already read, so that instruction had nothing to act on and the choice was
                # between candidates the gate had just implied were all wrong in the same way.
                pool = _diverse_shortlist(frontier, RESID_SHORTLIST)
                if len(pool) >= 2:
                    pick, rwhy = _choose_by_residual(nav_q, residual, evidence, pool, memory, counter)
                    if pick is not None:
                        for _i, _f in enumerate(frontier):
                            if _f is pick: del frontier[_i]; break
                        nxt = pick
                        ink("\n    residual-aimed teleport -> %s (%.2f) — %s"
                            % (pick["node"].name, pick["score"], clip(rwhy, 80)))
                        trace.append({"step":steps,"event":"residual_teleport",
                                      "node":pick["node"].name,"score":round(pick["score"],3),
                                      "missing":residual[:PARTS_MAX],"n_shortlist":len(pool),
                                      "top_score":round(pool[0]["score"],3),"reasoning":rwhy})
            if nxt is None:
                nxt = _pop_frontier()
            snapshot = sorted(frontier + ([{ "node":nxt["node"],"score":nxt["score"],"reason":nxt["reason"],"from":nxt["from"]}] if nxt else []),
                              key=lambda f: f["score"], reverse=True)
            trace.append({"step":steps,"event":"teleport","from":cur.name,
                          "frontier":[{"node":f["node"].name,"score":round(f["score"],3),"from":f["from"]} for f in snapshot[:12]],
                          "target":(nxt["node"].name if nxt else None),
                          "target_score":(round(nxt["score"],3) if nxt else None)})
            if nxt is None:
                ink("\n    frontier empty; stopping"); break
            teleports += 1
            ink("\n    TELEPORT -> %s (score %.2f). frontier:" % (nxt["node"].name, nxt["score"]))
            for f in snapshot[:8]:
                ink("\n        %.2f  %s" % (f["score"], f["node"].name))
            cur = nxt["node"]; visited.add(cur.node_id); trail.append("⇥"+_short(cur))
            descent_top = nxt["score"]   # new line: anchor near-tie comparisons on the target's own score
            line_entry_score = nxt["score"]   # doc-commitment score for the teleported-to line
            deferred.clear()   # deferred candidates belong to the file we just left; don't carry them over
            nosignal_run = 0   # new line: the no-signal run of the abandoned line does not carry over
            continue

        # ---- interior node: rank children, descend the best, push the rest to the frontier ----
        ranked = rank_children(nav_q, cur, kids, memory, counter)   # [(child, score, reason), ...] hi->lo
        best, best_sc, best_rs = ranked[0]
        # CONTRAST COLLAPSE ABORT: mid-excursion, the ranker has just judged EVERY child of the
        # contrast target irrelevant to the question (best below the noise floor). The excursion
        # exists to check a rival ACCOUNT; a subtree whose own contents all rank as noise holds
        # none, so grinding further steps down to a leaf read only re-derives this verdict at
        # higher cost. The evidence in hand was gate-certified sufficient when the excursion was
        # dispatched and has not changed since (an excursion ends at its first completed read), so
        # answer from it now. Scoped strictly to in-flight contrast excursions — ordinary descent
        # still follows low scores, which elsewhere can be ranker ignorance rather than
        # irrelevance (see the entity-definition steering above).
        if contrast_active and evidence and best_sc < NOISE_FLOOR:
            _penalize_subtree(cur.node_id)
            _set_contrast(memory, None)
            contrast_active = False
            ink("\n    contrast target collapsed on inspection (best child %.2f) — committed "
                "account confirmed ⇒ answer" % best_sc)
            trace.append({"step":steps,"at":cur.name,"event":"contrast_abort",
                          "best_score":round(best_sc,3),"n_evidence":len(evidence)})
            break
        # NO-SIGNAL DESCENT ABORT (navigation): at FOLDER routing levels, a best-child score below
        # the noise floor is the ranker declaring IGNORANCE — "nothing listed here relates" — so a
        # descent taken on it is a guess, not a decision. One guess is allowed (a level deeper the
        # ranker sees richer names/text and may recover a real signal), but when NOSIGNAL_ABORT
        # consecutive folder rankings on the same line all come back sub-floor, the guess is
        # falsified: drilling further only re-derives "nothing here" at higher cost (the observed
        # failure spent 6 of 18 steps this way while above-floor candidates sat on the frontier).
        # Abandon the line: keep every child reachable (frontier/reserve) and teleport to the best
        # candidate elsewhere. Gated on the frontier actually holding an above-floor alternative —
        # with nowhere better to go, greedy descent proceeds unchanged — and scoped to folders, so
        # section-level lowness inside a document (often summary-lossiness) never triggers it.
        # No keywords, no index: the trigger is the ranker's own scores.
        if cur.node_type in ("root", "folder"):
            nosignal_run = nosignal_run + 1 if best_sc < NOISE_FLOOR else 0
            if nosignal_run >= NOSIGNAL_ABORT and frontier:
                _push_frontier(ranked, cur.name)                    # children stay reachable
                nxt = _pop_frontier()
                nosignal_run = 0; teleports += 1
                trace.append({"step":steps,"at":cur.name,"event":"nosignal_abort",
                              "best_score":round(best_sc,3),"target":nxt["node"].name,
                              "target_score":round(nxt["score"],3)})
                ink("\n    step %d @ %s: no-signal descent (best child %.2f) — abandoning line; TELEPORT -> %s (%.2f)"
                    % (steps, cur.name, best_sc, nxt["node"].name, nxt["score"]))
                cur = nxt["node"]; visited.add(cur.node_id); trail.append("⇥"+_short(cur))
                descent_top = nxt["score"]; line_entry_score = nxt["score"]
                deferred.clear()
                continue
        descent_top = max(descent_top, best_sc)                     # track the committed line's best score
        if best.node_type in ("folder", "file"):
            line_entry_score = best_sc                              # doc-commitment score, on the SAME
                                                                    #   scale as frontier entries (section
                                                                    #   scores are excluded on purpose)
        _push_frontier(ranked[1:], cur.name)                        # runner-ups remembered for teleport
        trace.append({"step":steps,"at":cur.name,"node_type":cur.node_type,"event":"descend",
                      "chose":best.name,"chose_score":round(best_sc,3),"reasoning":best_rs,
                      "ranked":[{"node":c.name,"score":round(sc,3)} for c,sc,_ in ranked[:10]]})
        ink("\n    step %d @ %s -> descend %s (score %.2f)" % (steps, cur.name, best.name, best_sc))
        if live and len(ranked) > 1:
            for c,sc,_ in ranked[:5]:
                ink("\n        %.2f  %s" % (sc, c.name))
        visited.add(best.node_id); trail.append(_short(best)); cur = best

    # last resort: if we somehow gathered nothing, keep wherever we ended
    if not evidence and cur.node_id not in seen:
        _collect_node(cur, whole=not cur.is_leaf())

    # RETRIEVAL HAND-OFF (multi-part questions only): settle which collected piece carries which part
    # and rebuild the bundle part by part, so a part that was genuinely retrieved is not buried behind
    # the bulk gathered for the others. Synthesis is excluded: its bundle is deliberately whole
    # documents for an enumeration, which has one part by construction.
    final_evidence = evidence
    if (not synthesis) and len(parts) >= 2 and len(evidence) >= PART_SELECT_MIN:
        try:
            final_evidence, per_part = _select_by_part_coverage(nav_q, parts, evidence, counter)
            if per_part is not None:
                trace.append({"event":"part_coverage","parts":parts,
                              "kept":len(final_evidence),"of":len(evidence),
                              "per_part":{str(k): v for k, v in per_part.items()}})
                ink("\n    part coverage: kept %d of %d piece(s) — %s"
                    % (len(final_evidence), len(evidence),
                       ", ".join("part %d: %d" % (k, len(v)) for k, v in per_part.items())))
        except Exception:
            final_evidence = evidence
    if live: print()
    return {"evidence":final_evidence[:MAX_EVIDENCE],"memory":memory,"steps":steps,
            "teleports":teleports,"path":" › ".join(trail),"trace":trace,"synthesis":synthesis}

print("agent ready (greedy descent + teleport frontier; parent-take granularity, intra-file first)")


# In[7]:


# ---- final answer from whatever the agent collected, in the OICR house format ----

# Citation rule (requested): cite each document only ONCE. If several passages come from the same
# document, the model is told to group them and place a single [doc_id] at the end of that block.
# As a safety net we also collapse duplicates: for any [doc_id] that appears more than once we keep
# only its LAST occurrence, so the citation ends up at the end of that document's information.
ANSWER_DEBUG = {}   # per-question answer-assembly diagnostics, written into the report

def _dedupe_citations(text):
    if not text: return text
    toks = list(re.finditer(r"\[[^\]\[]+\]", text))
    if not toks: return text
    from collections import Counter as _C
    counts = _C(t.group(0) for t in toks)
    # indices (in char space) to drop: every occurrence of a repeated token except its last
    last_pos = {}
    for t in toks:
        last_pos[t.group(0)] = t.start()
    drop_spans = []
    for t in toks:
        tok = t.group(0)
        if counts[tok] > 1 and t.start() != last_pos[tok]:
            drop_spans.append((t.start(), t.end()))
    if not drop_spans: return text
    out = []
    prev = 0
    for a, b in sorted(drop_spans):
        out.append(text[prev:a]); prev = b
    out.append(text[prev:])
    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)     # collapse gaps left by removed tokens
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)  # tidy space-before-punctuation
    return cleaned.strip()

def answer_oicr(q, evidence, counter):
    # The prompt is deliberately minimal: SYSTEM_PROMPT + INSTRUCTIONS + documents + question,
    # nothing else — the OICR instructions already say everything about format, quoting, citations,
    # and the DUNNO fallback, so no extra scaffolding (headings, quote sections) is layered on top.
    groups={}; order=[]
    for e in evidence[:MAX_EVIDENCE]:
        k=e.metadata.get("source_file") or e.path or e.name
        if k not in groups: groups[k]=[]; order.append(k)
        groups[k].append(full(e.content or e.summary))
    docs="\n\n".join(f"--- START DOCUMENT [doc_id: {k}] ---\n"+"\n".join(groups[k])+"\n--- END DOCUMENT ---"
                       for k in order) or "(no documents were retrieved)"
    prompt=(
        f"{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}\n\n"
        f"DOCUMENTS:\n{docs}\n\n"
        f"QUESTION: {q.stem}\n\nANSWER:\n")
    # Extract the answer robustly. gpt-oss thinking can leave message.content empty (the llm()
    # wrapper then returns the reasoning text) or emit stray/unterminated <think> tags. This strips
    # reasoning in all its forms, falls back to the last substantive paragraph, and finally RETRIES
    # the call before ever returning empty — so a good retrieval always produces a non-empty answer.
    def _strip_reasoning(t):
        t = t or ""
        # gpt-oss harmony channels: the OLD bug was stripping the channel MARKERS but keeping all
        # the text between them — which concatenated analysis-channel deliberation into the answer.
        # Correct handling: if a 'final' channel exists, the answer is ONLY what's inside it; every
        # other channel (analysis etc.) is reasoning and is dropped entirely.
        parts = re.split(r"<\|channel\|>\s*final\s*<\|message\|>", t, flags=re.I)
        if len(parts) > 1:
            t = re.split(r"<\|", parts[-1])[0]
        else:
            glued = re.split(r"\bassistant\s*final\b\s*[:\-]?", t, flags=re.I)   # glued marker variant
            if len(glued) > 1: t = glued[-1]
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S|re.I)      # closed think blocks
        t = re.sub(r"<think>.*$", "", t, flags=re.S|re.I)             # unterminated <think> ... (no close)
        t = re.sub(r"<\|?(?:channel|start|end|message|assistant|analysis|final)\|?>", "", t, flags=re.I)  # stray tags
        return t.strip()

    def _bad_answer(t):
        """True when the text is not a usable answer: empty, an attempted tool call, or visible
        deliberation about using tools. These can never be shown as the answer; they route to the
        retry, and ultimately to DUNNO — reasoning is NEVER returned as an answer."""
        if not t: return True
        if re.search(r'\{\s*"(action|tool|tool_call|function|name)"\s*:', t): return True      # tool-call JSON
        if re.search(r'(?i)\b(use|call|invoke|run|issue)\b[^.\n]{0,40}\b(search|retrieval)\b[^.\n]{0,20}\b(tool|function)s?\b', t): return True
        if re.search(r'(?i)^\s*search(ing)?\s+for\b', t): return True                            # "Search for \"X\""
        if re.search(r'(?i)\bas\s+chatgpt\b|\bwe\s+cannot\s+actually\s+run\b|\bsimulate\b.{0,30}\btool', t): return True
        return False

    def _extract(raw):
        clean = _strip_reasoning(raw)
        m = re.search(r"ANSWER\s*:\s*(.*)", clean, flags=re.I|re.S)   # tolerate an echoed heading
        cand = (m.group(1) if m else clean).strip(' "\'\n`-:•–*#')
        if not cand:
            paras = [p.strip(' "\'\n`-:•–*#') for p in re.split(r"\n\s*\n", clean) if p.strip()]
            paras = [p for p in paras
                     if not re.match(r"(?i)^(question|documents?|rules?|--- )", p)]
            cand = paras[-1] if paras else clean.strip(' "\'\n`-:•–*#')
        return cand

    retry_prompt = (
        f"{SYSTEM_PROMPT}\n\n{INSTRUCTIONS}\n\n"
        "NOTE: the search and retrieval steps have ALREADY been completed — the resulting "
        "documents are provided in full below. Do NOT attempt to call, simulate, or describe any "
        "tool. Output ONLY the answer text, with no headings, no preamble, and no reasoning.\n\n"
        f"DOCUMENTS:\n{docs}\n\nQUESTION: {q.stem}\n\nANSWER:")

    # gpt-oss always reasons (ollama's think=False does not disable the harmony analysis channel)
    # and that reasoning is paid for out of num_predict. THE FAILURE THIS GUARDS: with sizeable
    # evidence the model can exhaust its whole budget in the analysis channel, so content comes
    # back EMPTY — which the bad-answer path then collapses into DUNNO even though the retrieval
    # was perfect and the sufficiency gate had just certified it. So: generous budgets, escalation
    # on detected exhaustion (done_reason == "length" with all tokens in thinking), and — because a
    # DUNNO that contradicts the gate's own sufficiency verdict is far more likely mechanical than
    # real — one re-ask before a DUNNO-with-evidence is accepted as genuine.
    ANSWER_DEBUG.clear()
    attempts = [(prompt, 4096), (retry_prompt, 6144)]
    ans = ""
    for n_att, (p, budget) in enumerate(attempts, 1):
        raw = llm(p, counter, num_predict=budget, temperature=0, think=False, thinking_fallback=False)
        meta = dict(LLM_LAST)
        exhausted = (not raw) and (meta.get("thinking_len", 0) > 0 or meta.get("done_reason") == "length")
        ANSWER_DEBUG.setdefault("attempts", []).append({
            "n": n_att, "budget": budget, "content_len": meta.get("content_len"),
            "thinking_len": meta.get("thinking_len"), "done_reason": meta.get("done_reason"),
            "reasoning_exhausted": exhausted})
        ans = _extract(raw)
        good = (not _bad_answer(ans)) and not (evidence and ans.strip() == DUNNO)
        if good:
            break
        if evidence and ans.strip() == DUNNO:
            ANSWER_DEBUG["dunno_with_evidence"] = True
            print("      [answer] WARNING: model said DUNNO despite gate-approved evidence"
                  + (" (reasoning exhausted the token budget)" if exhausted else "")
                  + ("; retrying once" if n_att < len(attempts) else "; accepting after retry"))
    if _bad_answer(ans):
        ans = DUNNO

    # ---- COVERAGE GATE: REMOVED --------------------------------------------------------------
    # A gate used to sit here. It re-read the evidence, asked the model to enumerate the elements
    # the answer ought to contain, and -- if any were absent -- made a repair call that rewrote the
    # answer with them added.
    #
    # Both are gone, and the reason is the benchmark's validity rather than answer length. What
    # this notebook is meant to measure is the quality of the EVIDENCE that agentic tree traversal
    # brings to a question, against the evidence that vector search brings. The answer prompt is a
    # controlled variable. qms_search drafts its answer in exactly one LLM call from its retrieved
    # text; every extra stage on TreeRAG's side -- a gate, a repair, a compression, a second look at
    # the documents -- is an intervention on the ANSWER that qms does not get, and it moves the
    # thing being compared away from retrieval and toward post-processing. A repaired TreeRAG answer
    # that beats qms tells you the repair loop works, not that the tree found better evidence.
    #
    # The gate was also strictly one-directional: it only ever ADDED elements, never removed
    # out-of-scope ones, so it ratcheted answers past 200 words while qms answered the same
    # questions in ~120. Its enumeration step scored elements against the documents rather than the
    # question, so a correctly-scoped draft read as incomplete and got a document survey grafted on.
    #
    # The two mechanical safeguards that remain are NOT part of this and are deliberately kept:
    # the DUNNO retry above and the citation repair below correct malformed OUTPUT (an empty
    # generation, a dropped [doc_id]) rather than the answer's content, and qms's own harness has
    # equivalents. They stay.

    # ---- word count (for the report) ---------------------------------------------------------
    # The LENGTH COMPRESSION pass that used to live here has been REMOVED, not tuned. It re-ran
    # _coverage_check on the compressed text and accepted the rewrite only if ZERO elements were
    # missing -- but elements were scoped to the DOCUMENTS, so the compressed answer was re-graded
    # against the very inventory it was trying to escape and any real compression "dropped an
    # element" and was rejected. It burned two LLM calls per long question and reverted. With
    # elements now scoped to the QUESTION the draft lands near target on its own, so the
    # compensating mechanism is deleted rather than repaired. Length is a soft target; the coverage
    # gate above remains the only length-affecting machinery and it is one-directional (adds only).
    def _wc(t): return len((t or "").split())

    if evidence and ans and ans.strip() != DUNNO:
        ANSWER_DEBUG["words"] = _wc(ans)

    # citation repair: INSTRUCTIONS require a [doc_id], but the model sometimes drops them entirely.
    # If we produced a real (non-DUNNO) answer from actual evidence yet it carries NO [..] citation,
    # re-emit the SAME answer with citations added — one per document, at the end of that document's
    # information (never repeated per sentence). doc_ids are the source_file keys we grouped by.
    doc_ids = list(order)
    has_cite = bool(re.search(r"\[[^\]]+\]", ans))
    if evidence and ans and ans.strip() != DUNNO and not has_cite:
        cite_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Rewrite the answer below so that the information from each document ends with a single "
            "citation to that document, formatted exactly as [doc_id], using ONLY these document ids: "
            f"{', '.join('[' + d + ']' for d in doc_ids)}. Cite each document at most ONCE — group the "
            "information from the same document together and place its single [doc_id] at the end of "
            "that group. Do NOT repeat a [doc_id] after every sentence. Keep the wording and facts the "
            "same; only add the [doc_id] citations. Output ONLY the rewritten answer, no headings.\n\n"
            f"DOCUMENTS:\n{docs}\n\n"
            f"QUESTION: {q.stem}\n\nANSWER TO CITE:\n{ans}\n\nCITED ANSWER:")
        raw3 = llm(cite_prompt, counter, num_predict=1024, temperature=0, think=False, thinking_fallback=False)
        cited = _strip_reasoning(raw3).strip(' "\'\n`-:•–*#')
        # accept the rewrite only if it actually added a citation, didn't collapse to nothing/DUNNO,
        # and isn't leaked deliberation or a tool-call attempt
        if cited and cited.strip() != DUNNO and re.search(r"\[[^\]]+\]", cited) and not _bad_answer(cited):
            ans = cited

    # final safety net: collapse any repeated [doc_id] so each document is cited only once (last pos)
    ans = _dedupe_citations(ans)
    return ans

# ---- judge: grade a free response against the correct option(s) by meaning ----
def _parse_judge(raw):
    clean=re.sub(r"<think>.*?</think>","",raw or "",flags=re.S).strip()
    clean=re.sub(r"^```(?:json)?|```$","",clean,flags=re.M).strip()
    score=0.0; reason=""
    m=re.search(r"\{.*\}",clean,flags=re.S)
    if m:
        try:
            d=json.loads(m.group(0)); score=float(d.get("score",0)); reason=str(d.get("reason","")).strip()
        except Exception:
            try:
                import ast; d=ast.literal_eval(m.group(0))
                score=float(d.get("score",0)); reason=str(d.get("reason","")).strip()
            except Exception: pass
    if not reason:
        sm=re.search(r"(?:score)[\s\"'=:]+([01](?:\.\d+)?|0?\.\d+)",clean,re.I)
        if sm: score=float(sm.group(1))
        rm=re.search(r"(?:reason)[\s\"'=:]+(.*)",clean,re.I)
        reason=(rm.group(1).strip(' "\'}') if rm else clean[:150].replace("\n"," ").strip())
    return max(0.0,min(1.0,score)), reason

def judge_score(q,response,counter):
    al=_ans_list(q); multi=_is_multi(q)   # 'All of the above' counts as MULTI: all parts required
    prompt=("you are a fair grader. a student answered an open question in their own words and could not see the "
            "choices. the multiple choice version below has the correct option(s) marked; those are ground truth.\n\n"
            f"QUESTION: {q.stem}\nOPTIONS:\n{_opts_text(q)}\nCORRECT OPTION(S): {_gold_letters_str(q)}\n{_gold_text(q)}\n\n"
            f"STUDENT RESPONSE:\n{response or '(empty)'}\n\n"
            "grade 0.0 to 1.0 how well the response matches the MEANING of the correct option(s). judge by meaning "
            "not wording. "
            +("when several options are correct, full credit only if the response conveys ALL of them, proportional "
              "partial credit for some. " if multi else
              "full or near-full credit when it conveys the correct idea in different words, partial when incomplete. ")
            +"low credit when it matches a wrong option or is irrelevant. reply ONLY json:\n"
            '{"score": <0.0-1.0>, "reason": "<one concise sentence>"}')
    return _parse_judge(llm(prompt,counter,num_predict=1024,temperature=0,think=False,model=JUDGE_MODEL))

# ---- joint judge: grade BOTH responses in ONE llm call, against the same ground truth ----
# Requested for consistency: when TreeRAG and qms convey the same idea they should not drift 0.4
# apart just because they were graded in two separate calls. Both are scored in a single call, on
# the same rubric, but each strictly on its own merit (independent scores, no forced ranking). To
# avoid any system bias the two are shown as neutral "RESPONSE A / RESPONSE B" and, per question,
# which system is A vs B is decided by a stable hash of the qid (so it varies but is reproducible).
def _parse_two(raw):
    clean=re.sub(r"<think>.*?</think>","",raw or "",flags=re.S).strip()
    clean=re.sub(r"^```(?:json)?|```$","",clean,flags=re.M).strip()
    a_s=b_s=0.0; a_r=b_r=""
    m=re.search(r"\{.*\}",clean,flags=re.S)
    if m:
        try:
            d=json.loads(m.group(0))
            a_s=float(d.get("a_score",0)); b_s=float(d.get("b_score",0))
            a_r=str(d.get("a_reason","")).strip(); b_r=str(d.get("b_reason","")).strip()
            return (max(0.0,min(1.0,a_s)),a_r,max(0.0,min(1.0,b_s)),b_r)
        except Exception: pass
    # loose fallback: pull the two scores positionally
    nums=re.findall(r'"?[ab]_score"?\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)',clean,re.I)
    if len(nums)>=2: a_s,b_s=float(nums[0]),float(nums[1])
    return (max(0.0,min(1.0,a_s)),a_r,max(0.0,min(1.0,b_s)),b_r)

def judge_both(q, tree_response, qms_response, counter):
    """Grade tree_response and qms_response together in one call. Returns
    (tree_score, tree_reason, qms_score, qms_reason). If qms_response is None, grades tree alone."""
    if qms_response is None:
        ts,tr=judge_score(q,tree_response,counter)
        return ts,tr,None,"no qms answer for this qid"
    al=_ans_list(q); multi=_is_multi(q)   # 'All of the above' counts as MULTI: all parts required
    # stable per-question A/B assignment so neither system is always 'A'
    tree_is_a = (int(hashlib.md5(str(q.qid).encode()).hexdigest(),16) % 2 == 0)
    a_resp, b_resp = (tree_response, qms_response) if tree_is_a else (qms_response, tree_response)
    rubric=("when several options are correct, full credit only if a response conveys ALL of them, proportional "
            "partial credit for some. " if multi else
            "full or near-full credit when a response conveys the correct idea in different words, partial when incomplete. ")
    prompt=("you are a fair grader. a student answered an open question in their own words and could not see the "
            "choices. the multiple choice version below has the correct option(s) marked; those are ground truth. "
            "TWO independent responses (A and B) are provided. Grade EACH ONE strictly on its OWN merit against the "
            "ground truth — do NOT compare them to each other, do NOT force them apart or together, do NOT rank. "
            "Two responses that convey the same correct idea MUST receive the same score.\n\n"
            f"QUESTION: {q.stem}\nOPTIONS:\n{_opts_text(q)}\nCORRECT OPTION(S): {_gold_letters_str(q)}\n{_gold_text(q)}\n\n"
            f"RESPONSE A:\n{a_resp or '(empty)'}\n\n"
            f"RESPONSE B:\n{b_resp or '(empty)'}\n\n"
            "grade each 0.0 to 1.0 by how well it matches the MEANING of the correct option(s). judge by meaning "
            "not wording. "+rubric+
            "low credit when a response matches a wrong option or is irrelevant. reply ONLY json:\n"
            '{"a_score": <0.0-1.0>, "a_reason": "<one concise sentence>", '
            '"b_score": <0.0-1.0>, "b_reason": "<one concise sentence>"}')
    a_s,a_r,b_s,b_r=_parse_two(llm(prompt,counter,num_predict=1024,temperature=0,think=False,model=JUDGE_MODEL))
    if tree_is_a:
        return a_s,a_r,b_s,b_r
    return b_s,b_r,a_s,a_r

# grade whether the RETRIEVED text contains the facts needed, isolating navigation from answering
def judge_evidence(q,evidence,counter):
    ev="\n\n".join(f"[{e.metadata.get('source_file') or e.path or e.name}] {clip(e.content or e.summary,1500)}"
                    for e in evidence[:MAX_EVIDENCE]) or "(nothing was retrieved)"
    prompt=("you are checking whether a retrieval system fetched the right information, not whether anyone "
            "answered. below is the correct answer(s) and the retrieved text.\n\n"
            f"QUESTION: {q.stem}\nCORRECT ANSWER(S): {_gold_letters_str(q)}\n{_gold_text(q)}\n\n"
            f"RETRIEVED TEXT:\n{ev}\n\n"
            "rate 0.0 to 1.0 how well the retrieved text CONTAINS the information needed to reach the correct "
            "answer(s), whether or not phrased as the answer; weight by how many answers are supported. reply ONLY "
            'json {"score": <0.0-1.0>, "reason": "<one concise sentence>"}')
    return _parse_judge(llm(prompt,counter,num_predict=1024,temperature=0,think=False,model=JUDGE_MODEL))

print("answer + judge ready (one-cite-per-doc; joint judge for tree+qms)")


# In[8]:


# ---- load the tree ----
if not TREE_FILE.exists():
    raise SystemExit(f"tree not found at {TREE_FILE.resolve()}; build it first with the indexer")
ROOT = TreeNode.from_dict(json.loads(TREE_FILE.read_text(encoding="utf-8")))
index_tree(ROOT)
print(f"loaded tree {ROOT.name}; {ROOT.count_leaves()} leaves and {len(ROOT.children)} top level children")


# In[9]:


# ---- load ALL questions and the pre-computed qms_search answers (+ files_read + searches) ----
import ast

def _load_json_or_py(path):
    raw = Path(path).read_text(encoding="utf-8")
    try: return json.loads(raw)
    except Exception: return ast.literal_eval(raw)

def _to_question(rec):
    qid = str(rec.get("qid") or rec.get("id") or "")
    opts = {str(k).upper(): str(v).strip() for k, v in (rec.get("options") or {}).items()}
    ansv = rec.get("answers") or ([rec.get("answer")] if rec.get("answer") else [])
    letters = {m.upper() for m in re.findall(r"(?<![A-Za-z])([A-Fa-f])(?![A-Za-z])", " ".join(map(str, ansv)))}
    ans = [L for L in "ABCDEF" if L in letters]
    stem = str(rec.get("stem") or rec.get("question") or "").strip()
    return Question(qid, stem, opts, ans[0] if ans else "", "", ans, bool(rec.get("modified")))

# the TEXT of the correct option(s), never the bare letter. Referential options are RESOLVED: for
# "E. All of the above" this returns the literal label AND the texts of A, B, C, since that content
# is what a prose answer actually has to convey (and what the judge grades against).
def gold_texts(q):
    al, eff = _ans_list(q), _effective_letters(q)
    out = [q.options.get(L, "") for L in al if q.options.get(L)]
    if eff != al:
        out += [q.options.get(X, "") for X in eff if q.options.get(X) and q.options.get(X) not in out]
    return out

# resolve a short files_read entry to its FULL path by scanning every array in `searches`
def resolve_full_paths(files_read, searches):
    flat = []
    for arr in (searches or []):
        if isinstance(arr, list): flat.extend(str(x) for x in arr)
        elif isinstance(arr, str): flat.append(arr)
    out = []
    for f in (files_read or []):
        f = str(f)
        full_hit = next((p for p in flat if p == f), None) \
                or next((p for p in flat if p.endswith("/"+f) or p.endswith("\\"+f)), None) \
                or next((p for p in flat if f and f in p), None)
        out.append(full_hit or f)   # fall back to the raw name if no path found
    return out

_qdata = _load_json_or_py(QUESTIONS_FILE)
if isinstance(_qdata, dict): _qdata = _qdata.get("questions", list(_qdata.values()))
QUESTIONS = [q for q in (_to_question(r) for r in _qdata) if q.qid]
print(f"loaded {len(QUESTIONS)} questions from {QUESTIONS_FILE}")

_qms_raw = _load_json_or_py(QMS_ANSWERS_FILE)
if isinstance(_qms_raw, dict):
    _qms_raw = _qms_raw.get("answers", list(_qms_raw.values())) if not any(
        k in _qms_raw for k in ("qid","id")) else [_qms_raw]
QMS = {}
for r in (_qms_raw if isinstance(_qms_raw, list) else []):
    if not isinstance(r, dict): continue
    qid = str(r.get("qid") or r.get("id") or "")
    if not qid: continue
    files_read = r.get("files_read") or []
    searches   = r.get("searches") or []
    QMS[qid] = {
        "answer": str(r.get("answer") or r.get("response") or r.get("final_answer") or r.get("qms_answer") or ""),
        "files_read": files_read,
        "files_read_full": resolve_full_paths(files_read, searches),   # <-- full paths via `searches`
        "searches": searches,
    }
print(f"loaded {len(QMS)} qms_search answers from {QMS_ANSWERS_FILE}")
_missing = [q.qid for q in QUESTIONS if q.qid not in QMS]
if _missing: print(f"note: {len(_missing)} question(s) have no qms entry, skipped for qms scoring: {_missing[:10]}")


# In[10]:


# ---- sequential benchmark. Per question it prints: the correct answer TEXT (not letters),
#      the qms_search files it read (FULL paths resolved via `searches`), the full TreeRAG
#      traversal (descends, teleports, relevance scores), then BOTH full answers. It judges
#      both with the same judge, and whenever qms beats treerag it runs a diagnosis that looks
#      at where qms's file ranked in the traversal and suggests a concrete design fix. Saves
#      after EVERY question (atomic replace) so an early stop / crash keeps all progress. ----

def _cite_paths(answer, names):
    _pat = re.compile(r"[\[\u3010]([0-9,\u2020L\-]+)[\]\u3011]")
    if not answer: return answer
    cited = []
    for m in _pat.finditer(answer):
        for g in m.group(1).split(","):
            cid = g.split("\u2020")[0]
            if cid not in cited: cited.append(cid)
    if not cited: return answer
    alias = {}
    for n, old in enumerate(cited):
        alias[old] = names[n] if n < len(names) else f"Unknown Source {old}"
    def _repl(m):
        out = []
        for g in m.group(1).split(","):
            p = alias[g.split("\u2020")[0]]
            if p not in out: out.append(p)
        return "[" + ", ".join(out) + "]"
    return _pat.sub(_repl, answer)

def _load_report():
    if Path(REPORT_FILE).exists():
        try: return json.loads(Path(REPORT_FILE).read_text(encoding="utf-8"))
        except Exception: pass
    return {"model": AGENT_MODEL, "questions_file": QUESTIONS_FILE, "results": []}

def _save_report(report):
    tmp = Path(REPORT_FILE + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REPORT_FILE)

def _print_trace(trace):
    if not trace:
        print("      (no traversal steps recorded)"); return
    for t in trace:
        ev = t.get("event")
        if ev == "mode":
            if t.get("synthesis"):
                print(f"      MODE: synthesis (breadth over depth) — {clip(t.get('why',''),80)}")
            continue
        if ev == "descend":
            print(f"      step {t['step']:>2} @ {t['at']} [{t.get('node_type','')}] -> descend: {t['chose']} (score {t['chose_score']})")
            if t.get("reasoning"): print(f"           reason: {clip(t['reasoning'],150)}")
            rk = t.get("ranked", [])
            if len(rk) > 1:
                print("           ranked: " + "  |  ".join(f"{r['score']:.2f} {r['node']}" for r in rk[:6])
                      + (f"  (+{len(rk)-6})" if len(rk) > 6 else ""))
        elif ev == "read_file":
            print(f"      step {t['step']:>2} read {t['at']} -> {t['decision'].upper()}")
            if t.get("reasoning"): print(f"           reason: {clip(t['reasoning'],150)}")
            if t.get("granularity"):
                if t.get("kept_scope") == "section/file":
                    print(f"           granularity: answer needed wider context -> kept whole unit '{t.get('kept_unit','')}'")
                for g in t["granularity"]:
                    print(f"             - {g['from']} -> {g['to']}: {g['scope']}"
                          + (f" ({clip(g['reasoning'],80)})" if g.get('reasoning') else ""))
            for sw in t.get("intra_file_sweep", []):
                if "triaged" in sw:
                    print(f"           intra-file triage over {len(sw['triaged'])} candidate(s) (heading+score shown to agent)")
                elif "selected" in sw:
                    auto = sw.get("auto_selected", [])
                    if auto:
                        print(f"           auto-read {len(auto)} high-scoring section(s) (cluster):")
                        for e in auto:
                            print(f"             * {e['score']:.2f}  {e['node']}")
                    if sw["selected"]:
                        print(f"           agent additionally selected {len(sw['selected'])} to read:")
                        for e in sw["selected"]:
                            print(f"             · {e['score']:.2f}  {e['node']}")
                    if not auto and not sw["selected"]:
                        print("           nothing further selected from this file")
                elif "read" in sw:
                    mark = "+" if sw.get("decision") == "take" else "-"
                    detail = sw.get("adds") if sw.get("decision") == "take" else sw.get("reasoning","")
                    print(f"           {mark} intra-file {sw.get('decision')} ({sw['score']:.2f}): {sw['read']}"
                          + (f" — {clip(detail,90)}" if detail else ""))
        elif ev == "deferred_take":
            print(f"      + recovered deferred ({t['score']:.2f}): {t['node']}")
            print(f"           adds: {clip(t.get('adds',''),110)}")
        elif ev == "deferred_skip":
            print(f"      - deferred rejected ({t['score']:.2f}): {t['node']}")
        elif ev == "synthesis_breadth":
            print(f"      step {t['step']:>2} synthesis: {t.get('n_files')} distinct document(s) gathered so far")
        elif ev == "sufficiency_check":
            print(f"      step {t['step']:>2} sufficiency: {'ENOUGH -> answer' if t['sufficient'] else 'not enough -> keep searching'}"
                  + (f" — {clip(t['reasoning'],120)}" if t.get('reasoning') else ""))
        elif ev == "teleport":
            tgt = t.get("target"); ts = t.get("target_score")
            print(f"      step {t['step']:>2} TELEPORT from {t['from']} -> {tgt} (score {ts})")
            fr = t.get("frontier", [])
            if fr:
                print("           frontier: " + "  |  ".join(f"{f['score']:.2f} {f['node']}" for f in fr[:6])
                      + (f"  (+{len(fr)-6})" if len(fr) > 6 else ""))

# ---- diagnosis: qms beat treerag. Where was qms's file in our traversal, and what would fix it? ----
def _rank_of_file_in_trace(trace, target_paths):
    """Locate the correct file (the one qms_search read) inside TreeRAG's traversal and report how
    far off we were. Returns (found, detail) where detail includes the best relevance score we ever
    gave that file, the LOCAL rank within the pool it appeared in, AND a GLOBAL rank: where that
    score sat among ALL distinct scores TreeRAG produced across the whole traversal (1 = we scored
    it highest of everything). Used only when qms wins on a DIFFERENT file."""
    names = [Path(str(p)).name.lower() for p in target_paths] + [str(p).lower() for p in target_paths]
    def _hit(node_name):
        nl = str(node_name).lower()
        return any(n and (n in nl or nl in n) for n in names)
    all_scores = []          # every score surfaced anywhere (for the global ranking)
    best = None              # best (highest-scoring) appearance of the correct file
    for t in trace:
        pools = []
        if t.get("event") == "descend": pools = t.get("ranked", [])
        elif t.get("event") == "teleport": pools = t.get("frontier", [])
        for pos, r in enumerate(pools):
            all_scores.append(r["score"])
            if _hit(r["node"]):
                rec = {"step": t["step"], "event": t["event"], "rank_pos": pos+1,
                       "of": len(pools), "score": r["score"], "node": r["node"]}
                if best is None or rec["score"] > best["score"]: best = rec
    if best is not None:
        distinct = sorted({round(s,4) for s in all_scores}, reverse=True)
        try: best["global_rank"] = distinct.index(round(best["score"],4)) + 1
        except ValueError: best["global_rank"] = None
        best["global_of"] = len(distinct)
    return (best is not None, best)

def build_failure_dossier(q, tree_answer, tree_result, qms_answer, qms_files_full, same_file,
                          tree_acc, qms_acc, dt, counter, answer_debug, redact=None):
    """Deterministic, DEIDENTIFIED record of one qms-beats-treerag question. No LLM involved — the
    in-loop diagnoser was dropped because it confabulated mechanisms (e.g. blaming summary scoring
    when the real, visible failure was the answer model exhausting its token budget into DUNNO).
    Instead this captures the salient raw signals and leaves the diagnosis to the external agent:
      - the full traversal event sequence (descends with ranked scores, takes/rejects, granularity,
        triage counts, sufficiency verdicts, teleports, floor dismissals)
      - where the correct (qms) file sat in OUR ranking, if it appeared at all
      - answer-assembly warnings (reasoning exhaustion, DUNNO-despite-evidence, retries)
      - timings, call counts, evidence counts, answer lengths
    Deidentification: document/section names are replaced by stable per-dossier placeholders
    (DOC_1, DOC_1/SEC_2, ...), and no document, answer, or (by default) question text is included —
    only lengths and structure."""
    _redact = redact if redact is not None else _make_redactor()

    found, where = _rank_of_file_in_trace(tree_result.get("trace", []), qms_files_full)
    ev_summary = []
    for t in tree_result.get("trace", []):
        e = t.get("event")
        if e == "descend":
            ev_summary.append({"e": "descend", "to": _redact(t.get("chose")), "score": t.get("chose_score"),
                               "pool": [{"n": _redact(r.get("node")), "s": r.get("score")} for r in t.get("ranked", [])[:6]]})
        elif e == "read_file":
            r = {"e": "read", "node": _redact(t.get("at")), "decision": t.get("decision")}
            if t.get("kept_scope"): r["granularity"] = t["kept_scope"]
            ev_summary.append(r)
            for sw in t.get("intra_file_sweep", []):
                if "triaged" in sw: ev_summary.append({"e": "triage", "candidates": len(sw["triaged"])})
                elif "floor_dismissed" in sw: ev_summary.append({"e": "floor_dismissed", "n": sw["floor_dismissed"]})
                elif "selected" in sw: ev_summary.append({"e": "triage_selected", "n": len(sw["selected"]),
                                                              "auto": len(sw.get("auto_selected", []))})
                elif "read" in sw: ev_summary.append({"e": "triage_read", "node": _redact(sw["read"]),
                                                      "score": sw.get("score"), "decision": sw.get("decision")})
        elif e == "sufficiency_check":
            ev_summary.append({"e": "sufficiency", "sufficient": t.get("sufficient"), "n_evidence": t.get("n_evidence")})
        elif e == "teleport":
            ev_summary.append({"e": "teleport", "to": _redact(t.get("target")), "score": t.get("target_score"),
                               "frontier_size": len(t.get("frontier", []))})
    warnings = []
    for att in (answer_debug or {}).get("attempts", []):
        if att.get("reasoning_exhausted"): warnings.append(f"reasoning_exhausted@attempt{att.get('n')}")
    if (answer_debug or {}).get("dunno_with_evidence"): warnings.append("dunno_with_evidence")
    _cov = (answer_debug or {}).get("coverage") or {}
    if _cov.get("missing"):
        warnings.append(f"coverage_missing_{len(_cov['missing'])}_of_{len(_cov.get('elements') or [])}")
    if _cov.get("repaired"): warnings.append("coverage_repaired")
    if _cov.get("missing_after"): warnings.append(f"coverage_incomplete_after_repair_{len(_cov['missing_after'])}")
    if _cov.get("repair_failed"): warnings.append("coverage_repair_failed")
    if (answer_debug or {}).get("compressed"): warnings.append("length_compressed")
    if (answer_debug or {}).get("compression_rejected"):
        warnings.append(f"compression_rejected_{answer_debug['compression_rejected'].replace(' ','_')}")
    _w = (answer_debug or {}).get("words")
    if isinstance(_w, int) and _w > ANSWER_MAX_WORDS: warnings.append(f"answer_long_{_w}w")
    tree_ans = tree_answer or ""
    dossier = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "qid": hashlib.md5(str(q.qid).encode()).hexdigest()[:10],
        "tree_score": tree_acc, "qms_score": qms_acc,
        "seconds": round(dt, 1), "llm_calls": counter.calls if counter else None,
        "steps": tree_result.get("steps"), "teleports": tree_result.get("teleports"),
        "n_evidence": len(tree_result.get("evidence", [])),
        "same_file_as_qms": same_file,
        "qms_file_in_our_ranking": ({"seen": True, "score": where.get("score"),
                                     "local_rank": f"{where.get('rank_pos')}/{where.get('of')}",
                                     "global_rank": (f"{where.get('global_rank')}/{where.get('global_of')}"
                                                     if where.get("global_rank") else None),
                                     "at_step": where.get("step")} if found else {"seen": False}),
        "answer_len_words": len(tree_ans.split()), "answer_was_dunno": tree_ans.strip() == DUNNO,
        "qms_answer_len_words": len((qms_answer or "").split()),
        "answer_debug": dict(answer_debug or {}),
        "warnings": warnings,
        "trace": ev_summary,
    }
    if DOSSIER_INCLUDE_QUESTION:
        dossier["question"] = q.stem
    else:
        dossier["question_len_words"] = len(q.stem.split())
    return dossier


import sys
_REAL_STDOUT = sys.stdout
LOG_DIR = Path("failure_logs"); LOG_DIR.mkdir(exist_ok=True)

class _ConsoleTee:
    """Mirror of stdout that also keeps the text, so each question's exact console output can be
    attached (deidentified) to its failure dossier."""
    def __init__(self, real): self.real = real; self._buf = []
    def write(self, s): self.real.write(s); self._buf.append(s)
    def flush(self): self.real.flush()
    def text(self): return "".join(self._buf)

def _make_redactor():
    """Stable per-question placeholder mapping shared by the dossier AND the console log, so DOC_3
    means the same document in both."""
    names = {}
    def _redact(name):
        name = str(name or "?")
        doc, _, rest = name.partition(" - ")
        if doc not in names: names[doc] = f"DOC_{len([k for k in names if ' - ' not in k])+1}"
        did = names[doc]
        if not rest: return did
        key = doc + " - " + rest
        if key not in names:
            n_sec = sum(1 for k in names if k.startswith(doc + " - "))
            names[key] = f"{did}/SEC_{n_sec+1}"
        return names[key]
    return _redact

def _deidentify_console(text, q, redact, counter, max_chars=60000):
    """Deidentify one question's raw console output for the improvement agent.
    Pass 1 (deterministic — guarantees exactness and placeholder consistency): every corpus node
    name that appears is replaced via the SAME redactor the dossier used; the organization, the
    question text, and the gold option texts are replaced with tagged placeholders.
    Pass 2 (local gpt-oss sweep): a strict reproduce-verbatim prompt that may ONLY placeholder
    residual sensitive strings (quoted document sentences, names pass 1 didn't know). Everything
    else — scores, steps, timings, warnings, structure, wording — must survive character-for-
    character. If a chunk's sweep fails, the pass-1 (already name-scrubbed) chunk is kept."""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[log truncated for deidentification]\n"
    present = sorted({n.name for n in _NODES.values() if n.name and n.name in text}, key=len, reverse=True)
    for nm in present:
        text = text.replace(nm, redact(nm))
    for s in (ORGANIZATION, "Ontario Institute for Cancer Research", "OICR"):
        text = text.replace(s, "[ORG]")
    if q.stem:
        text = text.replace(q.stem, f"[QUESTION, {len(q.stem.split())} words]")
    try:
        for gt in gold_texts(q):
            if gt and gt in text: text = text.replace(gt, "[CORRECT_OPTION_TEXT]")
    except Exception: pass
    out, CH = [], 2500
    for i in range(0, len(text), CH):
        chunk = text[i:i+CH]
        prompt = (
            "You are deidentifying a system log chunk. Reproduce it EXACTLY, character-for-character, "
            "with ONLY this change: replace any remaining real document/file/section titles, "
            "organization or person names, and directly quoted document sentences with neutral "
            "placeholders like [NAME] or [QUOTED TEXT]. KEEP UNCHANGED: all scores, step numbers, "
            "timings, counts, warnings, log structure and wording, placeholders already present "
            "(DOC_n, SEC_n, [ORG], [QUESTION...], [CORRECT_OPTION_TEXT]), and the standard phrase "
            "'The requested information could not be found.' Do NOT rephrase, summarize, reformat, "
            "shorten, or comment. Output ONLY the transformed chunk.\n\nLOG CHUNK:\n" + chunk)
        try:
            r = llm(prompt, counter, num_predict=4096, temperature=0, think=False, thinking_fallback=False)
        except Exception:
            r = ""
        out.append(r if r and len(r) >= 0.5 * len(chunk) else chunk)  # reject collapsed/failed sweeps
    return "".join(out)

report = _load_report()
by_qid = {r.get("qid"): r for r in report["results"]}
qmap = {q.qid: q for q in QUESTIONS}

def _lost_to_qms(r):
    ta, qa = r.get("treerag_accuracy"), r.get("qms_accuracy")
    return isinstance(ta, (int, float)) and isinstance(qa, (int, float)) and qa > ta

# processing order: FIRST re-run every already-recorded qms-loss, but ORDERED so the most
# diagnostic cases come first — the ones where treerag actually reached the SAME correct file as
# qms yet produced a worse answer (`same_file` True) go before the ones where it landed on the
# WRONG file (`same_file` False / unknown). Same-file losses isolate answer-assembly / granularity
# regressions independent of navigation, so re-testing them first tells you fastest whether a fix
# helped; wrong-file losses (a navigation/ranking miss) come after. Report order is preserved
# within each group. THEN the questions not yet recorded, in their normal order. Recorded wins/ties
# are kept untouched. A re-run REPLACES that question's entry in the report.
# QUEUE ORDER: same-file qms-losses  ->  priority (GEN-2)  ->  wrong-file qms-losses  ->  PENDING,
# where PENDING is one merged pool of the never-answered questions AND the modified ones (referential
# answers just resolved), taken together in questions.json order. A modified question is treated
# exactly like an unrecorded one — same pool, no separate pass.
PRIORITY_RERUN_QIDS = ["GEN-99"]   # always re-run these in their slot, IN THIS ORDER, whether or not
                                  # they are recorded as a qms loss

# resolve priority qids case-insensitively so "qms-33" also matches a "QMS-33" in questions.json
_qmap_ci = {str(k).lower(): k for k in qmap}
_prio, _missing_prio = [], []
for _qid in PRIORITY_RERUN_QIDS:
    _hit = _qid if _qid in qmap else _qmap_ci.get(str(_qid).lower())
    if _hit and _hit not in _prio: _prio.append(_hit)
    elif not _hit: _missing_prio.append(_qid)
_pset = set(_prio)

# a MODIFIED question counts as unanswered until it has been graded ONCE since modification. Each
# result stores "graded_as_modified"; a modified question whose recorded entry lacks that stamp (or
# has no entry, e.g. the resolver purged it) is pending.
def _graded_as_modified(qid): return bool(by_qid.get(qid, {}).get("graded_as_modified"))
def _is_pending(q):
    if q.qid in _pset: return False
    if q.qid not in by_qid: return True                                   # never answered
    return bool(getattr(q, "modified", False)) and not _graded_as_modified(q.qid)   # modified, not yet regraded

pending_qids = [q.qid for q in QUESTIONS if _is_pending(q)]   # unanswered + modified, one pool
_pendset = set(pending_qids)
_n_mod_pending = sum(1 for q in QUESTIONS if _is_pending(q) and getattr(q, "modified", False))

# recorded qms-losses, excluding anything already claimed by the priority or pending pools
_losses = [r for r in report["results"] if _lost_to_qms(r) and r.get("qid") in qmap
           and r["qid"] not in _pset and r["qid"] not in _pendset]
rerun_same = [r["qid"] for r in _losses if r.get("same_file")]        # right file, worse answer
rerun_diff = [r["qid"] for r in _losses if not r.get("same_file")]    # wrong file (or unknown)

rerun_qids = rerun_same + rerun_diff
queue = [qmap[qid] for qid in rerun_qids + pending_qids]
# __IMPROVE_INSTRUMENT__
import os as _impos
_only = _impos.environ.get('IMPROVE_ONLY_QIDS','').strip()
if _only:
    _oq = [x.strip() for x in _only.split(',') if x.strip()]
    _qm_all = {q.qid: q for q in QUESTIONS}
    queue = [_qm_all[x] for x in _oq if x in _qm_all]
    print(f'[improve] queue overridden to {len(queue)} qid(s)')
_kept = len(by_qid) - len({qid for qid in rerun_qids + pending_qids if qid in by_qid})
if _missing_prio: print(f"warning: priority qid(s) not in {QUESTIONS_FILE}, skipped: {_missing_prio}")
print(f"resuming: {len(by_qid)} recorded in {REPORT_FILE} — keeping {_kept} win/tie entr(ies), RE-RUNNING "
      f"{len(rerun_qids)}: {len(rerun_same)} same-file/worse-answer, then priority {_prio}, then "
      f"{len(rerun_diff)} wrong-file — then {len(pending_qids)} pending "
      f"({_n_mod_pending} modified + {len(pending_qids)-_n_mod_pending} unanswered, interleaved in "
      f"question order)\n")

# ---- live per-question progress ticker (heuristic ETA) --------------------------------------
# run_agent(live=False) is a long SILENT window (no prints until it returns), so a background
# thread can safely animate a one-line progress bar on the REAL stdout without scrambling output
# or polluting the per-question console capture (which only tees sys.stdout). The per-question bar
# fills elapsed/estimate where the estimate is a rolling MEDIAN of recent run_agent times (robust
# to the odd 800s outlier); the run-level ETA uses a rolling MEAN of full per-question times.
# Both windows seed from the durations already recorded in REPORT_FILE, so question 1 isn't blank.
import threading, statistics

class _QTicker(threading.Thread):
    def __init__(self, i, n, qid, est, win_full, out):
        super().__init__(daemon=True)
        self.i, self.n, self.qid, self.est = i, n, qid, est
        self.win_full, self.out = win_full, out
        self._ev = threading.Event(); self.t0 = time.perf_counter()
    @staticmethod
    def _fmt(s):
        s = int(max(0, s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"
    def run(self):
        while not self._ev.is_set():
            el = time.perf_counter() - self.t0
            if self.est and self.est > 0:
                frac = min(1.0, el / self.est); nb_ = int(frac * 14)
                bar = "[" + "#" * nb_ + "·" * (14 - nb_) + f"] {int(frac*100):>3d}%"
                ests = "~" + self._fmt(self.est)
            else:
                bar = "[  measuring…  ]"; ests = "~?"
            avg = statistics.mean(self.win_full) if self.win_full else (self.est or 0)
            eta = (self.n - self.i) * avg + max(0.0, (self.est or avg) - el)   # rest of run + rest of this q
            self.out.write(f"\r  ⏳ [{self.i}/{self.n}] {self.qid}  {bar}  "
                           f"this q {self._fmt(el)}/{ests}  ·  run eta ~{self._fmt(eta)}      ")
            self.out.flush(); self._ev.wait(0.5)
    def stop(self):
        if self._ev.is_set(): return
        self._ev.set()
        try: self.out.write("\r" + " " * 120 + "\r"); self.out.flush()
        except Exception: pass

_WIN = 15
_win_full  = [r["seconds"] for r in report["results"] if isinstance(r.get("seconds"), (int, float))][-_WIN:]
_win_agent = []   # run_agent-only seconds, learned as we go (fall back to _win_full until we have some)

try:
    for i, Q in enumerate(queue, 1):
        t0 = time.perf_counter(); counter = Counters(); tree_errored = False; tree_err = None
        sys.stdout = _ConsoleTee(_REAL_STDOUT)   # capture this question's exact console output
        print("="*94)
        print(f"[{i}/{len(queue)}] {Q.qid}: {clip(Q.stem,110)}"
              + ("   [PRIORITY RE-RUN]" if Q.qid in _pset else
                 "   [MODIFIED: referential answer resolved -> reprocessing]"
                     if (Q.qid in _pendset and getattr(Q, "modified", False)) else
                 "" if Q.qid in _pendset else
                 (("   [RE-RUN: same file as qms, worse answer]" if by_qid[Q.qid].get("same_file")
                   else "   [RE-RUN: qms found a different file]") if Q.qid in by_qid else "")))
        # (1) correct answer as TEXT, not letters
        print(f"      correct answer: {' | '.join(gold_texts(Q)) or '(none)'}")

        qrec = QMS.get(Q.qid)
        qms_answer = qrec["answer"] if qrec else None
        qms_files_full = qrec["files_read_full"] if qrec else []
        print("      --- qms_search ---")
        if qrec:
            print(f"      files read: {qms_files_full if qms_files_full else '(none listed)'}")
        else:
            print("      (no qms entry for this qid)")

        print("      --- treerag traversal ---")
        _est = (statistics.median(_win_agent) if len(_win_agent) >= 2
                else (statistics.median(_win_full) if _win_full else None))
        _ticker = _QTicker(i, len(queue), Q.qid, _est, list(_win_full), _REAL_STDOUT)
        _ticker.start(); _t_agent = time.perf_counter()
        try:
            result = run_agent(Q, counter, live=False)
            _win_agent.append(time.perf_counter() - _t_agent); del _win_agent[:-_WIN]
            _ticker.stop(); _ticker.join(timeout=1)
            _print_trace(result.get("trace", []))
            print(f"      path: {result['path']}")
            print(f"      steps {result['steps']}, teleports {result['teleports']}, evidence {len(result['evidence'])}")
            evidence = result["evidence"]
            response = answer_oicr(Q, evidence, counter)
        except Exception as e:
            _ticker.stop(); _ticker.join(timeout=1)
            result = {"path":"", "steps":0, "teleports":0, "trace":[], "evidence":[]}
            evidence, response = [], f"(error: {type(e).__name__}: {e})"
            tree_errored = True; tree_err = e
            print(f"      ERROR: {e}")

        # ONE joint judge call grades BOTH responses on the same rubric (consistency), each on its
        # own merit. On a treerag error we score tree 0 and judge qms alone.
        if tree_errored:
            tree_acc, tree_reason = 0.0, f"error during treerag: {tree_err}"
            if qms_answer is not None:
                try: qms_acc, qms_reason = judge_score(Q, qms_answer, Counters())
                except Exception as e: qms_acc, qms_reason = None, f"error during qms judge: {e}"
            else:
                qms_acc, qms_reason = None, "no qms answer for this qid"
        else:
            try:
                tree_acc, tree_reason, qms_acc, qms_reason = judge_both(Q, response, qms_answer, Counters())
            except Exception as e:
                tree_acc, tree_reason = 0.0, f"error during joint judge: {e}"
                qms_acc, qms_reason = (None, "joint judge error") if qms_answer is None else (0.0, f"error during joint judge: {e}")

        # (2) print BOTH full answers under the traversal
        print("      --- answers ---")
        print("      [treerag answer]"); print(textwrap.indent(textwrap.fill(response or "(empty)", 96), "        "))
        print("      [qms answer]");     print(textwrap.indent(textwrap.fill(qms_answer or "(none)", 96), "        "))

        # (5) if qms won, record a deterministic deidentified failure dossier (no LLM diagnosis)
        dossier = None
        tree_sources = {Path(str(e.metadata.get("source_file") or e.path or e.name)).name.lower() for e in evidence}
        qms_names    = {Path(str(p)).name.lower() for p in qms_files_full}
        same_file = bool(tree_sources & qms_names)
        dt = time.perf_counter() - t0
        _win_full.append(dt); del _win_full[:-_WIN]   # feed run-level ETA window
        if isinstance(qms_acc, float) and isinstance(tree_acc, float) and qms_acc > tree_acc:
            print("      --- qms beat treerag: writing failure dossier ---")
            try:
                _redactor = _make_redactor()
                dossier = build_failure_dossier(Q, response, result, qms_answer, qms_files_full,
                                                same_file, tree_acc, qms_acc, dt, counter, ANSWER_DEBUG,
                                                redact=_redactor)
                # attach the question's DIRECT console output, deidentified (deterministic name scrub
                # with the same placeholder mapping as the dossier, then a strict local-gpt-oss sweep),
                # so the improvement agent can read exactly what happened — not just the summary
                try:
                    q_console = sys.stdout.text() if isinstance(sys.stdout, _ConsoleTee) else ""
                    if q_console:
                        deid = _deidentify_console(q_console, Q, _redactor, Counters())
                        log_path = LOG_DIR / f"{dossier['ts'].replace(':','-')}_{dossier['qid']}.log"
                        log_path.write_text(deid, encoding="utf-8")
                        dossier["console_log"] = str(log_path)
                        print(f"      (deidentified console saved to {log_path})")
                except Exception as e:
                    print(f"      (console deidentification failed: {e})")
                loc = dossier.get("qms_file_in_our_ranking") or {}
                if loc.get("seen"):
                    print(f"      correct file's relevance in our traversal: score {loc.get('score')}"
                          + (f" (global rank {loc.get('global_rank')})" if loc.get('global_rank') else "")
                          + f", local {loc.get('local_rank')} at step {loc.get('at_step')}")
                elif not same_file:
                    print("      correct file never appeared in our ranking/frontier")
                if dossier.get("warnings"):
                    print(f"      warnings: {', '.join(dossier['warnings'])}")
                try:
                    try:
                        _dl = json.loads(Path(DOSSIER_FILE).read_text(encoding="utf-8"))
                        if not isinstance(_dl, list): _dl = []
                    except Exception:
                        _dl = []
                    _dl.append(dossier)
                    _tmp = Path(DOSSIER_FILE + ".tmp")
                    _tmp.write_text(json.dumps(_dl, indent=1, ensure_ascii=False), encoding="utf-8")
                    _tmp.replace(DOSSIER_FILE)
                    print(f"      (dossier {len(_dl)} appended to {DOSSIER_FILE})")
                except Exception as e:
                    print(f"      (could not write {DOSSIER_FILE}: {e})")
            except Exception as e:
                print(f"      dossier error: {e}")

        report["results"] = [r for r in report["results"] if r.get("qid") != Q.qid]
        report["results"].append({
            "qid": Q.qid, "question": Q.stem,
            "correct_option_texts": gold_texts(Q),
            "treerag_answer": response, "treerag_accuracy": tree_acc, "treerag_reason": tree_reason,
            "treerag_answer_debug": dict(ANSWER_DEBUG),
            "treerag_path": result.get("path",""), "treerag_trace": result.get("trace", []),
            "treerag_evidence_sources": sorted({e.metadata.get("source_file") or e.path or e.name for e in evidence}),
            "qms_answer": _cite_paths(qms_answer, qms_files_full), "qms_files_read_full": qms_files_full,
            "qms_accuracy": qms_acc, "qms_reason": qms_reason,
            "same_file": same_file, "dossier": dossier,
            "n_evidence": len(evidence), "seconds": round(dt,1), "llm_calls": counter.calls,
            "in_tokens": counter.in_tok, "out_tokens": counter.out_tok,
            "graded_as_modified": bool(getattr(Q, "modified", False)),
            "treerag_cited_docs": re.findall(r"\[([^\]\[]+)\]", response or ""),
        })
        _save_report(report)
        try: open("improve/run_logs/heartbeat.log","a").write(f"{time.strftime('%H:%M:%S')} {Q.qid} tree={tree_acc:.3f} qms={qms_acc if isinstance(qms_acc, float) else "-"} {dt:.0f}s\n")
        except Exception: pass
        if _impos.environ.get('IMPROVE_STOP_AT_FIRST_LOSS')=='1' and isinstance(qms_acc,float) and isinstance(tree_acc,float) and qms_acc>tree_acc:
            sys.stdout=_REAL_STDOUT; print(f'[improve] first loss at {Q.qid}; stopping early'); break
        qms_str = f"{qms_acc:.3f}" if isinstance(qms_acc, float) else "  -  "
        print(f"      => tree {tree_acc:.3f} | qms {qms_str} | {dt:.1f}s  (saved, {len(report['results'])} total)\n")
        sys.stdout = _REAL_STDOUT
except KeyboardInterrupt:
    sys.stdout = _REAL_STDOUT
    print(f"\ninterrupted; {len(report['results'])} question(s) saved to {REPORT_FILE}")
finally:
    sys.stdout = _REAL_STDOUT

print(f"\ndone. {len(report['results'])} question(s) in {REPORT_FILE}")


# In[11]:


# ---- summary: accuracy per approach + aggregated diagnoses from every qms-win ----
report = json.loads(Path(REPORT_FILE).read_text(encoding="utf-8"))
df = pd.DataFrame(report["results"])
paired = df.dropna(subset=["treerag_accuracy", "qms_accuracy"])

summary = pd.DataFrame([{
    "n_questions": len(df),
    "n_paired": len(paired),
    "treerag_mean": df["treerag_accuracy"].dropna().mean(),
    "treerag_std":  df["treerag_accuracy"].dropna().std(),
    "qms_mean":     df["qms_accuracy"].dropna().mean(),
    "qms_std":      df["qms_accuracy"].dropna().std(),
    "mean_diff_tree_minus_qms": (paired["treerag_accuracy"] - paired["qms_accuracy"]).mean(),
}])
print(f"benchmark over {len(df)} completed question(s) ({len(paired)} scored on both):\n")
print(summary.to_string(index=False))

if len(paired):
    tw = int((paired["treerag_accuracy"] > paired["qms_accuracy"]).sum())
    qw = int((paired["qms_accuracy"] > paired["treerag_accuracy"]).sum())
    print(f"\nhead-to-head on {len(paired)} paired: treerag wins {tw}, qms wins {qw}, ties {len(paired)-tw-qw}")

# ---- aggregate the failure dossiers (deterministic signals; the agent does the diagnosing) ----
doss = [r["dossier"] for r in report["results"] if r.get("dossier")]
if doss:
    print(f"\n{'='*70}\nFAILURE DOSSIERS from {len(doss)} question(s) where qms beat treerag\n{'='*70}")
    nav_missed = sum(1 for d in doss if not d.get("same_file_as_qms") and not (d.get("qms_file_in_our_ranking") or {}).get("seen"))
    nav_ranked = sum(1 for d in doss if not d.get("same_file_as_qms") and (d.get("qms_file_in_our_ranking") or {}).get("seen"))
    ans_fail   = sum(1 for d in doss if d.get("same_file_as_qms"))
    print(f"  navigation — qms file NEVER surfaced in our ranking/frontier: {nav_missed}")
    print(f"  navigation — qms file surfaced but scored too low to pick:    {nav_ranked}")
    print(f"  answer     — same file reached but treerag's answer was worse: {ans_fail}")
    warn_counts = Counter(w for d in doss for w in d.get("warnings", []))
    if warn_counts: print("  answer-assembly warnings:", dict(warn_counts))
    dunno = sum(1 for d in doss if d.get("answer_was_dunno"))
    if dunno: print(f"  answers that were the DUNNO string: {dunno}")
    slow = sorted(doss, key=lambda d: -(d.get("seconds") or 0))[:3]
    print("  slowest failures:", ", ".join(f"{d.get('seconds')}s ({d.get('llm_calls')} calls)" for d in slow))
    print(f"\n(full deidentified dossiers in {DOSSIER_FILE} — consumed by improve_loop.sh)")
else:
    print("\n(no qms-wins yet, so no failure dossiers)")
summary
