#!/usr/bin/env python
# coding: utf-8

# # TreeRAG prototype 5 — whole-corpus summary tree (phased, no model thrash)
# 
# Builds one summary tree over **every** file under `DOCS_ROOT`, nesting bottom-up:
# 
# ```
# unit (paragraph / table / figure) -> section -> document -> folder -> ... -> ROOT
# ```
# 
# Same structure-aware ingestion as prototype 2 (markdown headings, tables as
# their own units, figures captioned by the vision model, logos skipped, page
# numbers filtered, paragraphs/tables stitched across page breaks).
# 
# **Why this is faster than prototype 4.** Two models are involved — `gpt-oss:120b`
# for text and a multimodal model for figures — and a 120B model is too big to sit
# in GPU memory next to the vision model. Prototype 4 interleaved text and image
# calls, so Ollama kept **evicting and reloading the 120B between calls** (a single
# reload can take minutes — that was the ~600 s figure spikes). Prototype 5 runs in
# **three phases so each model loads once**:
# 
# 1. **Figures** — describe every figure with `VISION_MODEL` (loads once);
# 2. **Text** — summarise every paragraph/table with `SUMMARY_MODEL` (loads once);
# 3. **Combine** — build sections -> documents -> folders -> ROOT (`SUMMARY_MODEL`,
#    already resident).
# 
# That turns hundreds of model swaps into ~2. Each phase warms its model first, runs
# in parallel, and shares the per-node cache, so the build stays crash-resumable and
# a resume does zero work. The single progress bar shows the current phase plus a
# live ETA. Set `DESCRIBE_IMAGES = False` to skip figures and run on gpt-oss only.

# In[1]:


# Dependencies are provided by the qms_search virtual environment.  The source
# prototype was exported from a notebook and installed them through IPython.


# In[2]:


import importlib.util, sys, os, json, hashlib, re, time, textwrap, base64, threading
import concurrent.futures as cf
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

required = {
    "ollama": "ollama", "fitz": "pymupdf", "pymupdf4llm": "pymupdf4llm",
    "docx": "python-docx", "pandas": "pandas", "openpyxl": "openpyxl",
    "rank_bm25": "rank-bm25", "tqdm": "tqdm", "numpy": "numpy",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"Missing — run:  pip install {' '.join(missing)}")
    raise SystemExit(1)

import ollama
import fitz                          # PyMuPDF
import docx as _docx
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

try:
    import pymupdf4llm
    HAVE_PYMUPDF4LLM = True
except Exception as _e:
    HAVE_PYMUPDF4LLM = False
    print(f"pymupdf4llm import failed ({_e}); PDF loader will fall back to a basic parse.")

print("All packages OK" + ("" if HAVE_PYMUPDF4LLM else "  (pymupdf4llm missing)"))


# In[ ]:


# ---------------------------------------------------------------------------
# Ollama / models
# ---------------------------------------------------------------------------
OLLAMA_URL    = os.environ.get("TREEQUEST_OLLAMA_URL", "http://127.0.0.1:11434")
SUMMARY_MODEL = os.environ.get("TREEQUEST_MODEL", "gpt-oss:120b")
VISION_MODEL  = "gemma3:27b"               # figure descriptions (gpt-oss is text-only!)
AGENT_MODEL   = "gpt-oss:120b"             # traversal decisions (later stages)
CHAT_MODEL    = "gpt-oss:120b"             # final answer synthesis (later stages)
EMBED_MODEL   = "nomic-embed-text"         # vector-search baseline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_ROOT       = Path(os.environ.get("TREEQUEST_DOCS_ROOT", "folders"))
CACHE_DIR       = Path(os.environ.get("TREEQUEST_CACHE_DIR", "tree_cache"))
TREE_FILE       = CACHE_DIR / "corpus_tree.json"
NODE_CACHE_DIR  = CACHE_DIR / "nodes"      # per-node cache (crash-resumable)
PARSE_CACHE_DIR = CACHE_DIR / "parse"      # parsed-document cache (skip re-parsing)
IMG_DIR         = CACHE_DIR / "images"     # extracted figures (per document)

# ---------------------------------------------------------------------------
# Speed / parallelism
# ---------------------------------------------------------------------------
NUM_WORKERS     = int(os.environ.get("TREEQUEST_BUILD_WORKERS", "4"))
VISION_WORKERS  = 2      # concurrent figure requests (vision model is heavier per call)
GEN_NUM_PREDICT = 320    # max output tokens per summary (bounds per-call latency)
KEEP_ALIVE      = "30m"  # keep the large model resident between calls
DISABLE_THINKING = True  # gpt-oss is a reasoning model; skip visible thinking for speed
SKIP_SUMMARY_WORDS = 8   # text units this short are stored verbatim (no LLM call)

# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
DESCRIBE_IMAGES = False  # MultiHop-RAG articles are text-only; no vision calls
MD_DPI          = 150
LOGO_MIN_PX     = 100    # images with area < LOGO_MIN_PX**2 are treated as logos
DROP_LOGO_BY_DESC = True # also skip a figure if its description says it is a logo

# ---------------------------------------------------------------------------
# Progress / output
# ---------------------------------------------------------------------------
VERBOSE_CALLS      = False  # keep the public build log compact and auditable
CORPUS_PRINT_DEPTH = 1      # print only aggregate public-tree structure
SHOW_LEAVES_IN_TREE = False # include leaf paragraphs/tables/figures in the print

# ---------------------------------------------------------------------------
# Rebuild controls
# ---------------------------------------------------------------------------
FORCE_REBUILD = False     # clear the node cache and recompute all summaries
FORCE_REPARSE = False    # also re-parse the documents (set True if files changed)

for _d in (CACHE_DIR, NODE_CACHE_DIR, PARSE_CACHE_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print(f"Docs root  : {DOCS_ROOT.resolve()}")
print(f"Node cache : {NODE_CACHE_DIR}/   parse cache: {PARSE_CACHE_DIR}/")
print(f"Text model : {SUMMARY_MODEL}  (workers={NUM_WORKERS})")
print(f"Vision     : {VISION_MODEL}  (workers={VISION_WORKERS}, describe={DESCRIBE_IMAGES})")
print(f"Schedule   : phase 1 figures -> phase 2 text -> phase 3 combine (one load per model)")
print(f"Speedups   : parse+node cache, short-skip<={SKIP_SUMMARY_WORDS}w, "
      f"num_predict={GEN_NUM_PREDICT}, keep_alive={KEEP_ALIVE}")


# In[4]:


OLLAMA_TIMEOUT = 300   # seconds, per request (gpt-oss:120b can be slow to cold-load)
client = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)


def _model_names(list_resp) -> List[str]:
    raw = list_resp.get("models", []) if hasattr(list_resp, "get") else getattr(list_resp, "models", [])
    out = []
    for m in raw:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            out.append(name)
    return out


try:
    names = _model_names(client.list())
    print("Connected to Ollama. Available models:")
    for n in names:
        print(f"  {n}")
    _need = [SUMMARY_MODEL, EMBED_MODEL] + ([VISION_MODEL] if DESCRIBE_IMAGES else [])
    for required_model in _need:
        if not any(required_model in n for n in names):
            print(f"\nWARNING: '{required_model}' not found. Pull it with:")
            print(f"  OLLAMA_HOST={OLLAMA_URL} ollama pull {required_model}")
    print("\nTip: for real parallelism set OLLAMA_NUM_PARALLEL on the server "
          "(e.g. OLLAMA_NUM_PARALLEL=4) and match it with NUM_WORKERS.")
except Exception as e:
    print(f"Cannot reach Ollama at {OLLAMA_URL}: {type(e).__name__}: {e}")
    print("Run this script on the compute host that serves Ollama, or set")
    print("  TREEQUEST_OLLAMA_URL=http://<approved-ollama-host>:11434")


# In[5]:


# Warm the model (and confirm chat works). keep_alive keeps it resident after.
_t0 = time.time()
try:
    _r = client.chat(model=SUMMARY_MODEL,
                     messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                     options={"num_predict": 5}, keep_alive=KEEP_ALIVE)
    print(f"chat() OK in {time.time() - _t0:.1f}s  ->  {_r['message']['content'].strip()!r}")
except Exception as e:
    print(f"chat() FAILED after {time.time() - _t0:.1f}s: {type(e).__name__}: {e}")


# In[6]:


@dataclass
class TreeNode:
    """node_type: 'root' | 'folder' | 'document' | 'section' | 'chunk'.
    For 'chunk' nodes, metadata['kind'] is 'text' | 'table' | 'image';
    a skipped logo chunk carries metadata['skipped'] = 'logo'."""
    node_id:   str
    node_type: str
    name:      str
    path:      str
    summary:   str
    content:   str = ""
    children:  List["TreeNode"] = field(default_factory=list)
    metadata:  Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"node_id": self.node_id, "node_type": self.node_type, "name": self.name,
                "path": self.path, "summary": self.summary, "content": self.content,
                "children": [c.to_dict() for c in self.children], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: Dict) -> "TreeNode":
        node = cls(node_id=d["node_id"], node_type=d["node_type"], name=d["name"],
                   path=d["path"], summary=d["summary"], content=d.get("content", ""),
                   metadata=d.get("metadata", {}))
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node

    def is_leaf(self) -> bool:
        return self.node_type == "chunk"

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)

    def count_leaves(self) -> int:
        if self.is_leaf():
            return 1
        return sum(c.count_leaves() for c in self.children)


def _make_id(path: str, extra: str = "") -> str:
    return hashlib.md5(f"{path}|{extra}".encode()).hexdigest()[:12]

print("TreeNode class ready")


# In[7]:


# Central chat wrapper: one place for keep_alive, output cap, retries, and
# (for reasoning models like gpt-oss) turning off visible "thinking" for speed.
_THINK = {"use": DISABLE_THINKING}


def _chat(messages: List[Dict], num_predict: Optional[int] = None):
    opts = {"temperature": 0, "num_predict": num_predict or GEN_NUM_PREDICT}
    last = None
    for attempt in range(3):
        kw = dict(model=SUMMARY_MODEL, messages=messages, options=opts, keep_alive=KEEP_ALIVE)
        if _THINK["use"]:
            kw["think"] = False
        try:
            return client.chat(**kw)
        except TypeError:
            _THINK["use"] = False            # client too old for the think kwarg
        except Exception as e:
            last = e
            if _THINK["use"]:
                _THINK["use"] = False         # maybe the model rejects think -> drop and retry fast
            else:
                time.sleep(2 ** attempt)
    raise last if last else RuntimeError("chat failed")


def _words(s: str) -> List[str]:
    return re.findall(r"\S+", s or "")


def _enforce_not_longer(summary: str, source: str) -> str:
    """A summary must never be longer (in words) than its source; if it is, fall
    back to the source text. Equal-or-shorter is fine."""
    sw, mw = len(_words(source)), len(_words(summary))
    if sw and mw > sw:
        return source.strip()
    return summary


def _llm_summarise(text: str, context_hint: str = "") -> str:
    if not text.strip():
        return "(empty)"
    src_words = len(_words(text))
    cap = max(1, min(200, src_words))
    hint = f" The content comes from: {context_hint}." if context_hint else ""
    prompt = (
        f"Summarise the following text.{hint} "
        "Focus on key topics, concepts, and specific information (names, numbers, "
        "procedures, entities). If it is a Markdown table, state what it tabulates, "
        "its columns, and notable values. "
        f"Your summary MUST be shorter than the source and at most {cap} words; it "
        "need not reach that limit. If the source is very short, return it nearly "
        "verbatim or shorter, never longer. Respond with ONLY the summary.\n\n"
        f"{text}"
    )
    try:
        resp = _chat([{"role": "user", "content": prompt}])
        return _enforce_not_longer(resp["message"]["content"].strip(), text)
    except Exception as e:
        return f"(summary unavailable: {e})"


def _llm_combine_summaries(summaries: List[str], label: str) -> str:
    if not summaries:
        return "(no content)"
    joined = "\n".join(f"- {s}" for s in summaries)
    prompt = (
        f"You are summarising a section, document, or folder called '{label}'. Below "
        "are summaries of its contents. Write ONE specific summary of the overall "
        "scope and key topics. It MUST be shorter than the combined input below and "
        "at most 200 words; it need not reach that limit. Respond with ONLY the summary.\n\n"
        f"{joined}"
    )
    try:
        resp = _chat([{"role": "user", "content": prompt}])
        return _enforce_not_longer(resp["message"]["content"].strip(), joined)
    except Exception as e:
        return f"(summary unavailable: {e})"


def _resolve_image(image_path: str) -> Optional[Path]:
    if not image_path:
        return None
    p = Path(image_path)
    if p.exists():
        return p
    cand = IMG_DIR / p.name
    if cand.exists():
        return cand
    hits = list(IMG_DIR.rglob(p.name))
    return hits[0] if hits else None


def _llm_describe_image(image_path: str, caption_hint: str = "") -> str:
    p = _resolve_image(image_path)
    if p is None:
        base = f" Caption: {caption_hint}" if caption_hint else ""
        return f"(figure; image file not found){base}"
    hint = f" The figure's caption is: {caption_hint}." if caption_hint else ""
    prompt = (
        "Describe this figure from a document in 3-5 sentences so it can be found by "
        f"search.{hint} State the kind of visual (chart, diagram, photo, schematic, "
        "logo), what it depicts, any axis labels / units / labelled parts you can read, "
        "and the main takeaway. Respond with ONLY the description."
    )
    # NOTE: uses VISION_MODEL, not SUMMARY_MODEL — gpt-oss is text-only and cannot
    # accept images. No 'think' flag (gemma3 is not a reasoning model).
    for attempt in range(2):
        try:
            resp = client.chat(model=VISION_MODEL,
                               messages=[{"role": "user", "content": prompt, "images": [str(p)]}],
                               options={"temperature": 0, "num_predict": GEN_NUM_PREDICT},
                               keep_alive=KEEP_ALIVE)
            return resp["message"]["content"].strip()
        except Exception as e:
            if attempt == 1:
                base = f" Caption: {caption_hint}" if caption_hint else ""
                return f"(image description unavailable: {e}){base}"
            time.sleep(2 ** attempt)


def _llm_embed(text: str) -> Optional[List[float]]:
    try:
        return client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    except Exception:
        return None


print("LLM helpers ready (gpt-oss chat wrapper + summarise/combine/vision/embed)")


# In[8]:


def _node_cache_path(node_id: str) -> Path:
    return NODE_CACHE_DIR / f"{node_id}.json"


def _load_cached_node(node_id: str) -> Optional["TreeNode"]:
    p = _node_cache_path(node_id)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return TreeNode.from_dict(json.load(f))
    except Exception:
        return None


def _save_cached_node(node: "TreeNode") -> None:
    p = _node_cache_path(node.node_id)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(node.to_dict(), f, ensure_ascii=False)
    tmp.replace(p)        # atomic on POSIX — survives kill -9 mid-write


def clear_node_cache() -> int:
    n = 0
    for p in NODE_CACHE_DIR.glob("*.json"):
        p.unlink(); n += 1
    print(f"Cleared {n} cached node(s) from {NODE_CACHE_DIR}")
    return n


def cache_status() -> None:
    counts: Dict[str, int] = {}
    total_bytes = 0
    for p in NODE_CACHE_DIR.glob("*.json"):
        total_bytes += p.stat().st_size
        try:
            with open(p, encoding="utf-8") as f:
                k = json.load(f).get("node_type", "other")
            counts[k] = counts.get(k, 0) + 1
        except Exception:
            counts["other"] = counts.get("other", 0) + 1
    print(f"Per-node cache @ {NODE_CACHE_DIR}:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<10s}: {v}")
    print(f"  total size: {total_bytes / 1024:.1f} KB")


print("Per-node cache helpers ready")


# In[9]:


# ===========================================================================
# Structure-aware loaders (markdown-based).
#
#   * PDF  -> converted to Markdown with pymupdf4llm. Markdown HEADINGS define
#            the section hierarchy (nested by level). Markdown TABLES become
#            their own units (caption preserved + tabular structure kept).
#            Figures are extracted as image files and become their own units
#            (captioned later by the vision model in the builder).
#   * DOCX -> paragraphs + tables iterated in true reading order; heading
#            styles define the hierarchy; tables become their own units.
#   * Page-number / running-header artifacts ("Page 1 of 36") are filtered out
#            so they never become paragraphs.
#
# Output of load_structured(path) is a list of nested section dicts:
#   {"title": str, "level": int, "page": int|None,
#    "paragraphs": [ {"type": "text"|"table"|"image", "text"/"path": ...,
#                     "page": int|None, "caption": str?}, ... ],
#    "subsections": [ <section dict>, ... ]}
# ===========================================================================


def _clean(text: str) -> str:
    """Normalise whitespace within a unit of text."""
    return re.sub(r"\s+", " ", text).strip()


# ---- artifact / table / caption detection --------------------------------
_PAGE_ARTIFACT_PATTERNS = [
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.I),
    re.compile(r"^page\s+\d+$", re.I),
    re.compile(r"^\d+\s+of\s+\d+$", re.I),
    re.compile(r"^p\s*a\s*g\s*e\s*\|?\s*\d+$", re.I),
    re.compile(r"^\d+\s*\|\s*page$", re.I),
    re.compile(r"^-\s*\d+\s*-$"),
]


def _is_page_artifact(text: str) -> bool:
    """True for standalone page numbers / footers like 'Page 1 of 36' or '42'."""
    t = text.strip().strip("*_ ").strip()
    if not t:
        return True
    if any(p.match(t) for p in _PAGE_ARTIFACT_PATTERNS):
        return True
    if re.fullmatch(r"\d{1,4}", t):     # bare short number -> almost always a page no.
        return True
    return False


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    if "-" not in s or "|" not in s:
        return False
    return all(ch in "|:- " for ch in s)


_TABLE_CAP = re.compile(r"(?:\*\*)?\s*(?:table|tbl)\s*\.?\s*\d+", re.I)
_FIG_CAP   = re.compile(r"(?:\*\*)?\s*(?:figure|fig)\s*\.?\s*\d+", re.I)


def _trailing_caption(text: str, kind: str = "table"):
    """If `text` ends with a Table caption, return (caption, remainder_text)."""
    pat = _TABLE_CAP if kind == "table" else _FIG_CAP
    if pat.match(text.strip()):
        return text.strip(), ""
    matches = list(pat.finditer(text))
    if matches:
        start = matches[-1].start()
        return text[start:].strip(), text[:start].strip()
    return None


def _leading_caption(text: str, kind: str = "image"):
    """If `text` starts with a Figure caption, return (caption_line, remainder)."""
    pat = _FIG_CAP if kind == "image" else _TABLE_CAP
    stripped = text.strip()
    if not pat.match(stripped):
        return None
    parts = stripped.split("\n", 1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")


# ---- markdown -> typed blocks --------------------------------------------
def _parse_markdown_blocks(md_text: str, page: Optional[int]) -> List[Dict]:
    """Turn one page of Markdown into ordered typed blocks (heading/text/table/image)."""
    lines = md_text.split("\n")
    blocks: List[Dict] = []
    buf: List[str] = []

    def flush_text():
        if buf:
            t = _clean(" ".join(buf))
            buf.clear()
            if t and not _is_page_artifact(t):
                blocks.append({"type": "text", "text": t, "page": page})

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            flush_text(); i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush_text()
            title = _clean(m.group(2))
            if title and not _is_page_artifact(title):
                blocks.append({"type": "heading", "level": len(m.group(1)),
                               "text": title, "page": page})
            i += 1; continue

        im = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if im:
            flush_text()
            blocks.append({"type": "image", "alt": _clean(im.group(1)),
                           "path": im.group(2).strip(), "page": page})
            i += 1; continue

        if s.startswith("|") and i + 1 < n and _is_table_sep(lines[i + 1]):
            flush_text()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i].rstrip())
                i += 1
            blocks.append({"type": "table", "text": "\n".join(tbl), "page": page})
            continue

        buf.append(s)
        i += 1

    flush_text()
    return blocks


def _reattach_captions(blocks: List[Dict]) -> List[Dict]:
    """Move a 'Table N:' caption off the preceding paragraph and onto its table;
    move a 'Figure N:' caption off the following paragraph and onto its image."""
    drop = set()
    for idx, blk in enumerate(blocks):
        if blk["type"] == "table":
            j = idx - 1
            while j >= 0 and j in drop:
                j -= 1
            if j >= 0 and blocks[j]["type"] == "text":
                res = _trailing_caption(blocks[j]["text"], "table")
                if res:
                    caption, remainder = res
                    blk["caption"] = caption
                    blk["text"] = caption + "\n\n" + blk["text"]
                    if remainder:
                        blocks[j]["text"] = remainder
                    else:
                        drop.add(j)
        elif blk["type"] == "image":
            k = idx + 1
            while k < len(blocks) and k in drop:
                k += 1
            if k < len(blocks) and blocks[k]["type"] == "text":
                res = _leading_caption(blocks[k]["text"], "image")
                if res:
                    caption, remainder = res
                    blk["caption"] = caption
                    if remainder:
                        blocks[k]["text"] = remainder
                    else:
                        drop.add(k)
    return [b for i, b in enumerate(blocks) if i not in drop]


def _filter_repeated(blocks: List[Dict], num_pages: int) -> List[Dict]:
    """Drop short text blocks that recur across many pages (running headers/footers)."""
    if num_pages < 4:
        return blocks
    seen: Dict[str, set] = {}
    for b in blocks:
        if b["type"] == "text" and len(b["text"].split()) <= 10:
            seen.setdefault(b["text"], set()).add(b.get("page"))
    threshold = max(2, (num_pages + 1) // 2)
    repeated = {t for t, pages in seen.items() if len(pages) >= threshold}
    return [b for b in blocks
            if not (b["type"] == "text" and b["text"] in repeated)]


# ---- typed blocks -> nested section tree ---------------------------------
def _build_section_tree(blocks: List[Dict]) -> List[Dict]:
    root: List[Dict] = []
    stack = []  # (level, section)
    frontmatter = {"title": "(front matter)", "level": 0,
                   "paragraphs": [], "subsections": [], "page": None}

    def add_para(p):
        (stack[-1][1]["paragraphs"] if stack else frontmatter["paragraphs"]).append(p)

    for blk in blocks:
        if blk["type"] == "heading":
            sec = {"title": blk["text"], "level": blk["level"],
                   "paragraphs": [], "subsections": [], "page": blk.get("page")}
            while stack and stack[-1][0] >= blk["level"]:
                stack.pop()
            (stack[-1][1]["subsections"] if stack else root).append(sec)
            stack.append((blk["level"], sec))
        else:
            add_para(blk)

    sections = []
    if frontmatter["paragraphs"]:
        sections.append(frontmatter)
    sections.extend(root)
    return sections


# ---- DOCX helpers ---------------------------------------------------------
def _docx_table_to_md(tbl) -> str:
    rows = []
    for r in tbl.rows:
        rows.append([_clean(c.text) for c in r.cells])
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    def fmt(r):
        return "| " + " | ".join(c.replace("|", r"\|") for c in r) + " |"
    out = [fmt(rows[0]), "| " + " | ".join(["---"] * ncol) + " |"]
    out += [fmt(r) for r in rows[1:]]
    return "\n".join(out)


def _docx_heading_level(style_name: str) -> int:
    m = re.search(r"(\d+)", style_name)
    return int(m.group(1)) if m else 1


# ---- logo / decorative-image filtering -----------------------------------
def _image_dims(p: Path):
    """(width, height) in pixels of an extracted image, or (0, 0) on failure."""
    try:
        pix = fitz.Pixmap(str(p))
        return pix.width, pix.height
    except Exception:
        return 0, 0


def _filter_logo_images(blocks: List[Dict]) -> List[Dict]:
    """Drop logos / decorative images so they are NOT treated as figures (and so
    text split around them can be stitched back together). An image is treated as
    a logo if it is tiny (area < LOGO_MIN_PX**2) or byte-identical to another
    occurrence (a recurring header/footer logo)."""
    imgs = [b for b in blocks if b["type"] == "image"]
    if not imgs:
        return blocks
    by_hash: Dict[str, list] = {}
    for b in imgs:
        p = _resolve_image(b.get("path", ""))
        if p is None:
            b["_logo"] = True
            continue
        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            h = None
        w, ht = _image_dims(p)
        if w and ht and w * ht < LOGO_MIN_PX * LOGO_MIN_PX:
            b["_logo"] = True
        if h:
            by_hash.setdefault(h, []).append(b)
    n_drop = 0
    for grp in by_hash.values():
        if len(grp) >= 2:                       # same image appears 2+ times -> decorative
            for b in grp:
                b["_logo"] = True
    kept = []
    for b in blocks:
        if b.get("_logo"):
            n_drop += 1
            continue
        b.pop("_logo", None)
        kept.append(b)
    return kept


def _looks_like_logo(desc: str) -> bool:
    return bool(re.search(r"\blogos?\b|\bletterhead\b|\bbranding\b", desc or "", re.I))


# ---- stitching: paragraphs split by page/link/logo; tables split by page --
_TERMINAL_TRAIL = "\"'\u2019\u201d)]\u00bb"


def _ends_sentence(a: str) -> bool:
    t = a.rstrip().rstrip(_TERMINAL_TRAIL)
    return bool(t) and t[-1] in ".!?"


def _is_link_only(s: str) -> bool:
    s = s.strip()
    return bool(re.fullmatch(r"\[.*?\]\(.*?\)", s)
                or re.fullmatch(r"<?https?://\S+>?", s)
                or re.fullmatch(r"www\.\S+", s))


def _join_text(a: str, b: str) -> str:
    a, b = a.rstrip(), b.lstrip()
    if a.endswith("-"):                  # hyphenated line break -> stitch the word
        return a[:-1] + b
    return (a + " " + b).strip()


def _should_merge_text(prev: Dict, blk: Dict) -> bool:
    a, b = prev["text"], blk["text"]
    if not a.strip() or not b.strip():
        return True
    if _is_link_only(b):
        return True
    if a.rstrip().endswith("-"):
        return True
    if not _ends_sentence(a):             # previous block was cut mid-sentence
        return True
    return b.lstrip()[:1].islower()       # continuation that starts lowercase


def _split_table(text: str):
    lines = text.split("\n")
    gi = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
    if gi is None:
        return (text.strip() or None), []
    caption = "\n".join(lines[:gi]).strip() or None
    grid = [l for l in lines[gi:] if l.strip().startswith("|")]
    return caption, grid


def _ncols(grid) -> int:
    return len(grid[0].strip().strip("|").split("|")) if grid else 0


def _tables_continuation(prev: Dict, blk: Dict) -> bool:
    pa, pb = prev.get("page"), blk.get("page")
    if pa is None or pb is None or pa == pb:    # only stitch tables across a page break
        return False
    _, ga = _split_table(prev["text"])
    _, gb = _split_table(blk["text"])
    return bool(ga) and bool(gb) and _ncols(ga) == _ncols(gb)


def _merge_two_tables(a_text: str, b_text: str) -> str:
    cap_a, ga = _split_table(a_text)
    _, gb = _split_table(b_text)
    b_body = gb[:]
    if len(b_body) >= 2 and _is_table_sep(b_body[1]):
        if ga and b_body[0].strip() == ga[0].strip():
            b_body = b_body[2:]                 # header repeated on the next page
        else:
            b_body = [b_body[0]] + b_body[2:]    # no real header on page 2; keep row, drop sep
    merged = ga + b_body
    prefix = (cap_a + "\n\n") if cap_a else ""
    return prefix + "\n".join(merged)


def _merge_blocks(blocks: List[Dict]) -> List[Dict]:
    """Stitch paragraphs split by page breaks, links, or removed logos; stitch
    tables continued across page breaks. Headings stay hard boundaries."""
    out: List[Dict] = []
    for blk in blocks:
        if out:
            prev = out[-1]
            if prev["type"] == "text" and blk["type"] == "text" and _should_merge_text(prev, blk):
                prev["text"] = _join_text(prev["text"], blk["text"])
                continue
            if prev["type"] == "table" and blk["type"] == "table" and _tables_continuation(prev, blk):
                prev["text"] = _merge_two_tables(prev["text"], blk["text"])
                if not prev.get("caption") and blk.get("caption"):
                    prev["caption"] = blk["caption"]
                continue
        out.append(dict(blk))
    return out


# ---- PDF fallback (no pymupdf4llm): page-based text blocks ----------------
def _load_pdf_blocks_fallback(path: Path) -> List[Dict]:
    print("  (pymupdf4llm unavailable -> basic page-based PDF parse; "
          "no heading/table/image structure)")
    blocks = []
    try:
        doc = fitz.open(str(path))
        for i, page in enumerate(doc):
            for blk in page.get_text("dict").get("blocks", []):
                if blk.get("type", 0) != 0:
                    continue
                parts = []
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text"):
                            parts.append(span["text"])
                    parts.append(" ")
                t = _clean("".join(parts))
                if t and not _is_page_artifact(t):
                    blocks.append({"type": "text", "text": t, "page": i + 1})
        doc.close()
    except Exception as e:
        print(f"  PDF error {path.name}: {e}")
        return []
    return _build_section_tree(blocks)


# ---- public loaders -------------------------------------------------------
def load_pdf_structured(path: Path) -> List[Dict]:
    """PDF -> Markdown (pymupdf4llm) -> nested sections with text/table/image units."""
    if not HAVE_PYMUPDF4LLM:
        return _load_pdf_blocks_fallback(path)
    img_dir = IMG_DIR / _make_id(str(path))
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        pages = pymupdf4llm.to_markdown(
            str(path), page_chunks=True,
            write_images=DESCRIBE_IMAGES, image_path=str(img_dir),
            image_format="png", dpi=MD_DPI, margins=0, show_progress=False,
        )
    except Exception as e:
        print(f"  pymupdf4llm error {path.name}: {e}; using fallback parser")
        return _load_pdf_blocks_fallback(path)

    blocks: List[Dict] = []
    for i, pg in enumerate(pages):
        blocks.extend(_parse_markdown_blocks(pg.get("text", "") or "", i + 1))
    if not blocks:
        return []
    blocks = _filter_logo_images(blocks)          # drop logos so text can re-stitch
    blocks = _filter_repeated(blocks, len(pages))  # drop running headers/footers/page-nos
    blocks = _reattach_captions(blocks)            # Table N: / Figure N: captions
    blocks = _merge_blocks(blocks)                 # stitch page/link/logo-split units
    return _build_section_tree(blocks)


def load_docx_structured(path: Path) -> List[Dict]:
    """DOCX -> nested sections; paragraphs and tables iterated in reading order."""
    try:
        from docx.document import Document as _DocClass            # noqa: F401
        from docx.table import Table as _Table
        from docx.text.paragraph import Paragraph as _Paragraph
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
    except Exception as e:
        print(f"  DOCX in-order iteration unavailable ({e}); paragraphs only")
        CT_Tbl = CT_P = None

    blocks: List[Dict] = []
    try:
        document = _docx.Document(str(path))
        if CT_P is not None:
            for child in document.element.body.iterchildren():
                if isinstance(child, CT_P):
                    para = _Paragraph(child, document)
                    t = _clean(para.text)
                    if not t:
                        continue
                    style = ((para.style.name if para.style else "") or "").lower()
                    if style.startswith("heading") or style.startswith("title"):
                        lvl = 1 if style.startswith("title") else _docx_heading_level(style)
                        blocks.append({"type": "heading", "level": lvl, "text": t, "page": None})
                    elif not _is_page_artifact(t):
                        blocks.append({"type": "text", "text": t, "page": None})
                elif isinstance(child, CT_Tbl):
                    md = _docx_table_to_md(_Table(child, document))
                    if md:
                        blocks.append({"type": "table", "text": md, "page": None})
        else:
            for para in document.paragraphs:
                t = _clean(para.text)
                if not t or _is_page_artifact(t):
                    continue
                style = ((para.style.name if para.style else "") or "").lower()
                if style.startswith("heading") or style.startswith("title"):
                    blocks.append({"type": "heading", "level": _docx_heading_level(style),
                                   "text": t, "page": None})
                else:
                    blocks.append({"type": "text", "text": t, "page": None})
    except Exception as e:
        print(f"  DOCX error {path.name}: {e}")
        return []

    blocks = _reattach_captions(blocks)
    blocks = _merge_blocks(blocks)
    sections = _build_section_tree(blocks)
    if not sections and blocks:
        sections = [{"title": path.stem, "level": 1, "page": None,
                     "paragraphs": [b for b in blocks if b["type"] != "heading"],
                     "subsections": []}]
    return sections


def load_structured(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_structured(path)
    if ext == ".docx":
        return load_docx_structured(path)
    return []


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
print("Structure-aware loaders ready (markdown headings + tables + figures; "
      "logo filtering, page/link/logo stitching, page-split table merging)")


# In[10]:


# ===========================================================================
# Corpus builder: every file in DOCS_ROOT, nested unit -> section -> document
# -> folder -> ... -> root. Leaf summaries run in PARALLEL; intermediate
# summaries (sections/documents/folders/root) are combined bottom-up. Per-node
# caching makes the whole thing crash-resumable. A single progress bar tracks
# total LLM calls with a live ETA; each call also prints a one-line metric.
# ===========================================================================
import threading
import concurrent.futures as cf


# ---- progress tracking ----------------------------------------------------
def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class Progress:
    """Single global bar over all LLM calls + per-call one-line metrics."""

    def __init__(self, total: int, desc: str = "Summarising"):
        self.total = max(0, total)
        self.done = 0
        self.t0 = time.time()
        self.bar = tqdm(total=self.total, desc=desc, unit="call",
                        dynamic_ncols=True, smoothing=0.3)

    def step(self, label: str, dt: float, summary: Optional[str] = None):
        self.done += 1
        self.bar.update(1)
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        self.bar.set_postfix_str(
            f"ETA {_fmt_eta(remaining)} | {rate:.2f}/s | last {dt:.1f}s")
        if VERBOSE_CALLS and summary is not None:
            text = re.sub(r"\s+", " ", summary).strip() or "(empty summary)"
            for line in textwrap.wrap(text, width=100):
                self.bar.write(line)
            self.bar.write("")

    def reconcile(self):
        # combines/cached nodes may differ from the estimate; snap to reality.
        self.bar.total = self.done
        self.bar.refresh()

    def set_phase(self, label: str):
        self.bar.set_description(label)
        self.bar.refresh()

    def close(self):
        self.bar.close()
        print(f"Total LLM calls: {self.done} in {_fmt_eta(time.time() - self.t0)}")


# ---- parse caching (skip re-parsing unchanged files on re-runs) -----------
def _parse_cache_path(path: Path) -> Optional[Path]:
    try:
        st = path.stat()
        key = _make_id(str(path), f"{st.st_mtime_ns}:{st.st_size}")
        return PARSE_CACHE_DIR / f"{key}.json"
    except Exception:
        return None


def parse_document_cached(path: Path) -> List[Dict]:
    cp = _parse_cache_path(path)
    if cp is not None and cp.exists() and not FORCE_REPARSE:
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    sections = load_structured(path)
    if cp is not None:
        try:
            cp.write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return sections


def clear_parse_cache() -> int:
    n = 0
    for p in PARSE_CACHE_DIR.glob("*.json"):
        p.unlink(); n += 1
    print(f"Cleared {n} cached parse(s) from {PARSE_CACHE_DIR}")
    return n


# ---- structure walking (ID scheme shared by planner and builder) ----------
def _walk_sections(sections: List[Dict], parent_extra: Optional[str] = None):
    """Yield (id_extra, section) for every section/subsection, matching the IDs
    used by the recursive builder so the cache lines up exactly."""
    for i, sec in enumerate(sections):
        extra = f"s{i}" if parent_extra is None else f"{parent_extra}.{i}"
        yield extra, sec
        yield from _walk_sections(sec["subsections"], extra)


def _iter_docs(folder: Path):
    try:
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except Exception:
        return
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir():
            yield from _iter_docs(e)
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_file() and e.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield e


def _leaf_name(path: Path, sec_title: str, kind: str, pi: int) -> str:
    word = {"text": "para", "table": "table", "image": "figure"}.get(kind, "para")
    return f"{path.name} - {sec_title} - {word} {pi}"


def _leaf_meta(path: Path, sec_title: str, para: Dict, pi: int, kind: str) -> Dict:
    meta = {"source_file": str(path), "section": sec_title, "unit_index": pi,
            "file_type": path.suffix.lower(), "kind": kind}
    if para.get("page") is not None:
        meta["page"] = para["page"]
    if para.get("caption"):
        meta["caption"] = para["caption"]
    if kind == "image" and para.get("path"):
        meta["image_path"] = para["path"]
    return meta


# ---- planning: parse everything, pre-fill trivial leaves, list LLM jobs ----
def plan_corpus(root: Path):
    docs = list(_iter_docs(root))
    print(f"Discovered {len(docs)} document(s) under {root}")
    parsed: Dict[str, List[Dict]] = {}
    leaf_jobs: List[Dict] = []
    n_sections = 0
    n_prefilled = 0

    for path in tqdm(docs, desc="Parsing", unit="doc", dynamic_ncols=True):
        sections = parse_document_cached(path)
        parsed[str(path)] = sections
        for extra, sec in _walk_sections(sections):
            n_sections += 1
            for pi, para in enumerate(sec["paragraphs"]):
                cid = _make_id(str(path), f"{extra}p{pi}")
                if _load_cached_node(cid) is not None:
                    continue
                kind = para.get("type", "text")
                if kind == "text" and len(_words(para["text"])) <= SKIP_SUMMARY_WORDS:
                    # too short to summarise meaningfully -> store text as its own summary
                    node = TreeNode(node_id=cid, node_type="chunk",
                                    name=_leaf_name(path, sec["title"], kind, pi), path="",
                                    summary=para["text"], content=para["text"],
                                    metadata=_leaf_meta(path, sec["title"], para, pi, kind))
                    _save_cached_node(node)
                    n_prefilled += 1
                    continue
                leaf_jobs.append({"path": str(path), "sec_title": sec["title"],
                                  "extra": extra, "pi": pi, "para": para,
                                  "cid": cid, "kind": kind})

    folders = {str(Path(p).resolve().parent) for p in parsed}
    folders.add(str(root.resolve()))
    est_combines = n_sections + len(docs) + len(folders) + 1
    print(f"  sections: {n_sections} | LLM leaf calls: {len(leaf_jobs)} | "
          f"pre-filled short units: {n_prefilled} | est. combine calls: {est_combines}")
    return docs, parsed, leaf_jobs, est_combines


# ---- parallel leaf summarisation -----------------------------------------
def _summarise_leaf(job: Dict):
    path = Path(job["path"]); para = job["para"]; kind = job["kind"]
    sec_title = job["sec_title"]; cid = job["cid"]; pi = job["pi"]
    t0 = time.time()
    caption = para.get("caption", "")
    meta = _leaf_meta(path, sec_title, para, pi, kind)
    name = _leaf_name(path, sec_title, kind, pi)
    label = f"{kind}: {path.name} / {sec_title}"
    try:
        if kind == "image":
            desc = _llm_describe_image(para.get("path", ""), caption_hint=caption)
            if DROP_LOGO_BY_DESC and not caption and _looks_like_logo(desc):
                meta["skipped"] = "logo"
                node = TreeNode(node_id=cid, node_type="chunk", name=name, path="",
                                summary="", content="", metadata=meta)
                _save_cached_node(node)
                return ("skip", time.time() - t0, label + " [logo]", f"(skipped logo) {desc}")
            content = ("[FIGURE] " + (caption + " — " if caption else "") + desc).strip()
            summary = desc
        elif kind == "table":
            content = para["text"]
            summary = _llm_summarise(content, context_hint=f"{path.name}, {sec_title} (Markdown table)")
        else:
            content = para["text"]
            summary = _llm_summarise(content, context_hint=f"{path.name}, {sec_title}")
        node = TreeNode(node_id=cid, node_type="chunk", name=name, path="",
                        summary=summary, content=content, metadata=meta)
        _save_cached_node(node)
        return ("ok", time.time() - t0, label, summary)
    except Exception as e:
        return ("err", time.time() - t0, f"{label} -> {type(e).__name__}: {e}",
                f"(error: {e})")


def _warm(model: str):
    """Load a model into memory once (with keep_alive) so the first real call in a
    phase isn't paying the cold-load cost, and so we don't swap models mid-phase."""
    try:
        client.chat(model=model, messages=[{"role": "user", "content": "ok"}],
                    options={"num_predict": 1}, keep_alive=KEEP_ALIVE)
    except Exception:
        pass


def fill_leaves(leaf_jobs: List[Dict], progress: Progress, workers: Optional[int] = None):
    if not leaf_jobs:
        return
    with cf.ThreadPoolExecutor(max_workers=workers or NUM_WORKERS) as ex:
        futs = [ex.submit(_summarise_leaf, j) for j in leaf_jobs]
        for fut in cf.as_completed(futs):
            try:
                _status, dt, label, summary = fut.result()
            except Exception as e:
                _status, dt, label, summary = "err", 0.0, f"worker crashed: {e}", f"(worker crashed: {e})"
            progress.step(label, dt, summary)


# ---- bottom-up combine build (leaves are already cached) ------------------
def _build_section_node(path: Path, sec: Dict, extra: str, progress: Progress) -> Optional[TreeNode]:
    sid = _make_id(str(path), extra)
    cached = _load_cached_node(sid)
    if cached is not None and cached.node_type == "section":
        return cached

    children: List[TreeNode] = []
    for pi, _para in enumerate(sec["paragraphs"]):
        leaf = _load_cached_node(_make_id(str(path), f"{extra}p{pi}"))
        if leaf is None or leaf.metadata.get("skipped"):
            continue
        children.append(leaf)
    for ci, sub in enumerate(sec["subsections"]):
        sn = _build_section_node(path, sub, f"{extra}.{ci}", progress)
        if sn is not None:
            children.append(sn)
    if not children:
        return None

    t0 = time.time()
    summary = _llm_combine_summaries([c.summary for c in children], label=sec["title"])
    progress.step(f"section: {path.name} / {sec['title']}", time.time() - t0, summary)
    node = TreeNode(node_id=sid, node_type="section",
                    name=f"{path.name} - {sec['title']}", path="",
                    summary=summary, children=children,
                    metadata={"source_file": str(path), "section": sec["title"],
                              "level": sec.get("level"), "page": sec.get("page")})
    _save_cached_node(node)
    return node


def _build_document_node(path: Path, sections: List[Dict], progress: Progress) -> Optional[TreeNode]:
    did = _make_id(str(path))
    cached = _load_cached_node(did)
    if cached is not None and cached.node_type == "document":
        return cached
    sec_nodes = []
    for i, sec in enumerate(sections):
        sn = _build_section_node(path, sec, f"s{i}", progress)
        if sn is not None:
            sec_nodes.append(sn)
    if not sec_nodes:
        return None
    t0 = time.time()
    summary = _llm_combine_summaries([n.summary for n in sec_nodes], label=path.name)
    progress.step(f"document: {path.name}", time.time() - t0, summary)
    node = TreeNode(node_id=did, node_type="document", name=path.name, path=str(path),
                    summary=summary, children=sec_nodes,
                    metadata={"file_type": path.suffix.lower(), "num_sections": len(sec_nodes)})
    _save_cached_node(node)
    return node


def _build_folder_node(folder: Path, parsed: Dict[str, List[Dict]], progress: Progress,
                       is_root: bool = False) -> Optional[TreeNode]:
    fid = _make_id(str(folder))
    cached = _load_cached_node(fid)
    if cached is not None and cached.node_type in ("folder", "root"):
        return cached
    try:
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except Exception:
        return None

    children: List[TreeNode] = []
    for e in entries:                       # subfolders first (depth-first)
        if e.name.startswith(".") or not e.is_dir():
            continue
        n = _build_folder_node(e, parsed, progress)
        if n is not None:
            children.append(n)
    for e in entries:                       # then documents
        if e.name.startswith(".") or not e.is_file():
            continue
        if e.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        sections = parsed.get(str(e)) or parse_document_cached(e)
        dn = _build_document_node(e, sections, progress)
        if dn is not None:
            children.append(dn)
    if not children:
        return None

    label = folder.name or str(folder)
    t0 = time.time()
    summary = _llm_combine_summaries([c.summary for c in children], label=label)
    progress.step(f"{'root' if is_root else 'folder'}: {label}", time.time() - t0, summary)
    node = TreeNode(node_id=fid, node_type=("root" if is_root else "folder"),
                    name=label, path=str(folder), summary=summary, children=children,
                    metadata={"num_children": len(children)})
    _save_cached_node(node)
    return node


def build_corpus(root: Path) -> Optional[TreeNode]:
    if not root.exists():
        print(f"DOCS_ROOT does not exist: {root.resolve()}")
        return None
    t_start = time.time()
    docs, parsed, leaf_jobs, est_combines = plan_corpus(root)
    if not docs:
        print(f"No supported (.pdf/.docx) documents under {root.resolve()}")
        return None

    total = len(leaf_jobs) + est_combines
    progress = Progress(total, desc="Summarising corpus")

    # Split leaves by model so each model loads ONCE instead of swapping per call.
    image_jobs = [j for j in leaf_jobs if j["kind"] == "image"]
    text_jobs = [j for j in leaf_jobs if j["kind"] != "image"]

    if image_jobs:                                   # Phase 1: figures (vision model)
        progress.set_phase(f"1/3 figures [{VISION_MODEL}]")
        _warm(VISION_MODEL)
        fill_leaves(image_jobs, progress, workers=VISION_WORKERS)

    if text_jobs:                                    # Phase 2: text + tables (gpt-oss)
        progress.set_phase(f"2/3 text [{SUMMARY_MODEL}]")
        _warm(SUMMARY_MODEL)
        fill_leaves(text_jobs, progress, workers=NUM_WORKERS)

    progress.set_phase(f"3/3 combining [{SUMMARY_MODEL}]")   # Phase 3: bottom-up combines
    _warm(SUMMARY_MODEL)
    root_node = _build_folder_node(root, parsed, progress, is_root=True)
    progress.reconcile()
    progress.close()
    print(f"Corpus built in {_fmt_eta(time.time() - t_start)}")
    return root_node


# ---- printing (structure + summaries; raw text optional) ------------------
def print_tree_summaries(node: TreeNode, depth: int = 0, max_depth: int = 99,
                         show_leaves: bool = False) -> None:
    if depth > max_depth:
        return
    kind = node.metadata.get("kind")
    base = {"root": "ROOT", "folder": "FOLDER", "document": "DOCUMENT",
            "section": "SECTION", "chunk": "UNIT"}.get(node.node_type, node.node_type.upper())
    if node.node_type == "chunk":
        base = {"text": "PARAGRAPH", "table": "TABLE", "image": "FIGURE"}.get(kind, "PARAGRAPH")
        if not show_leaves:
            return
    pad = "  " * depth
    loc = f" (page {node.metadata['page']})" if node.metadata.get("page") is not None else ""
    n_leaves = node.count_leaves()
    extra = f"  [{n_leaves} units]" if node.node_type in ("root", "folder", "document") else ""
    print(f"{pad}{base}: {node.name}{loc}{extra}")
    summ = re.sub(r"\s+", " ", node.summary or "").strip() or "(no summary)"
    for line in textwrap.wrap(summ, width=88):
        print(f"{pad}    {line}")
    print("")
    for child in node.children:
        print_tree_summaries(child, depth + 1, max_depth, show_leaves)


print("Corpus builder ready (parallel leaves + bottom-up folder nesting + progress/ETA)")


# In[11]:


def print_tree_full(node: TreeNode, depth: int = 0) -> None:
    """Drill into ONE node (e.g. a single document) showing summaries AND raw
    text/tables/descriptions — useful for spot-checking a file. Not called on the
    whole corpus by default (that would be huge); pass a document node to it."""
    kind = node.metadata.get("kind")
    base = {"root": "ROOT", "folder": "FOLDER", "document": "DOCUMENT",
            "section": "SECTION", "chunk": "PARAGRAPH"}.get(node.node_type, node.node_type.upper())
    if node.node_type == "chunk":
        base = {"text": "PARAGRAPH", "table": "TABLE", "image": "FIGURE"}.get(kind, "PARAGRAPH")
    pad = "  " * depth
    loc = f" (page {node.metadata['page']})" if node.metadata.get("page") is not None else ""
    print(f"{pad}{base}: {node.name}{loc}")
    if node.metadata.get("caption"):
        print(f"{pad}  CAPTION: {node.metadata['caption']}")
    summ = re.sub(r"\s+", " ", node.summary or "").strip() or "(no summary)"
    print(f"{pad}  {'DESCRIPTION' if kind == 'image' else 'SUMMARY'}:")
    for line in textwrap.wrap(summ, width=92):
        print(f"{pad}    {line}")
    if node.content:
        if kind == "table":
            print(f"{pad}  ORIGINAL TABLE:")
            for line in node.content.split("\n"):
                print(f"{pad}    {line}")
        else:
            orig = re.sub(r"\s+", " ", node.content).strip()
            print(f"{pad}  ORIGINAL {'CONTENT' if kind == 'image' else 'TEXT'}:")
            for line in textwrap.wrap(orig, width=92):
                print(f"{pad}    {line}")
    print("")
    for child in node.children:
        print_tree_full(child, depth + 1)


def find_node(node: TreeNode, name_substr: str) -> Optional[TreeNode]:
    """Find the first node whose name contains name_substr (e.g. a filename)."""
    if name_substr.lower() in node.name.lower():
        return node
    for c in node.children:
        hit = find_node(c, name_substr)
        if hit:
            return hit
    return None


print("Drill-down printer ready (use print_tree_full(find_node(root, 'somefile.pdf')))")


# In[12]:


# Build the whole-corpus summary tree, with progress + ETA, then print it.
if FORCE_REBUILD:
    clear_node_cache()
if FORCE_REPARSE:
    clear_parse_cache()
cache_status()
print("=" * 72)

root_node = build_corpus(DOCS_ROOT)

if root_node:
    print("")
    print("#" * 72)
    print(f"CORPUS ROOT - {root_node.name}")
    print(f"  {root_node.count_nodes()} nodes | {root_node.count_leaves()} leaf units")
    print("#" * 72)
    print("")
    print_tree_summaries(root_node, max_depth=CORPUS_PRINT_DEPTH,
                         show_leaves=SHOW_LEAVES_IN_TREE)

    with open(TREE_FILE, "w", encoding="utf-8") as f:
        json.dump(root_node.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"(Saved corpus tree JSON to {TREE_FILE})")

    print("\n" + "=" * 72)
    print("ROOT SUMMARY")
    print("=" * 72)
    print(re.sub(r"\s+", " ", root_node.summary).strip())
