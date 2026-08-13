# OICR compute-host setup (private operations)

This document belongs only in the private Courtot Lab repository. The public export
intentionally omits it.

## First login and mandatory password change

Obtain the temporary password through the approved private channel. Do not paste it into
Git, a shell command, a script, chat, or `.env` file.

```bash
ssh asharma@10.30.134.39
passwd
```

`passwd` prompts for the current temporary password, then the new password twice. Input
is intentionally not echoed. Use a unique password and store it in the approved password
manager. Log out and sign in again to confirm the change.

## Configure key-based SSH after changing the password

On the laptop:

```bash
ssh-keygen -t ed25519 -C "treequest-oicr"
cat ~/.ssh/id_ed25519.pub | ssh asharma@10.30.134.39 \
  'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
```

Then connect with `ssh asharma@10.30.134.39`. Keep password authentication available
until the key has been tested in a second terminal.

## Install and run on the host

```bash
ssh asharma@10.30.134.39
git clone git@github.com:courtotlab/tree-rag.git
cd tree-rag
git switch main
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[build,test]'
curl -fsS http://127.0.0.1:11434/api/version
ollama list
```

The repository and Ollama both run on `10.30.134.39`; therefore the correct application
endpoint is cluster-local `http://127.0.0.1:11434`. No laptop tunnel or port 11528 is
required. Do not expose Ollama's port outside the compute host.

## Survive disconnects and laptop shutdown

```bash
tmux new -s treequest
cd ~/tree-rag
source .venv/bin/activate
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
./scripts/query_demo.sh "Your question"
```

Detach with `Ctrl-b`, `d`; reconnect with `tmux attach -t treequest`. For a fresh public
tree build, run `./scripts/build_multihop_demo.sh` in its own `tmux` session.

## Two end-to-end public demos from a personal computer

The personal computer is only an SSH client. Connect it to the OICR VPN, then run all
repository code, model calls, caches, and outputs on `10.30.134.39`. Do not install or
run Ollama on the personal computer, create a laptop-local port forward, or expose the
cluster Ollama port. These demos use only the public MultiHop-RAG corpus.

### Frozen model behavior

Do not change the model or image settings for these demonstrations:

- Agentic traversal and answer synthesis use cluster-local `gpt-oss:120b`.
- MultiHop-RAG tree construction uses `gpt-oss:120b` for text summaries and bottom-up
  combines.
- The general document builder uses `gemma3:27b` for figure descriptions, but the
  frozen MultiHop-RAG builder deliberately sets `DESCRIBE_IMAGES=False` because its
  609 news articles are text-only. It therefore makes no `gemma3:27b` calls. Do not
  toggle this flag merely to exercise the vision model.
- The frozen builder also checks for `nomic-embed-text`. This inherited build-time
  dependency does not turn the query demo into dense-vector evidence retrieval.

Check the cluster-local service and install only missing required models:

```bash
curl -fsS http://127.0.0.1:11434/api/version
ollama list
ollama show gpt-oss:120b >/dev/null 2>&1 || ollama pull gpt-oss:120b
ollama show nomic-embed-text >/dev/null 2>&1 || ollama pull nomic-embed-text
```

`gemma3:27b` need not be downloaded for the text-only MultiHop-RAG demos. It should
already be available before using `scripts/build_tree.py` on a PDF or DOCX collection
that actually contains figures.

### Demo 1: rebuild the public MultiHop-RAG tree

The complete build previously required about 13.7 hours and 19,976 model calls. Run it
inside `tmux`. The versioned run directory is outside the Git checkout, so it cannot
overwrite the committed public tree or an earlier build.

```bash
ssh asharma@10.30.134.39
tmux new -s treequest-build

cd ~/tree-rag
git switch main
git pull --ff-only
source .venv/bin/activate

export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
export TREEQUEST_BUILD_WORKERS=4
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$HOME/treequest-runs/multihop-build-$RUN_ID"
mkdir -p "$RUN_ROOT"
export TREEQUEST_CACHE_DIR="$RUN_ROOT/tree_cache"

./scripts/build_multihop_demo.sh 2>&1 | tee "$RUN_ROOT/build.log"
test -s "$RUN_ROOT/tree_cache/corpus_tree.json"
printf 'Tree: %s\nLog:  %s\n' \
  "$RUN_ROOT/tree_cache/corpus_tree.json" "$RUN_ROOT/build.log"
```

Detach without stopping the build with `Ctrl-b`, then `d`. Reconnect from any computer
on the VPN with:

```bash
ssh asharma@10.30.134.39
tmux attach -t treequest-build
```

The per-node and parse caches make the build resumable. If the process itself stops,
rerun the same command with the same `TREEQUEST_CACHE_DIR`; do not create a new run ID
when the intent is to resume that build.

### Demo 2: run one question through agentic traversal

This demo uses the committed public tree and cluster-local `gpt-oss:120b` for branch
selection, evidence retention, sufficiency checks, teleport recovery, and final answer
synthesis.

```bash
ssh asharma@10.30.134.39
tmux new -s treequest-query

cd ~/tree-rag
git switch main
git pull --ff-only
source .venv/bin/activate

export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
export TREEQUEST_MODE=thorough
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$HOME/treequest-runs/query-$RUN_ID"
mkdir -p "$RUN_ROOT"

./scripts/query_demo.sh \
  "Which developments are compared across multiple reports?" \
  2>&1 | tee "$RUN_ROOT/query.json"
printf 'Result: %s\n' "$RUN_ROOT/query.json"
```

To query the newly built tree instead of the committed demonstration tree, preserve
the build path printed by Demo 1 and set it explicitly:

```bash
export TREEQUEST_TREE_PATH="$HOME/treequest-runs/multihop-build-YYYYMMDD_HHMMSS/tree_cache/corpus_tree.json"
./scripts/query_demo.sh "Your question"
```

## Prompt for a coding agent on the personal computer

Give the following prompt to the coding agent on the replacement computer. It is
deliberately explicit about the compute boundary and the frozen implementation:

```text
Set up and run the two existing TreeQuest public demos on the OICR compute host. My
personal computer must be only an SSH client. Connect through the OICR VPN to
asharma@10.30.134.39, and perform all repository operations, model calls, caches, and
outputs on that host. Do not run Ollama locally, open port 11434 or 11528 to my laptop,
or copy any corpus to my laptop.

Use only the private git@github.com:courtotlab/tree-rag.git repository on branch main.
If ~/tree-rag exists, do not reclone or delete it; run git status, stop if it has
uncommitted changes, and otherwise use git switch main followed by git pull --ff-only.
If it does not exist, clone it. Never touch or push to the public TreeRAG repository.
Do not modify source code, model defaults, the committed demo tree, or prior outputs.

Create or reuse ~/tree-rag/.venv and install the package with:
python -m pip install -e '.[build,test]'
Use the cluster-local endpoint http://127.0.0.1:11434. Confirm that Ollama is reachable
and that gpt-oss:120b and nomic-embed-text exist; pull only a missing model. Do not
change the frozen MultiHop builder's DESCRIBE_IMAGES=False setting. The MultiHop-RAG
articles are text-only, so this demo correctly uses gpt-oss:120b for summaries and
combines and makes no gemma3:27b calls. gemma3:27b is only the existing vision model
for general PDF/DOCX builds that contain figures.

Run Demo 1 in a tmux session named treequest-build. Execute
./scripts/build_multihop_demo.sh with TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434,
TREEQUEST_MODEL=gpt-oss:120b, and TREEQUEST_BUILD_WORKERS=4. Put TREEQUEST_CACHE_DIR
under a new versioned directory in ~/treequest-runs, outside the Git checkout, and tee
the log there. Never overwrite or delete an existing run. Report the tmux session name,
tree path, log path, and whether the process is still running. The full build may take
about 13.7 hours; do not wait while consuming agent usage. Leave it running in tmux.

Run Demo 2 in a separate tmux session named treequest-query. From ~/tree-rag, activate
the same virtual environment and run ./scripts/query_demo.sh in thorough mode with the
question "Which developments are compared across multiple reports?". Use the committed
public MultiHop-RAG tree and cluster-local gpt-oss:120b. Tee the complete output to a
new versioned directory under ~/treequest-runs. Report the answer output path and any
error exactly. Do not inspect or use private corpus files, private questions, or private
benchmark reports.

If SSH, GitHub authentication, VPN access, or a required model needs an interactive
credential or approval, stop and ask me rather than embedding a password or token in a
command, file, URL, log, or chat response.
```
