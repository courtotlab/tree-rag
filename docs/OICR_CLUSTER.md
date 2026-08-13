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
git clone https://github.com/courtotlab/tree-rag.git
cd tree-rag
git switch codex/publication-ready
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
