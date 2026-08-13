# Running through the OICR Ollama tunnel

TreeQuest code, corpus data, caches, and outputs run on the workstation. Only Ollama API
traffic crosses an authenticated SSH tunnel to the approved OICR server.

Open the tunnel in one terminal:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519_oicr" \
  -L 127.0.0.1:11528:127.0.0.1:11434 \
  asharma@10.30.134.39
```

Use it from a second local terminal:

```bash
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11528
export TREEQUEST_MODEL=gpt-oss:120b
curl -fsS "$TREEQUEST_OLLAMA_URL/api/version"
./scripts/query_demo.sh "Your question"
```

Keep the tunnel terminal open while TreeQuest runs. Do not bind port `11528` to an
external interface, expose Ollama port `11434`, or send restricted corpus content to a
third-party endpoint. The full private procedure and smoke-test instructions are in
`docs/OICR_CLUSTER.md`.
