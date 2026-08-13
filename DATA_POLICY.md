# Data policy

This repository contains **no organizational corpus, generated corpus tree,
questions, answers, traces, or private benchmark artifacts**. These are ignored
by Git and must not be added to issues, pull requests, releases, or CI logs.

The public reproduction uses only the ODC-BY MultiHop-RAG dataset downloaded by
the experiment script. Users are responsible for applying appropriate access
controls to their own document libraries and model endpoints.

TreeRAG can run entirely on-premises with an open-weights Ollama model. During
search, selected node summaries and evidence passages are sent to the configured
Ollama endpoint. Configure that endpoint inside the same trusted environment as
the indexed corpus when documents are confidential.

