# Committed public demonstration tree

`corpus_tree.json` was built only from the public ODC-BY MultiHop-RAG corpus. It contains
20,495 nodes: 1 root, 64 folders, 609 documents, 609 sections, and 19,212 chunks. The
maximum depth is five edges and maximum fan-out is 236. Serialized size is 28,659,729
bytes.

This artifact is included so reviewers can exercise retrieval without spending roughly
13.7 hours and 19,976 model calls rebuilding the index. Rebuilds write elsewhere and do
not overwrite this file.
