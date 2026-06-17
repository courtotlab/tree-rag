# TreeRAG

## File Descriptions

This Github repository includes the most updated versions of three scripts central to this research project:
1. The algorithm constructing the summary tree encapsulating the given directory of 2,852 files (treePrototype7.ipynb). [Final node count: 130154.]
2. The algorithm implementing three important repairs to the tree addressing truncation, chunk retention, and table summary issues.
3. The algorithm implementing the query response logic, including agentic traversal of the previously constructed tree.

## How to Run

1. Start an Ollama SSH tunnel at port 11528.
2. Run the following commands: 

export OLLAMA_HOST=localhost:11528
ollama pull gpt-oss:120b      
ollama pull gemma3:27b         
ollama pull nomic-embed-text  

pip install -r requirements.txt
jupyter lab [script name here].ipynb
