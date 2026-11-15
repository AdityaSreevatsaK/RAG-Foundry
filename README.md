# RAG Foundry

A small collection of **experimental Python RAG examples** using LangChain, FAISS, local Ollama inference and Hugging Face embeddings.

**Scope:** This repository currently contains **three example scripts**. It is a learning and prototyping collection, not a packaged, tested or production-ready RAG framework.

## Available examples

| Example | Source |
| --- | --- |
| Single-turn question answering over a text document | [FAISS + Phi-3 text QA](src/Single-Turn%20-%20QA%20%28FAISS%20%2B%20Phi3%29.py) |
| Question answering over a PDF document | [FAISS + Phi-3 PDF QA](src/Single-Turn%20-%20PDF%20Question-Answering%20System%20%28FAISS%20%2B%20Phi-3%29.py) |
| Conversational question answering | [FAISS + Phi-3 conversational QA](src/Conversational%20-%20QA%20%28FAISS%20%2B%20Phi3%29.py) |

The supplied [example data](data/) is for experiments. Review the relevant script and data path before execution.

## Running an example

The scripts use legacy LangChain imports and do **not** have pinned dependencies or a verified one-command installation. They need a compatible Python/LangChain environment, an Ollama installation with a local `phi3` model, and the embedding model downloaded on first use. Some dependencies or model downloads may require an internet connection.

Bundled data paths now resolve from the repository location, not the terminal working directory. With compatible dependencies installed, run from the repository root:

```powershell
Set-Location src
python ".\Single-Turn - QA (FAISS + Phi3).py"
```

**Caution:** Successful execution with current dependency releases has not been independently verified. Offline unit tests and Python syntax checks cover document-path resolution and shared utilities. There is still no reproducible model/dependency lock or live RAG integration test.

## Not implemented here

Multi-hop retrieval, structured extraction, document/code copilots and other RAG patterns are possible future experiments, **not current features**. The examples do not demonstrate validated citation accuracy, latency improvements or security hardening.

Licensed under [MIT](LICENSE).
