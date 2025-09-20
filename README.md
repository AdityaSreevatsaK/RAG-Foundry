# <p align="center">RAG Foundry</p>

## <p align="center"><i>Forging Truth from Data.</i></p>

RAG-Foundry is an experimental space for building and testing Retrieval-Augmented Generation (RAG) systems. This
collection of focused projects explores various techniques for grounding LLMs, featuring practical implementations with
tools like LangChain, LlamaIndex, and various vector databases. Ideal for prototyping, learning, and hands-on
experimentation.

---

### 1. Single-Turn QA (Grounded Answers)

Projects that answer one-off questions from a fixed corpus with explicit citations to sources. The emphasis is on clean
chunking, solid retrieval (keyword/dense/hybrid), and tight prompting to minimize hallucinations. Ideal for establishing
baselines for quality, latency, and cost before adding complexity.

✦ [Implementing a Baseline RAG for Document Question-Answering with FAISS and Microsoft's Phi-3](src/Baseline%20RAG%20-%20QA%20(FAISS%20+%20Phi3).py) <br />
✦ [Building a PDF Question-Answering System with FAISS and Phi-3](src/Baseline%20RAG%20-%20PDF%20Question-Answering%20System%20(FAISS%20+%20Phi-3).py) <br />
---

### 2. Conversational QA (Multi-Turn + Memory)

Projects that sustain a dialogue over documents, carrying context across turns while re-retrieving fresh evidence each
time. The focus is on reference resolution, follow-up disambiguation, and safe memory use rather than long prompts
alone. Great for testing how well the system adapts to clarifications and evolving user intent.

---

### 3. Long-Form Synthesis (Summarize & Compare)

Projects that produce grounded briefs, reports, or comparisons by weaving together many sources. You’ll explore
map-reduce or refine patterns, quote attribution, outline-driven drafting, and redundancy control. Success is measured
by coverage, faithfulness, and readability at lengths beyond a paragraph.

---

### 4. Structured Extraction (RAG-Extract)

Projects that convert unstructured text into JSON or tables with field-level provenance. They combine targeted retrieval
with schema-aware prompting and require citing exactly where each value came from. This is where precision/recall of
fields, normalization, and traceability really matter.

---

### 5. Multi-Hop / Compositional QA

Projects that require chaining evidence across multiple chunks or documents to reach an answer. Expect query
decomposition, iterative retrieval, and reasoning steps that verify intermediate claims. Evaluation centers on factual
correctness under composition and robustness to partial retrieval.

---

### 6. Code & Tech Docs Copilot

Projects that answer developer questions by retrieving from codebases, API specs, and technical docs. They integrate
symbol/file search, snippet grounding with file paths and line numbers, and pragmatic “how-to” guidance. The goal is to
make help feel like an informed teammate rather than a generic chatbot.

---

### 7. Tables, Data & Semi-Structured QA

Projects that answer questions over CSV/Excel, HTML/PDF tables, and metadata catalogs with precise cell-level grounding.
They often pair retrieval with parsers or light SQL to compute aggregates and surface the exact rows/columns used.
Accuracy, numeric faithfulness, and transparent evidence are the key tests here.

---

