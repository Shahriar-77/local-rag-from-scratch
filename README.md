## Lightweight Local RAG System (CPU/GPU-Constrained)

A **fully local Retrieval-Augmented Generation (RAG) system** designed to run on consumer hardware, emphasizing **clarity, modularity, and data-aware retrieval** rather than opaque frameworks or managed services.

This project demonstrates how a grounded question-answering system can be built **from first principles** using:

* structured ingestion
* format-aware chunking
* lexical + semantic hybrid retrieval
* persistent embeddings
* VRAM-constrained local LLM inference

No external APIs. No hosted vector databases. No cloud dependencies.

---

## Key Features

### 📂 Multi-Format Ingestion

Supports structured extraction from:

* PDF
* DOCX
* PPTX
* CSV
* XLSX
* TXT

Each format is parsed into **semantically meaningful blocks**, preserving document structure where possible.

---

### ✂️ Structure-Aware Chunking

Chunks are created based on document semantics:

* PDF → per page
* DOCX → heading-scoped sections
* PPTX → per slide
* CSV/XLSX → table-level chunks
* TXT → sliding window with overlap

Each chunk receives a **stable content-derived ID** to support persistence and re-indexing.

---

### 🔍 Hybrid Retrieval (Lexical + Semantic)

Retrieval combines:

* **Lexical scoring** (token frequency)
* **Semantic similarity** (Sentence-Transformers embeddings)

Scores are min-max normalized and combined via weighted fusion:

```
final_score = α · semantic + β · lexical
```

This improves robustness for:

* numeric queries
* structured data
* exact term matching
* paraphrased questions

---

### 🧠 Local Embeddings with Persistence

* Embeddings generated using `all-MiniLM-L6-v2`
* Stored locally (`.npy`) alongside chunk metadata
* Incremental corpus updates without recomputation

---

### 🧾 Grounded Generation

* Local GGUF models via `llama.cpp`
* Supports [**Phi-3 Mini**](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main) and [**Mistral-7B (Q4)**](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main)
* Explicit context assembly with token budgeting
* Strict prompt enforcing **context-only answers**

---

### 💾 Stateful Design

* Corpus and embeddings cached across sessions
* LLM instances reused to avoid reload overhead
* Previous answer optionally injected for short conversational continuity

---

### 🖥️ Interfaces

* **CLI application** for iterative querying
* **Streamlit UI** for interactive use and inspection

---

## Project Structure

```
.
├── core/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│
├── generation.py
├── state.py
├── app.py
├── streamlit_app.py
├── data_state/
│   ├── corpus.pkl
│   └── embeddings.npy
└── models/
    └── *.gguf
```

---

## Hardware Assumptions

* Tested on consumer GPUs (e.g., RTX 3060 class)
* Designed to operate within ~4–6 GB VRAM
* No requirement for FAISS, Milvus, or cloud services

---

## Motivation

This project intentionally avoids:

* agent frameworks
* orchestration layers
* cloud-managed abstractions

The goal is to **understand and control each stage of the RAG pipeline**, from raw document ingestion to final token generation.

---

## Limitations

* Not optimized for large-scale corpora
* Single-node, in-memory retrieval
* No re-ranking models beyond simple fusion
* Intended for learning, inspection, and extension — not production deployment

---

## Future Extensions (Optional)

* Cross-encoder re-ranking
* Query-adaptive chunk selection
* Streaming generation
* Evaluation harness (faithfulness / recall)

---

## License

MIT

