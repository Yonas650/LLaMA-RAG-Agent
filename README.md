# Production RAG Pipeline

A production-ready, fully local retrieval-augmented generation (RAG) toolkit. The
pipeline ingests your PDF knowledge base into FAISS, serves a FastAPI retrieval
service, and delivers fluent answers with Flan-T5-base through both JSON APIs and
a minimal web UI.

## Key Capabilities

- Local ingestion of PDFs into a FAISS inner-product index
- Query-time retrieval with adjustable `top_k` and `score_threshold`
- Answer generation via `google/flan-t5-base` with inline citations (PDF, page)
- Browser UI at `/` for rapid testing alongside JSON endpoints
- Pluggable device selection (CPU default, optional MPS/CUDA)

## Architecture Overview

```text
PDFs ──► ingest.py ──► FAISS index + metadata.json
                        │
                        ▼
                  FastAPI app (app.py)
                ├─ POST /search  → top-k chunks
                ├─ POST /query   → generated answer + references
                └─ GET  /        → minimal UI
```

## Prerequisites

- Python 3.10 (tested on macOS; Linux/Windows should work with equivalent deps)
- Conda (recommended) or virtualenv
- Hugging Face account (optional; no API key required for public models)

## Quick Start

```bash
# create environment
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt

# add pdfs to ingest
mkdir -p data/pdfs
cp /path/to/*.pdf data/pdfs/

# build the index (run whenever pdf set changes)
HF_HUB_ENABLE_HF_TRANSFER=1 python ingest.py --device cpu --batch-size 16

# launch api + web ui (http://127.0.0.1:8000/)
RAG_DEVICE=cpu TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
  uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns index status and model/device metadata. |
| `/search` | POST | Retrieves top-k chunks. Body: `{ "question": ..., "top_k": 5, "score_threshold": 0.15 }` |
| `/query`  | POST | Retrieves + generates answer. Same body as `/search`; response includes `answer`, `references`, `used_tokens`, `new_tokens`. |

Example query:

```bash
curl -s -H 'Content-Type: application/json' \
     -X POST http://127.0.0.1:8000/query \
     -d '{"question":"what problem does Sarathi-Serve solve?","top_k":3,"score_threshold":0.15}'
```

## Web UI

- Navigate to `http://127.0.0.1:8000/`
- Enter a question and press **Ask** (or ⌘/Ctrl + Enter)
- Answer panel shows the generated summary; References list the supporting PDFs

## Configuration

| Setting | Where | Default | Notes |
|---------|-------|---------|-------|
| `--pdf-dir` | ingest.py flag | `data/pdfs` | Source PDFs |
| `--out-dir` | ingest.py flag | `data/index` | Index & metadata output |
| `--chunk-size` | ingest.py flag | 800 chars | Adjust chunk granularity |
| `--overlap` | ingest.py flag | 150 chars | Overlap between chunks |
| `--batch-size` | ingest.py flag | 64 | Embedding batch size |
| `--device` | ingest.py flag | auto | Use `cpu`, `mps`, or `cuda` |
| `RAG_DEVICE` | env var | `cpu` | Device for retrieval + generation |
| `TOKENIZERS_PARALLELISM` | env var | `false` | Prevent tokenizer warnings |
| `OMP_NUM_THREADS` | env var | `1` | Deterministic generation |

## Project Layout

```text
app.py            # fastapi service + web ui
ingest.py         # pdf ingestion → faiss index / metadata
retriever.py      # faiss wrapper + sentence-transformers embedder
generator.py      # flan-t5-base generation helper
tests/
  test_chunking.py  # chunking unit test
requirements.txt  # pinned dependencies
```

## Testing

```bash
PYTHONPATH=. pytest tests/test_chunking.py
```

## Troubleshooting

- **Segfault on macOS (MPS):** set `RAG_DEVICE=cpu` and rerun.
- **“Index missing” errors:** rebuild with `python ingest.py` and restart uvicorn.
- **Short/terse answers:** increase `top_k`, lower `score_threshold`, or adjust prompt in `generator.py`.
- **Downloads hang:** ensure `HF_HUB_ENABLE_HF_TRANSFER=1` and only one ingest process runs.

## License

MIT License — see [LICENSE](LICENSE).
