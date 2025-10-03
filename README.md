# Production RAG Pipeline

This project ingests local PDF documents into a FAISS index, exposes a FastAPI
retrieval endpoint, and can generate short answers with Flan-T5-base. A minimal
browser UI is included for quick testing.

## Quick Start

```bash
#set up environment
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt

#add pdfs to ingest
mkdir -p data/pdfs
cp /path/to/*.pdf data/pdfs/

#build the index (run once or whenever pdfs change)
HF_HUB_ENABLE_HF_TRANSFER=1 python ingest.py --device cpu --batch-size 16

#run the api (serves json endpoints and a simple web ui at /)
RAG_DEVICE=cpu TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
  uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health` – Shows index status (vectors, model name, device)
- `POST /search` – Retrieves chunks
  ```json
  {"question": "what is attention?", "top_k": 3}
  ```
- `GET /` – Minimal web page for entering a question
- `POST /query` – Retrieves + generates answer (supports `top_k` and `score_threshold`)
  ```bash
  curl -s -H 'Content-Type: application/json' \
       -X POST http://127.0.0.1:8000/query \
       -d '{"question":"what is attention?","top_k":3,"score_threshold":0.15}'
  ```
  Response fields: `answer`, `references`, `used_tokens`, `new_tokens`.

## Configuration Tips

- `ingest.py` flags: `--pdf-dir`, `--out-dir`, `--chunk-size`, `--overlap`, `--batch-size`, `--device`.
- Environment variables:
  - `RAG_DEVICE` (`cpu` default) – override retriever/generator device.
  - `TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS` – tune performance.
  - `HF_HUB_ENABLE_HF_TRANSFER=1` – faster/resumable Hugging Face downloads.

## Testing

A small unit test checks the chunker:
```bash
PYTHONPATH=. pytest tests/test_chunking.py
```

## Troubleshooting

- Segfaults on macOS MPS: set `RAG_DEVICE=cpu` and rerun.
- Missing index error: run `python ingest.py` and restart uvicorn.
- Empty answers: increase `top_k` or lower `score_threshold` inside `retriever.py`.
