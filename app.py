"""Production RAG Pipeline FastAPI service.

Endpoints:
  - GET /health: returns simple readiness info
  - POST /search: {question, top_k?} -> [{pdf,page,chunk_id,text,score}]

Run:
  uvicorn app:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

#import local components lazily in startup
from retriever import Retriever, setup_logging
from generator import Generator

#default knobs for retrieval and answer generation
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.15
DEFAULT_MAX_CONTEXT_TOKENS = 400

RETRIEVER: Optional[Retriever] = None
GENERATOR: Optional[Generator] = None


#define request model for retrieval
class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


#structure of a retrieved chunk
class SearchHit(BaseModel):
    pdf: str
    page: int
    chunk_id: int
    text: str
    score: float


#extend search request for query endpoint reuse
class QueryRequest(SearchRequest):
    pass


#payload returned by the query endpoint
class QueryResponse(BaseModel):
    answer: str
    references: List[SearchHit]
    used_tokens: int
    new_tokens: int


#health check payload
class HealthResponse(BaseModel):
    ok: bool
    vectors: int
    index_path: str
    metadata_path: str
    embed_model: str
    device: str


app = FastAPI(title="Production RAG Pipeline")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    #serve a minimal html interface for interactive queries
    return """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Production RAG Pipeline</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 720px; }
    textarea { width: 100%; height: 7rem; padding: 0.75rem; font-size: 1rem; box-sizing: border-box; }
    button { margin-top: 0.75rem; padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #status { margin-left: 1rem; color: #555; font-style: italic; }
    .panel { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
    .references ul { padding-left: 1.2rem; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Production RAG Pipeline</h1>
  <p>Ask a question about your ingested PDFs. Answers use retrieved context and Flan-T5-base.</p>
  <textarea id=\"question\" placeholder=\"What is attention?\"></textarea><br>
  <button id=\"submit\">Ask</button><span id=\"status\"></span>
  <div class=\"panel\">
    <h2>Answer</h2>
    <pre id=\"answer\"></pre>
  </div>
  <div class=\"panel references\">
    <h2>References</h2>
    <ul id=\"refs\"></ul>
  </div>
  <script>
    const questionEl = document.getElementById('question');
    const submitBtn = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const refsEl = document.getElementById('refs');

    async function ask() {
      const question = questionEl.value.trim();
      if (!question) {
        alert('Please enter a question.');
        return;
      }
      submitBtn.disabled = true;
      statusEl.textContent = 'Working…';
      answerEl.textContent = '';
      refsEl.innerHTML = '';

      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, top_k: 5 })
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
          throw new Error(err.detail || response.statusText);
        }
        const data = await response.json();
        answerEl.textContent = data.answer || '(no answer)';
        refsEl.innerHTML = '';
        if (Array.isArray(data.references) && data.references.length) {
          data.references.forEach(ref => {
            const li = document.createElement('li');
            li.textContent = `${ref.pdf} (page ${ref.page}, chunk ${ref.chunk_id}) — score ${ref.score.toFixed(3)}`;
            refsEl.appendChild(li);
          });
        } else {
          const li = document.createElement('li');
          li.textContent = 'No references returned.';
          refsEl.appendChild(li);
        }
      } catch (err) {
        answerEl.textContent = `Error: ${err.message}`;
      } finally {
        statusEl.textContent = '';
        submitBtn.disabled = false;
      }
    }

    submitBtn.addEventListener('click', ask);
    questionEl.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) {
        ask();
      }
    });
  </script>
</body>
</html>
"""


@app.on_event("startup")
def on_startup() -> None:
    #keep startup fast and responsive; lazy init heavy models on demand
    setup_logging()
    global RETRIEVER, GENERATOR
    RETRIEVER = None
    GENERATOR = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_path = Path("data/index/faiss.index")
    metadata_path = Path("data/index/metadata.json")
    #if retriever is already initialised, report its status
    if "RETRIEVER" in globals() and isinstance(RETRIEVER, Retriever):
        return HealthResponse(
            ok=True,
            vectors=RETRIEVER.index.ntotal,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            embed_model=RETRIEVER.embed_model_name,
            device=RETRIEVER.device,
        )
    #otherwise, report presence of artifacts without loading models
    vectors = 0
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                vectors = len(json.load(f))
        except Exception:
            vectors = 0
    return HealthResponse(
        ok=index_path.exists() and metadata_path.exists(),
        vectors=vectors,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


@app.post("/search", response_model=List[SearchHit])
def search(req: SearchRequest) -> List[SearchHit]:
    global RETRIEVER
    if "RETRIEVER" not in globals() or not isinstance(RETRIEVER, Retriever):
        try:
            RETRIEVER = Retriever()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Index missing. Run ingest.py first.")
        except Exception as e:
            logging.error("Retriever init failed: %s", e)
            raise HTTPException(status_code=500, detail="Failed to init retriever. Check server logs.")
    hits = RETRIEVER.search(req.question, top_k=req.top_k)
    return [SearchHit(**h.__dict__) for h in hits]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    global RETRIEVER, GENERATOR

    #reuse search pipeline to gather supporting chunks
    hits = search(req)
    if not hits:
        return QueryResponse(
            answer="I could not find relevant context to answer right now.",
            references=[],
            used_tokens=0,
            new_tokens=0,
        )

    if "GENERATOR" not in globals() or not isinstance(GENERATOR, Generator):
        try:
            GENERATOR = Generator()
        except Exception as e:
            logging.error("Generator init failed: %s", e)
            raise HTTPException(status_code=500, detail="Failed to init generator. Check server logs.")

    max_context_tokens = 400
    context_segments: List[str] = []
    selected_hits: List[SearchHit] = []
    context_tokens = 0
    for hit in hits:
        text = hit.text
        token_len = GENERATOR.count_tokens(text)
        if context_tokens + token_len > max_context_tokens:
            break
        header = f"PDF: {hit.pdf} | Page: {hit.page} | Chunk: {hit.chunk_id}"
        context_segments.append(f"{header}\n{text}")
        context_tokens += token_len
        selected_hits.append(hit)

    if not context_segments:
        return QueryResponse(
            answer="The retrieved context was too large to fit within the limit. Please ask a narrower question.",
            references=[],
            used_tokens=0,
            new_tokens=0,
        )

    result = GENERATOR.generate(req.question, context_segments)
    return QueryResponse(
        answer=result.answer,
        references=selected_hits,
        used_tokens=result.used_tokens,
        new_tokens=result.new_tokens,
    )
