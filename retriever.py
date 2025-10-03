"""Minimal retriever service for the simple RAG pipeline.

Loads a FAISS index and chunk metadata, embeds a query with
`sentence-transformers/all-MiniLM-L6-v2`, and returns top‑k results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, cast

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


def setup_logging(debug: bool = False) -> None:
    #configure root logging only once for retriever use
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_device() -> str:
    #discover the most capable device while defaulting to cpu
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


#represent a single retrieved chunk hit
@dataclass
class Hit:
    pdf: str
    page: int
    chunk_id: int
    text: str
    score: float


class Retriever:
    def __init__(
        self,
        index_path: Path = Path("data/index/faiss.index"),
        metadata_path: Path = Path("data/index/metadata.json"),
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embed_model_name = embed_model
        #prefer explicit device, then env var rag_device, otherwise default to cpu for stability
        env_device = os.getenv("RAG_DEVICE")
        if device is not None:
            self.device = device
        elif env_device in {"cpu", "mps", "cuda"}:
            self.device = cast(str, env_device)
        else:
            #defaulting to cpu avoids known segfaults with some macOS mps stacks
            self.device = "cpu"

        if not self.index_path.exists() or not self.metadata_path.exists():
            #fail fast when artifacts are missing so api can advise ingest
            raise FileNotFoundError(
                f"Missing index or metadata. Expected {self.index_path} and {self.metadata_path}. "
                "Run `python ingest.py` first."
            )

        logging.info("Loading FAISS index: %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        logging.info("Total vectors: %d", self.index.ntotal)

        logging.info("Loading metadata: %s", self.metadata_path)
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logging.info("Loading embedder %s on %s", self.embed_model_name, self.device)
        try:
            self.embedder = SentenceTransformer(
                self.embed_model_name, device=self.device
            )
        except Exception as e:
            #fallback to cpu if the chosen device fails (e.g., mps segfault issues)
            logging.warning(
                "Embedder load failed on %s (%s). Falling back to CPU.", self.device, str(e)
            )
            self.device = "cpu"
            self.embedder = SentenceTransformer(
                self.embed_model_name, device=self.device
            )

    def _embed(self, text: str) -> np.ndarray:
        #convert question to a normalized vector for cosine similarity
        vec = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")

    def search(self, question: str, top_k: int = 5, score_threshold: float = 0.15) -> List[Hit]:
        #perform faiss search and return structured hits filtered by threshold
        q = question.strip()
        if not q:
            return []
        q_vec = self._embed(q)
        scores, idxs = self.index.search(q_vec, top_k)
        hits: List[Hit] = []
        for score, idx in zip(scores[0], idxs[0]):
            #skip invalid slots or those below similarity threshold
            if idx < 0 or idx >= len(self.metadata):
                continue
            if float(score) < score_threshold:
                continue
            meta = self.metadata[idx]
            hits.append(
                Hit(
                    pdf=meta.get("pdf", "unknown.pdf"),
                    page=int(meta.get("page", 0)),
                    chunk_id=int(meta.get("chunk_id", 0)),
                    text=str(meta.get("text", "")),
                    score=float(score),
                )
            )
        return hits
