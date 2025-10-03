#!/usr/bin/env python3
"""Minimal ingestion pipeline.

Reads PDFs → chunks text → embeds with Sentence-Transformers → builds FAISS index
and writes both the index (`data/index/faiss.index`) and chunk metadata
(`data/index/metadata.json`).

Run:
  pip install -r requirements.txt
  python ingest.py --pdf-dir data/pdfs --out-dir data/index
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def setup_logging(debug: bool = False) -> None:
    #configure root logger with optional debug verbosity
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_device() -> str:
    #probe available accelerators with safe cpu fallback
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def normalize_text(text: str) -> str:
    #normalize whitespace while keeping words intact
    text = text.replace("\u00a0", " ")  #non-breaking space → space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping char-length chunks near punctuation boundaries.

    Rules:
      - Prefer breaking at last punctuation (., !, ?) within the window if found
      - Guarantee overlap < chunk_size
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        #choose a window and backtrack to punctuation for smoother chunks
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            #find last punctuation + space
            last = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if last != -1 and last > chunk_size // 2:
                end = start + last + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        #advance with overlap
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    #read each pdf page and capture its text payload
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return pages


def ingest_pdfs(pdf_dir: Path, out_dir: Path, embed_model: str, batch_size: int, chunk_size: int, overlap: int, device: str) -> Tuple[faiss.IndexFlatIP, List[dict]]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}. Place files then retry.")

    logging.info("Found %d PDF(s). Using %s on %s (batch=%d)", len(pdf_files), embed_model, device, batch_size)

    all_texts: List[str] = []
    metadata: List[dict] = []
    global_id = 0

    for pdf in tqdm(pdf_files, desc="PDFs", unit="pdf"):
        #iteratively chunk each page of every pdf
        pages = extract_pdf_pages(pdf)
        for page_idx, page_text in enumerate(pages):
            if not page_text or not page_text.strip():
                continue
            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for c_idx, chunk in enumerate(chunks):
                all_texts.append(chunk)
                metadata.append({
                    "id": global_id,
                    "pdf": pdf.name,
                    "page": page_idx,
                    "chunk_id": c_idx,
                    "text": chunk,
                })
                global_id += 1

    if not all_texts:
        raise RuntimeError("No text chunks produced from PDFs. Check PDF contents.")

    logging.info("Total chunks: %d", len(all_texts))

    #compute sentence-transformer embeddings for every chunk
    model = SentenceTransformer(embed_model, device=device)
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    #store embeddings into an inner-product faiss index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logging.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)

    return index, metadata


def persist(index: faiss.Index, metadata: List[dict], out_dir: Path) -> None:
    #persist index and metadata so the api can reload them later
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "faiss.index"
    meta_path = out_dir / "metadata.json"
    faiss.write_index(index, str(faiss_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logging.info("Saved: %s (%s) and %s", faiss_path.name, faiss_path.parent, meta_path)


def parse_args() -> argparse.Namespace:
    #define command-line interface for ingestion workflow
    p = argparse.ArgumentParser(description="Ingest PDFs into a FAISS index.")
    p.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"), help="Directory with PDF files")
    p.add_argument("--out-dir", type=Path, default=Path("data/index"), help="Output directory for index + metadata")
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument("--chunk-size", type=int, default=800, help="Chars per chunk")
    p.add_argument("--overlap", type=int, default=150, help="Chars of overlap between chunks")
    p.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (auto-detect by default)")
    p.add_argument("--debug", action="store_true", help="Enable debug logs")
    return p.parse_args()


def main() -> None:
    #entry point orchestrating directory prep, ingestion, and persistence
    args = parse_args()
    setup_logging(args.debug)

    device = args.device or detect_device()
    pdf_dir: Path = args.pdf_dir
    out_dir: Path = args.out_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting ingestion …")
    index, metadata = ingest_pdfs(
        pdf_dir=pdf_dir,
        out_dir=out_dir,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=device,
    )
    persist(index, metadata, out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
