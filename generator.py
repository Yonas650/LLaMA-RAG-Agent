"""Simple local generator using Flan-T5."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def detect_device(prefer: str | None = None) -> str:
    #select best available device while defaulting to cpu
    if prefer in {"cpu", "cuda", "mps"}:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class GenerationResult:
    answer: str
    used_tokens: int
    new_tokens: int


class Generator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str | None = None,
        max_new_tokens: int = 196,
    ) -> None:
        self.model_name = model_name
        self.device = detect_device(device)
        self.max_new_tokens = max_new_tokens

        logging.info("Loading generator %s on %s", self.model_name, self.device)
        #load tokenizer and model weights once during generator init
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        #estimate token count for budgeting context
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def build_prompt(self, question: str, context_segments: List[str]) -> str:
        #assemble prompt instructing the model to paraphrase with citations
        context_block = "\n\n---\n\n".join(context_segments)
        return (
            "You are a technical writer. Using only the context below, craft a fluent,"
            " well-structured answer in 2-4 sentences."
            " Summarise in your own words and include inline citations such as"
            " (PDF, pX) where X is the page number."
            " If the context does not cover the question, say so explicitly."
            f"\n\nContext:\n{context_block}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    def generate(self, question: str, context_segments: List[str]) -> GenerationResult:
        #tokenize prompt, run seq2seq generate, and capture stats
        prompt = self.build_prompt(question, context_segments)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        sequence = outputs[0]
        answer = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        used_tokens = int(inputs.input_ids.shape[1])
        new_tokens = max(0, int(sequence.shape[0]) - used_tokens)
        return GenerationResult(answer=answer, used_tokens=used_tokens, new_tokens=new_tokens)
