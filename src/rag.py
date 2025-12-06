# src/rag.py
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
import groq  # for exception types

from .config import INDEX_DIR, TOP_K, MAX_CONTEXT_CHARS, PROJECT_ROOT

load_dotenv()  # load .env at project root


def _load_prompt() -> str:
    p = PROJECT_ROOT / "prompts" / "system_prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "You are a helpful assistant for question-answering over a private knowledge base. "
        "Use ONLY the provided context to answer. If the answer is not in the context, say you don’t know. "
        "Cite sources as [source: filename#chunk_id]. Be concise and factual."
    )


def _normalize_model_id(model_id: str) -> str:
    """Map deprecated model ids to current Groq ids."""
    mapping = {
        "llama3-8b-8192": "llama-3.1-8b-instant",
        "llama3-70b-8192": "llama-3.3-70b-versatile",
    }
    return mapping.get(model_id, model_id)


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class RetrievedDoc:
    text: str
    source: str
    chunk_id: int
    score: float


class RAGPipeline:
    def __init__(self, index_dir: Path = INDEX_DIR):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Put it in .env or the environment.")
        self.client = Groq(api_key=api_key)

        self.index_dir = index_dir
        self.index = None
        self.docs = None
        self.meta = {}
        self.system_prompt = _load_prompt()

        self._load_index()

        # Embedding model (same as used during ingestion)
        self.embed_model_name = self.meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(self.embed_model_name)

        # Default LLM model (can be overridden via .env GROQ_MODEL)
        configured = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.default_model = _normalize_model_id(configured)

        # Optional re-ranking
        self.use_rerank = _bool_env("RERANK", False)
        self.rerank_k = int(os.getenv("RERANK_K", "30"))
        self.rerank_model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker = None  # lazy load

        # Cache available models once; helps with nice fallbacks
        try:
            self.available_models = {m.id for m in self.client.models.list().data}
        except Exception:
            self.available_models = set()

        # If configured default isn't available, pick a sane fallback
        if self.available_models and self.default_model not in self.available_models:
            if "llama-3.1-8b-instant" in self.available_models:
                self.default_model = "llama-3.1-8b-instant"
            else:
                self.default_model = sorted(self.available_models)[0]

    def _load_index(self):
        index_path = self.index_dir / "index.faiss"
        docs_path = self.index_dir / "docs.pkl"
        meta_path = self.index_dir / "meta.json"
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"Missing index. Run ingestion first: python -m src.ingest. Expected files in {self.index_dir}"
            )
        self.index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def _ensure_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.rerank_model_name)
            except Exception as e:
                # If loading fails, disable rerank gracefully
                self.use_rerank = False

    def _search(self, query: str, top_k: int, candidate_k: int = None) -> List[RetrievedDoc]:
        """Vector search; returns candidate docs (no reranking here)."""
        k = candidate_k or top_k
        # Clamp to index size
        try:
            ntotal = int(getattr(self.index, "ntotal", 0))
            if ntotal > 0:
                k = min(k, ntotal)
        except Exception:
            pass

        q_vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
        D, I = self.index.search(np.array([q_vec]), k)
        texts = self.docs["texts"]
        metas = self.docs["metas"]
        results: List[RetrievedDoc] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            md = metas[idx]
            results.append(
                RetrievedDoc(
                    text=texts[idx],
                    source=md.get("source", "unknown"),
                    chunk_id=md.get("chunk_id", -1),
                    score=float(score),
                )
            )
        return results

    def _rerank_docs(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        self._ensure_reranker()
        if not self.use_rerank or self._reranker is None or not docs:
            return docs[:top_k]
        pairs = [[query, d.text] for d in docs]
        scores = self._reranker.predict(pairs)
        order = np.argsort(-np.array(scores))  # descending
        reranked = [docs[i] for i in order[:top_k]]
        return reranked

    def _build_context(self, docs: List[RetrievedDoc]) -> str:
        parts, total = [], 0
        for d in docs:
            snippet = d.text.strip()[:3000]  # guard per-chunk size
            tag = f"[source: {d.source}#{d.chunk_id}]"
            block = f"{tag}\n{snippet}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def ask(self, query: str, top_k: int = TOP_K, model: str = None, temperature: float = 0.2):
        model_id = _normalize_model_id(model or self.default_model)

        # Use more candidates if re-ranking is enabled
        candidate_k = max(top_k, self.rerank_k) if self.use_rerank else top_k
        candidates = self._search(query, top_k, candidate_k=candidate_k)
        retrieved = self._rerank_docs(query, candidates, top_k) if self.use_rerank else candidates[:top_k]

        context = self._build_context(retrieved)
        if not context.strip():
            return "I couldn’t find relevant context in the knowledge base.", retrieved

        user_prompt = (
            "Answer the question strictly using the context. If not answerable, say you don't know.\n"
            "Cite sources as [source: filename#chunk_id]. Be concise and factual.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=model_id,  # e.g., "llama-3.1-8b-instant", "llama-3.3-70b-versatile"
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(temperature),
            )
            answer = resp.choices[0].message.content
        except groq.BadRequestError:
            # Common cause: model decommissioned or invalid id
            fallback = "llama-3.1-8b-instant"
            answer = (
                "The selected model is unavailable or has been decommissioned. "
                f"Try GROQ_MODEL={fallback} (fast) or 'llama-3.3-70b-versatile' (larger), "
                "or pass --model on the CLI."
            )
        return answer, retrieved