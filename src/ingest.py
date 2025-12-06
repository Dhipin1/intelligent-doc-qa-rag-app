import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DATA_RAW_DIR, INDEX_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from .utils import collect_documents

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_faiss_index(
    data_dir: Path = DATA_RAW_DIR,
    index_dir: Path = INDEX_DIR,
    model_name: str = EMBED_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = 64,
):
    _ensure_dir(index_dir)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Scanning and chunking documents from: {data_dir}")
    texts, metas = collect_documents(data_dir, chunk_size, chunk_overlap)
    if len(texts) == 0:
        raise RuntimeError("No chunks produced. Check your data directory.")

    print(f"Total chunks: {len(texts)}")
    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity using inner product
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index + docstore
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    with open(index_dir / "docs.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    meta = {
        "embedding_model": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_vectors": int(index.ntotal),
        "dim": int(dim),
    }
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved docstore to {index_dir/'docs.pkl'}")
    print(f"Saved metadata to {index_dir/'meta.json'}")

def main():
    build_faiss_index()

if __name__ == "__main__":
    main()