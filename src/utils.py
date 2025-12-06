import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple
from tqdm import tqdm

def iter_all_files(data_dir: Path) -> Iterable[Path]:
    """Yield all files under data_dir recursively (skip hidden)."""
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.startswith("."):
                continue
            yield Path(root) / f

def read_text_file(p: Path) -> str:
    """Read a text-like file robustly with fallback encodings."""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return p.read_bytes().decode("utf-8", errors="ignore")

def clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_long(s: str, size: int) -> List[str]:
    """Split a long string near sentence boundaries."""
    if len(s) <= size:
        return [s]
    parts = []
    start = 0
    while start < len(s):
        end = min(start + size, len(s))
        window = s[start:end]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "), window.rfind("\n"))
        if cut == -1 or end == len(s):
            parts.append(s[start:end])
            start = end
        else:
            cut = start + cut + 1
            parts.append(s[start:cut].strip())
            start = cut
    return parts

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text by paragraphs then pack into chunks with overlap."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(p) > chunk_size:
            for piece in split_long(p, chunk_size):
                if len(cur) + len(piece) + 2 > chunk_size and cur:
                    chunks.append(cur.strip())
                    cur = cur[-overlap:] if overlap > 0 else ""
                cur += ("\n\n" if cur else "") + piece
        else:
            if len(cur) + len(p) + 2 > chunk_size and cur:
                chunks.append(cur.strip())
                cur = cur[-overlap:] if overlap > 0 else ""
            cur += ("\n\n" if cur else "") + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def collect_documents(data_dir: Path, chunk_size: int, overlap: int) -> Tuple[list, list]:
    """Return (texts, metadatas) for all files."""
    texts, metas = [], []
    files = list(iter_all_files(data_dir))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")
    progress = tqdm(files, desc="Reading + chunking files", unit="file")
    for f in progress:
        try:
            raw = read_text_file(f)
            chunks = chunk_text(raw, chunk_size, overlap)
            for idx, c in enumerate(chunks):
                texts.append(c)
                metas.append({"source": str(f.relative_to(data_dir)), "chunk_id": idx})
        except Exception as e:
            tqdm.write(f"Skipping {f} due to error: {e}")
    return texts, metas