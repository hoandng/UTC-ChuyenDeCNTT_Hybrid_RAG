"""Index FAISS exact cosine và BM25 có chung thứ tự chunk_id."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .config import ROOT, EmbeddingConfig, load_config
from .schemas import Chunk, read_jsonl, write_jsonl


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def fingerprint(chunks: list[Chunk]) -> str:
    payload = "\n".join(c.model_dump_json() for c in chunks)
    return hashlib.sha256(payload.encode()).hexdigest()


class Encoder:
    def __init__(self, config: EmbeddingConfig):
        import os

        os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(config.model, device=config.device)

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        # E5 yêu cầu prefix kể cả khi văn bản không phải tiếng Anh.
        if "e5" in self.config.model.lower():
            texts = [("query: " if query else "passage: ") + text for text in texts]
        vectors = self.model.encode(texts, batch_size=self.config.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=not query)
        return np.ascontiguousarray(vectors, dtype=np.float32)


def build_index(chunks: list[Chunk], directory: Path, encoder, model_name: str) -> dict:
    if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Cần chunk không rỗng, ID duy nhất")
    vectors = np.ascontiguousarray(encoder.encode([c.text for c in chunks]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding không hợp lệ")
    faiss.normalize_L2(vectors)
    ids = np.array([c.chunk_id for c in chunks], dtype=np.int64)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)
    tokens = [tokenize(c.text) for c in chunks]
    BM25Okapi(tokens)  # Kiểm tra corpus trước khi ghi dữ liệu.
    assert faiss.vector_to_array(index.id_map).tolist() == ids.tolist()
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "chunks.jsonl", chunks)
    # Lưu corpus JSON, dựng lại BM25 lúc nạp; không deserialize pickle.
    (directory / "bm25.json").write_text(json.dumps({"ids": ids.tolist(), "tokens": tokens}, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(directory / "faiss.index"))
    manifest = dict(model=model_name, dimension=vectors.shape[1], count=len(chunks),
                    fingerprint=fingerprint(chunks), strategy=chunks[0].strategy,
                    normalized=True, query_prefix="query: " if "e5" in model_name.lower() else "")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    chunks = read_jsonl(directory / "chunks.jsonl", Chunk)
    corpus = json.loads((directory / "bm25.json").read_text())
    index = faiss.read_index(str(directory / "faiss.index"))
    ids = [c.chunk_id for c in chunks]
    if (len(set(ids)) != len(ids) or faiss.vector_to_array(index.id_map).tolist() != ids
            or corpus["ids"] != ids or corpus["tokens"] != [tokenize(c.text) for c in chunks]
            or manifest["fingerprint"] != fingerprint(chunks)
            or manifest["count"] != len(chunks) or index.d != manifest["dimension"]):
        raise ValueError("Chỉ mục lệch metadata/IDs; chạy lại make index")
    return chunks, index, BM25Okapi(corpus["tokens"]), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "recursive", "all"], default="all")
    args = parser.parse_args()
    config = load_config()
    encoder = Encoder(config.embedding)
    for strategy in (["fixed", "recursive"] if args.strategy == "all" else [args.strategy]):
        chunks = read_jsonl(ROOT / "data/processed" / strategy / "chunks.jsonl", Chunk)
        manifest = build_index(chunks, ROOT / "indexes" / strategy, encoder, config.embedding.model)
        print(strategy, manifest)


if __name__ == "__main__":
    main()
