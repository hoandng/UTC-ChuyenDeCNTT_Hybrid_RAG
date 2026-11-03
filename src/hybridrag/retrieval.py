"""BM25 Top-8 + FAISS Top-8 → RRF(k=60) → Top-5."""
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .config import EmbeddingConfig, RetrievalConfig
from .indexer import Encoder, load_index, tokenize
from .schemas import Chunk

MODES = ("hybrid_rrf", "bm25_only", "faiss_only")


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None


def rrf_fuse(bm25_results: list[SearchHit], faiss_results: list[SearchHit], k: int = 60, top_k: int = 5) -> list[SearchHit]:
    if k <= 0 or top_k <= 0:
        raise ValueError("k và top_k phải dương")
    fused = {}
    for results in (bm25_results, faiss_results):
        seen = set()
        for rank, hit in enumerate(results, 1):
            cid = hit.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            item = fused.setdefault(cid, SearchHit(hit.chunk, 0))
            item.score += 1 / (k + rank)
            if hit.bm25_score is not None:
                item.bm25_score = hit.bm25_score
            if hit.dense_score is not None:
                item.dense_score = hit.dense_score
    return sorted(fused.values(), key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k]


class Retriever:
    def __init__(self, directory: Path, config: RetrievalConfig | None = None, encoder=None, device: str = "cuda"):
        self.chunks, self.index, self.bm25, self.manifest = load_index(directory)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.config = config or RetrievalConfig()
        self.encoder, self.device = encoder, device

    def _eligible(self, doc_id=None, page=None, chapter=None):
        import re

        def matches(c):
            chapter_match = re.match(r"(?:Chương|Chapter)\s+(\d+)\b", c.section, re.I)
            return ((doc_id is None or c.doc_id == doc_id)
                    and (page is None or page in c.pages)
                    and (chapter is None or (chapter_match and int(chapter_match[1]) == chapter)))
        return {c.chunk_id for c in self.chunks if matches(c)}

    def search_bm25(self, query: str, top_n: int = 8, doc_id=None, page=None, chapter=None) -> list[SearchHit]:
        eligible = self._eligible(doc_id, page, chapter)
        scores = self.bm25.get_scores(tokenize(query))
        rows = [(c, float(s)) for c, s in zip(self.chunks, scores) if c.chunk_id in eligible and s > 0]
        return [SearchHit(c, s, bm25_score=s) for c, s in sorted(rows, key=lambda x: (-x[1], x[0].chunk_id))[:top_n]]

    def search_faiss(self, query: str, top_n: int = 8, doc_id=None, page=None, chapter=None) -> list[SearchHit]:
        eligible = self._eligible(doc_id, page, chapter)
        if not eligible:
            return []
        if self.encoder is None:
            self.encoder = Encoder(EmbeddingConfig(model=self.manifest["model"], device=self.device))
        vector = np.ascontiguousarray(self.encoder.encode([query], query=True), dtype=np.float32)
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError("Vector truy vấn không hợp lệ")
        faiss.normalize_L2(vector)
        # Exact scan corpus nhỏ, lọc TRƯỚC khi lấy Top-N để không làm rơi sách đã chọn.
        scores, ids = self.index.search(vector, self.index.ntotal)
        return [SearchHit(self.by_id[int(cid)], float(score), dense_score=float(score))
                for cid, score in zip(ids[0], scores[0]) if int(cid) in eligible][:top_n]

    def search(self, query: str, mode: str = "hybrid_rrf", doc_id=None, page=None, chapter=None, top_k=None) -> list[SearchHit]:
        if mode not in MODES:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")
        if not query.strip():
            return []
        limit = top_k or self.config.top_k
        candidates = max(self.config.candidates, limit)
        filters = dict(doc_id=doc_id, page=page, chapter=chapter)
        bm25 = self.search_bm25(query, candidates, **filters) if mode != "faiss_only" else []
        dense = self.search_faiss(query, candidates, **filters) if mode != "bm25_only" else []
        if mode == "bm25_only":
            return bm25[:limit]
        if mode == "faiss_only":
            return dense[:limit]
        return rrf_fuse(bm25, dense, self.config.rrf_k, limit)


def evidence_gate(hits: list[SearchHit], config: RetrievalConfig) -> bool:
    # Điểm RRF không thể dùng làm ngưỡng semantic relevance.
    if not hits:
        return False
    dense = [h.dense_score for h in hits if h.dense_score is not None]
    if dense:
        return max(dense) >= config.min_dense_similarity
    return max((h.bm25_score or 0) for h in hits) >= config.min_bm25_score
