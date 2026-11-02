"""Hai chiến lược dùng đơn vị từ tách bằng khoảng trắng (không phải subword)."""
import json
import re
from collections import Counter
from statistics import mean

from .config import ROOT, load_config
from .schemas import Chunk, PageRecord, read_jsonl, write_jsonl


def make_chunk(text: str, spans, strategy: str, chunk_id: int) -> Chunk:
    first, block_type, section = spans[0]
    return Chunk(chunk_id=chunk_id, text=text, source=first.source, doc_id=first.doc_id,
                 page=first.page, page_label=first.page_label, section=section,
                 strategy=strategy, block_type=block_type,
                 pages=sorted({p.page for p, _, _ in spans}))


class FixedChunker:
    def __init__(self, size: int = 256, overlap: int = 35):
        if not 0 <= overlap < size:
            raise ValueError("Cần 0 <= overlap < size")
        self.size, self.overlap = size, overlap

    def chunk(self, pages: list[PageRecord]) -> list[Chunk]:
        # Duy trì ánh xạ từng từ về trang/mục, không gộp qua tài liệu.
        chunks = []
        for doc_id in dict.fromkeys(p.doc_id for p in pages):
            words, origins = [], []
            for page in (p for p in pages if p.doc_id == doc_id):
                blocks = page.blocks or []
                for text, kind, section in ([(b.text, b.block_type, b.section) for b in blocks] or [(page.text, "content", page.section)]):
                    tokens = text.split()
                    words.extend(tokens)
                    origins.extend([(page, kind, section)] * len(tokens))
            for start in range(0, len(words), self.size - self.overlap):
                end = min(start + self.size, len(words))
                chunks.append(make_chunk(" ".join(words[start:end]), origins[start:end], "fixed", len(chunks)))
                if end == len(words):
                    break
        return chunks


class RecursiveChunker:
    def __init__(self, target: int = 300, maximum: int = 450):
        if not 0 < target <= maximum:
            raise ValueError("Cần 0 < target <= maximum")
        self.target, self.maximum = target, maximum

    def chunk(self, pages: list[PageRecord]) -> list[Chunk]:
        chunks, groups = [], []
        for page in pages:
            for text, kind, section in ([(b.text, b.block_type, b.section) for b in page.blocks] or [(page.text, "content", page.section)]):
                if not text.strip():
                    continue
                key = (page.doc_id, section, kind)
                if not groups or groups[-1][0] != key:
                    groups.append((key, []))
                groups[-1][1].append((text, (page, kind, section)))

        for _, blocks in groups:
            # Offsets giữ đúng trang khi câu tiếp diễn ở trang tiếp theo.
            text, offsets = "", []
            for part, origin in blocks:
                # Ngắt dòng trong PDF không đồng nghĩa kết thúc câu.
                part = " ".join(part.split())
                separator = "\n\n" if text.endswith((".", "?", "!", ":")) else " "
                start = len(text) + (len(separator) if text else 0)
                text += (separator if text else "") + part
                offsets.append((start, len(text), origin))
            units = []
            for paragraph in re.finditer(r"[^\n]+", text):
                value = paragraph.group()
                if len(value.split()) <= self.maximum:
                    units.append((value, paragraph.start(), paragraph.end()))
                else:
                    for sentence in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", value):
                        if sentence.group().strip():
                            units.append((sentence.group().strip(), paragraph.start() + sentence.start(), paragraph.start() + sentence.end()))
            pending, spans, count = [], [], 0

            def flush():
                nonlocal pending, spans, count
                if pending:
                    chunks.append(make_chunk("\n\n".join(pending), spans, "recursive", len(chunks)))
                pending, spans, count = [], [], 0

            for value, start, end in units:
                length = len(value.split())
                if pending and (count + length > self.maximum or count >= self.target):
                    flush()
                pending.append(value)
                spans.extend(origin for left, right, origin in offsets if left < end and right > start)
                count += length
                if count >= self.target:
                    flush()
            flush()
        return chunks


def main() -> None:
    config = load_config()
    pages = [p for doc in config.documents for p in read_jsonl(ROOT / "data/interim" / f"{doc.doc_id}_pages.jsonl", PageRecord)]
    cfg = config.chunking
    for strategy, chunker in [("fixed", FixedChunker(cfg.fixed_size, cfg.fixed_overlap)),
                              ("recursive", RecursiveChunker(cfg.recursive_target, cfg.recursive_max))]:
        chunks = chunker.chunk(pages)
        if not chunks:
            raise ValueError("Không có nội dung để chia chunk")
        directory = ROOT / "data/processed" / strategy
        write_jsonl(directory / "chunks.jsonl", chunks)
        lengths = [len(c.text.split()) for c in chunks]
        stats = dict(total=len(chunks), mean_words=mean(lengths), min_words=min(lengths), max_words=max(lengths),
                     by_document=dict(Counter(c.doc_id for c in chunks)),
                     oversized_sentences=sum(n > cfg.recursive_max for n in lengths) if strategy == "recursive" else 0)
        (directory / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(strategy, stats)


if __name__ == "__main__":
    main()
