import json

import numpy as np
import pytest

from evaluate import precision_recall, relevant_ids
from hybridrag.chunking import FixedChunker, RecursiveChunker
from hybridrag.config import RetrievalConfig, load_config
from hybridrag.generator import REFUSAL, generate_answer
from hybridrag.indexer import build_index, load_index
from hybridrag.ingest import clean_text, extract_document
from hybridrag.retrieval import Retriever, SearchHit, evidence_gate, rrf_fuse
from hybridrag.schemas import Chunk, PageRecord, TextBlock


def chunk(cid=0, text="candidate key uniquely identifies a record", doc="db_design_en", page=37):
    return Chunk(chunk_id=cid, text=text, source="book.pdf", doc_id=doc, page=page,
                 page_label=str(page), section="Chapter 8 > Keys", strategy="recursive")


class ToyEncoder:
    """Vector xác định chỉ dùng trong test; không phải embedding chạy ứng dụng."""

    def encode(self, texts, query=False):
        return np.array([[1 + t.lower().count("key"), 1 + t.lower().count("database"), 1] for t in texts], dtype=np.float32)


def test_pdf_cleanup_preserves_text_and_removes_footers():
    assert clean_text("multi-\nlingual   text\n\n\n17 | Chapter 8 Keys\n") == "multilingual text"
    assert clean_text("Chapter 8 Keys | 17\nvalid") == "valid"
    assert clean_text("1\n2\n3") == "1\n2\n3"  # Số trong bảng không phải footer.


def test_real_pdfs_have_all_pages_offsets_chapters_and_no_footer():
    from hybridrag.config import ROOT

    for doc in load_config().documents:
        pages = extract_document(ROOT / "books" / doc.filename, doc)
        assert len(pages) == doc.expected_pages
        assert all(int(p.page_label) == p.page + doc.page_offset for p in pages)
        assert all(p.text.strip() for p in pages)
        assert all(p.text.splitlines()[-1].strip() != p.page_label for p in pages)
        if doc.doc_id == "ml_yearning_vi":
            assert pages[-1].section.startswith("Chương 35.")
        else:
            assert pages[-1].section.startswith("Chapter 8 ")


def test_fixed_overlap_and_page_provenance():
    pages = [PageRecord(doc_id="a", source="a.pdf", page=i, page_label=str(i + 3), section="Chương 1",
                        text=" ".join(str(n) for n in range((i - 1) * 5, i * 5))) for i in (1, 2)]
    chunks = FixedChunker(size=7, overlap=2).chunk(pages)
    assert chunks[0].text == "0 1 2 3 4 5 6"
    assert chunks[1].text == "5 6 7 8 9"
    assert chunks[0].pages == [1, 2]
    assert chunks[1].page == 2
    assert chunks[1].page_label == "5"


def test_recursive_keeps_sentence_across_pages_and_separates_chapters():
    pages = [PageRecord(doc_id="a", source="a.pdf", page=1, page_label="4", section="Chương 1", text="Một câu còn"),
             PageRecord(doc_id="a", source="a.pdf", page=2, page_label="5", section="Chương 1", text="tiếp diễn ở trang sau."),
             PageRecord(doc_id="a", source="a.pdf", page=3, page_label="6", section="Chương 2", text="Chương tiếp theo.")]
    chunks = RecursiveChunker(target=10, maximum=12).chunk(pages)
    assert len(chunks) == 2
    assert chunks[0].text == "Một câu còn tiếp diễn ở trang sau."
    assert chunks[0].pages == [1, 2]
    assert chunks[1].section == "Chương 2"


def test_recursive_splits_long_paragraph_at_sentences():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
    page = PageRecord(doc_id="a", source="a.pdf", page=1, page_label="1", section="Chapter 1", text=text)
    chunks = RecursiveChunker(target=5, maximum=10).chunk([page])
    assert [c.text for c in chunks] == ["One two three four five.", "Six seven eight nine ten.", "Eleven twelve thirteen fourteen fifteen."]


def test_index_ids_round_trip_and_corruption_detected(tmp_path):
    chunks = [chunk(17), chunk(42, "database stores related data")]
    build_index(chunks, tmp_path, ToyEncoder(), "test-only")
    loaded, index, bm25, manifest = load_index(tmp_path)
    assert [c.chunk_id for c in loaded] == [17, 42]
    assert index.ntotal == manifest["count"] == 2
    corpus = json.loads((tmp_path / "bm25.json").read_text())
    corpus["ids"].reverse()
    (tmp_path / "bm25.json").write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="lệch"):
        load_index(tmp_path)


def test_document_filter_before_dense_top_n(tmp_path):
    chunks = [chunk(i, "key key key key", doc="ml_yearning_vi", page=1) for i in range(12)] + [chunk(99, "database")]
    build_index(chunks, tmp_path, ToyEncoder(), "test-only")
    retriever = Retriever(tmp_path, encoder=ToyEncoder())
    assert retriever.search_faiss("key", top_n=1, doc_id="db_design_en")[0].chunk.chunk_id == 99
    assert not retriever.search(" ")
    assert not retriever.search_faiss("key", page=50)


def test_rrf_uses_ranks_and_merges_same_id():
    a, b = chunk(10), chunk(20)
    hits = rrf_fuse([SearchHit(a, 900, bm25_score=900), SearchHit(b, 1, bm25_score=1)],
                    [SearchHit(b, 0.9, dense_score=0.9), SearchHit(a, 0.8, dense_score=0.8)])
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == 10  # Hòa điểm: ID tăng dần.
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0].bm25_score == 900 and hits[0].dense_score == 0.8


def test_gate_uses_cosine_not_rrf_and_refuses_without_api(monkeypatch):
    hits = [SearchHit(chunk(), score=100, bm25_score=20, dense_score=0.1)]
    assert not evidence_gate(hits, RetrievalConfig())
    monkeypatch.setattr("hybridrag.generator.call_llm", lambda *_: pytest.fail("Không được gọi LLM khi thiếu căn cứ"))
    assert generate_answer("unknown", hits, dry_run=False).text == REFUSAL


def test_generation_validates_citations_and_dry_run(monkeypatch):
    hits = [SearchHit(chunk(), 0.9, dense_score=0.95)]
    dry = generate_answer("candidate key?", hits, dry_run=True)
    assert dry.dry_run and "Dry-run" in dry.text
    assert "trang 37" in dry.citations[0]
    monkeypatch.setattr("hybridrag.generator.call_llm", lambda *_: "Một khóa định danh duy nhất. [S9]")
    assert generate_answer("key?", hits, dry_run=False).refused
    monkeypatch.setattr("hybridrag.generator.call_llm", lambda *_: "Một khóa định danh duy nhất. [S1]")
    answer = generate_answer("key?", hits, dry_run=False)
    assert not answer.refused and answer.citations[0].startswith("[S1] Nguồn:")


def test_metrics_have_explicit_ground_truth_and_deduplicate():
    assert precision_recall([1, 1, 9], {1, 2}) == (1 / 3, 1 / 2)
    assert relevant_ids([chunk(10), chunk(11, page=38)], "db_design_en", {37}) == {10}
    with pytest.raises(ValueError):
        precision_recall([1], set())
