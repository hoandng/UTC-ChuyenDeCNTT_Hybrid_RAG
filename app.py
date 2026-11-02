"""Chạy bằng: streamlit run app.py."""
import os
import time

import httpx
import streamlit as st

from hybridrag.config import ROOT, load_config
from hybridrag.generator import generate_answer
from hybridrag.retrieval import MODES, Retriever

st.set_page_config(page_title="Trợ lý giáo trình · Hybrid RAG", page_icon="📚", layout="wide")
config = load_config()


@st.cache_resource
def get_retriever(strategy, manifest_stamp, device):
    return Retriever(ROOT / "indexes" / strategy, config.retrieval, device=device)


def show_hits(hits):
    st.subheader("Đoạn tài liệu làm căn cứ")
    for i, hit in enumerate(hits, 1):
        c = hit.chunk
        with st.expander(f"[S{i}] {c.section} · PDF trang {', '.join(map(str, c.pages))} · điểm {hit.score:.4f}"):
            st.caption(f"{c.source} · chunk_id={c.chunk_id} · {c.block_type}")
            st.write(c.text)
            st.json({"score": hit.score, "bm25": hit.bm25_score, "cosine": hit.dense_score})
            source = ROOT / "data/raw" / c.source
            if source.exists():
                st.download_button("Tải PDF để đối chiếu", source.read_bytes(), file_name=c.source,
                                   mime="application/pdf", key=f"pdf-{i}")


st.title("📚 Trợ lý hỏi đáp giáo trình")
st.caption("Machine Learning Yearning · Database Design | BM25 + FAISS + RRF")
with st.sidebar:
    st.header("Tra cứu và đối sánh")
    titles = {None: "Cả hai giáo trình", **{d.doc_id: d.title for d in config.documents}}
    document = st.selectbox("Tài liệu", list(titles), format_func=titles.get)
    strategy = st.radio("Chiến lược chia đoạn", ["recursive", "fixed"], horizontal=True)
    mode = st.selectbox("Phương pháp tìm kiếm", MODES)
    position = st.radio("Xem đoạn trích", ["Sau câu trả lời", "Trước câu trả lời"])
    with st.expander("Lọc theo vị trí"):
        page = st.number_input("Trang PDF (0 = tất cả)", min_value=0, max_value=60, value=0)
        chapter = st.number_input("Chương (0 = tất cả)", min_value=0, max_value=35, value=0)
    dry_run = st.checkbox("Dry-run: xem trích đoạn", value=os.getenv("LLM_PROVIDER", "dry-run") == "dry-run")
    if not dry_run and os.getenv("LLM_PROVIDER", "dry-run") == "dry-run":
        st.info("Cần chọn LLM_PROVIDER và đặt LLM_MODEL trong .env để sinh đáp án.")
    st.caption("Số trang trong trích dẫn là trang in trên sách; bộ lọc dùng trang PDF.")

manifest = ROOT / "indexes" / strategy / "manifest.json"
if not manifest.exists():
    st.info("Chưa có chỉ mục. Chạy các lệnh sau trong thư mục dự án:")
    st.code("make setup\nmake pipeline\nmake app", language="bash")
    st.stop()

try:
    retriever = get_retriever(strategy, manifest.stat().st_mtime_ns, config.embedding.device)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f"Không thể nạp chỉ mục: {error}")
    st.stop()

samples = ["Tập phát triển và tập kiểm tra nên có cùng phân phối vì sao?",
           "What is a candidate key?", "So sánh độ chệch và phương sai, nêu cách khắc phục từng loại."]
st.write("Thử một câu hỏi:")
for column, sample in zip(st.columns(3), samples):
    if column.button(sample, use_container_width=True):
        st.session_state["question"] = sample

with st.form("ask"):
    question = st.text_area("Câu hỏi của bạn", key="question", placeholder="Nhập câu hỏi về nội dung hai giáo trình…")
    submitted = st.form_submit_button("Tìm kiếm và trả lời", type="primary")

if submitted:
    st.session_state.pop("last_result", None)
    if not question.strip():
        st.warning("Nhập câu hỏi trước khi tra cứu.")
    else:
        try:
            start = time.perf_counter()
            with st.spinner("Đang tìm các đoạn liên quan…"):
                hits = retriever.search(question, mode, document, page or None, chapter or None)
            retrieval_ms = (time.perf_counter() - start) * 1000
            if position == "Trước câu trả lời":
                show_hits(hits)
            with st.spinner("Đang chuẩn bị câu trả lời…"):
                answer = generate_answer(question, hits, config.retrieval, dry_run=dry_run)
            st.session_state["last_result"] = dict(
                selection=(strategy, mode, document, page, chapter, dry_run),
                question=question,
                hits=hits, answer=answer, retrieval_ms=retrieval_ms,
                total_seconds=time.perf_counter() - start,
            )
        except (OSError, ValueError, RuntimeError, ImportError, httpx.HTTPError) as error:
            st.error(f"Chưa thể hoàn tất truy vấn: {error}")
            st.caption("Kiểm tra kết nối tải embedding, .env và lệnh make index.")

result = st.session_state.get("last_result")
if result and result["selection"] == (strategy, mode, document, page, chapter, dry_run):
    # Giữ kết quả khi rerun do tải PDF hoặc đổi thứ tự hiển thị.
    if position == "Trước câu trả lời" and not submitted:
        show_hits(result["hits"])
    answer = result["answer"]
    st.subheader("Kết quả")
    st.caption(f"Câu hỏi đã tra cứu: {result['question']}")
    if answer.refused:
        st.info(answer.text)
    else:
        st.markdown(answer.text)
        for source in answer.citations:
            st.caption(source)
    a, b, c, d = st.columns(4)
    a.metric("Tìm kiếm", f"{result['retrieval_ms']:.0f} ms")
    b.metric("Tổng thời gian", f"{result['total_seconds']:.2f} s")
    c.metric("Chunk đã nạp", len(retriever.chunks))
    d.metric("Độ dài context", f"{len(answer.context.split())} từ")
    if position == "Sau câu trả lời":
        show_hits(result["hits"])
