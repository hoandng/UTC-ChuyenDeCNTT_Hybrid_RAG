# GEMINI.md — Quy tắc dự án Hybrid RAG

## 1. Định hướng dự án
- **Học phần:** Chuyên đề Công nghệ thông tin (UTC) — Project 1: Trợ lý hỏi đáp tài liệu học tập.
- **Tài liệu tham chiếu:** Luôn đọc `Agent.md` và `Architecture.md` trước khi code.
- **Phương châm cốt lõi:** Bám sát bài toán và dữ liệu đầu vào (2 file giáo trình), tinh gọn, thực tế, **tuyệt đối không over-engineer hoặc thêm công nghệ phức tạp ngoài phạm vi**.

## 2. Quy tắc code & Kiến trúc
- **Ngôn ngữ & Thư viện:** Python 3.10+, PyMuPDF (`fitz`), FAISS-cpu, SentenceTransformers, rank_bm25, Streamlit, Pydantic.
- **Cấu trúc gọn nhẹ:** Giữ toàn bộ logic nghiệp vụ trong 6 file tại `src/hybridrag/`:
  - `config.py`: Đọc config YAML & env.
  - `schemas.py`: Pydantic schema cho Chunk & PageRecord.
  - `ingest.py`: Đọc PDF theo trang, làm sạch, tách chương/mục.
  - `chunking.py`: 2 chiến lược `fixed` và `recursive`.
  - `indexer.py`: Embedding, FAISS index, BM25 index.
  - `retrieval.py`: Hybrid search (BM25 + FAISS + RRF fusion).
  - `generator.py`: Prompt, gọi LLM, format trích dẫn.
  - Giao diện đặt tại `app.py`, đánh giá đặt tại `evaluate.py`.
- **Không tự ý thêm:**
  - Không thêm Semantic Chunking phức tạp (chỉ dùng `fixed` và `recursive`).
  - Không thêm Query Decomposition LLM Agent (RRF Top-5 đã đủ cho câu hỏi so sánh).
  - Không cài thêm các thư viện NLP tiếng Việt nặng nề (dùng regex/split chuẩn hóa đơn giản).
  - Không tạo quá nhiều file/module con gây loãng cấu trúc.

## 3. Quy tắc dữ liệu & Truy vết
- Đọc PDF **theo từng trang**, bảo toàn thông tin `page` (trang PDF từ 1) và `page_label` (trang in trên sách).
  - ML Yearning: `page_label = page + 3`.
  - Database Design: `page_label = page`.
- `chunk_id` phải **thống nhất hoàn toàn** giữa FAISS và BM25 trong cùng 1 strategy.
- LLM chỉ được phép trả lời dựa trên Context được cung cấp, kèm trích dẫn: `Nguồn: <Tên tài liệu>, <Chương/Mục>, trang <Số trang>`. Nếu thiếu dữ liệu phải từ chối suy đoán.
- Tuyệt đối không commit API key hay file bí mật.

