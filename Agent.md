# Agent.md — Hướng dẫn cho Coding Agent

> **Dự án:** Project 1 — Trợ lý hỏi đáp tài liệu học tập (Hybrid RAG + So sánh 2 chiến lược Chunking)
> **Học phần:** Chuyên đề Công nghệ thông tin — ĐH Giao thông Vận tải
> **Định hướng cốt lõi:** Bám sát yêu cầu đề bài trong `docs/`, tối ưu cho 2 giáo trình đầu vào, giữ cấu trúc code đơn giản, mạch lạc, không thêm các công nghệ phức tạp ngoài phạm vi.

---

## 1. Nguyên tắc triển khai (Core Rules)

1. **Đọc PDF theo từng trang:** Tuyệt đối không nối cả PDF thành 1 chuỗi dài rồi mất số trang. Mọi chunk đều phải lưu đúng `page` (trang PDF từ 1) và `page_label` (trang in trên sách).
   - Với sách Học máy (`ml_yearning_vi`): `page_label = page + 3`.
   - Với sách CSDL (`db_design_en`): `page_label = page`.
2. **Đồng nhất `chunk_id`:** Trong cùng một chiến lược chunking, BM25 và FAISS phải dùng chung một danh sách `chunk_id`.
3. **Hai chiến lược Chunking để đối sánh:**
   - `fixed`: Baseline cắt cố định 256 tokens, overlap 15%.
   - `recursive`: Tách theo Chapter → Heading → Đoạn → Câu (target 300, max 450 tokens). Không cắt ngang câu hay qua ranh giới chương.
4. **Hybrid Search chuẩn:** BM25 Top-8 + FAISS Top-8 $\rightarrow$ Hợp nhất qua RRF ($k=60$) $\rightarrow$ Lấy Top-5 chunk làm ngữ cảnh.
5. **LLM chỉ trả lời từ Context:** Có trích dẫn rõ ràng `Nguồn: <Tài liệu>, <Chương/Mục>, trang <Số trang>`. Nếu ngữ cảnh không có thông tin $\rightarrow$ từ chối suy đoán.
6. **Không over-engineer:** Tập trung vào 6 file nghiệp vụ trong `src/hybridrag/`, không chia nhỏ thành quá nhiều module phụ trợ, không dùng các thư viện NLP nặng hay agent phân rã truy vấn nhiều bước.
7. **Bảo mật:** Không commit API key lên git. Đọc cấu hình từ `.env` và `configs/default.yaml`.

---

## 2. Công nghệ sử dụng

| Hạng mục            | Công nghệ                                                                  | Ghi chú                                                   |
| ------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Ngôn ngữ**        | Python 3.10+                                                               | Type hints rõ ràng, code dễ hiểu                          |
| **Đọc PDF**         | `PyMuPDF` (`fitz`)                                                         | Trích xuất text, số trang, font size để nhận diện heading |
| **Embedding**       | `sentence-transformers` (`BAAI/bge-m3` hoặc `keepitreal/vietnamese-sbert`) | Đa ngôn ngữ, hoạt động tốt trên cả CPU và GPU             |
| **Vector DB**       | `faiss-cpu` (`IndexFlatIP` + `IndexIDMap`)                                 | Tìm kiếm chính xác (Exact search), gán đúng `chunk_id`    |
| **Keyword Search**  | `rank_bm25` (`BM25Okapi`)                                                  | Tokenizer regex chuẩn hóa chữ thường, đơn giản, hiệu quả  |
| **LLM Client**      | API linh hoạt qua `.env` (Gemini / OpenAI / Groq / Ollama)                 | Hỗ trợ dry-run nếu chưa có API key                        |
| **Giao diện Web**   | `streamlit`                                                                | Trực quan, chạy nhanh, dễ demo                            |
| **Cấu hình & Data** | `pydantic`, `pyyaml`                                                       | Validate schema metadata của chunk                        |

---

## 3. Lộ trình Milestones (M0 – M7)

### M0 — Khởi tạo dự án

- Tạo cấu trúc thư mục tinh gọn.
- Tạo `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, `configs/default.yaml`.
- Copy 2 file PDF từ `books/processed/` sang `data/raw/`.
- **DoD:** Chạy `python -c "import fitz, faiss, sentence_transformers, rank_bm25, streamlit; print('OK')"` thành công.

### M1 — Ingest & Làm sạch PDF

- Viết `src/hybridrag/ingest.py`:
  - Đọc từng trang của 2 PDF bằng `fitz`.
  - Làm sạch: Xóa footer số trang rác, chuẩn hóa khoảng trắng/dòng trống, xử lý từ ngắt dòng cơ bản.
  - Nhận diện Chương/Heading và gắn nhãn `page`, `page_label`, `section`.
  - Xuất ra `data/interim/<doc_id>_pages.jsonl`.
- **DoD:** Trích xuất đủ 60 trang (ML) và 48 trang (DB). Kiểm tra ngẫu nhiên 5 trang thấy footer đã sạch và `page_label` chuẩn xác.

### M2 — Chunking (Fixed & Recursive)

- Viết `src/hybridrag/chunking.py`:
  - `FixedChunker`: Cắt đều 256 từ/tokens, overlap 35.
  - `RecursiveChunker`: Cắt ưu tiên Chapter → Heading → Đoạn → Câu.
  - Validate mỗi chunk bằng Pydantic `Chunk` schema.
  - Lưu vào `data/processed/fixed/chunks.jsonl` và `data/processed/recursive/chunks.jsonl`.
- **DoD:** Xuất file thống kê `stats.json` (tổng số chunk, độ dài trung bình, min/max). `recursive` không cắt qua ranh giới chương.

### M3 — Indexing (FAISS & BM25)

- Viết `src/hybridrag/indexer.py`:
  - Tạo vector embedding cho toàn bộ chunk bằng SentenceTransformer.
  - Build FAISS index (`IndexFlatIP` + `IndexIDMap` với ID = `chunk_id`).
  - Build BM25 index trên token của chunk với danh sách IDs khớp hoàn toàn.
  - Lưu vào `indexes/<strategy>/`.
- **DoD:** `assert set(faiss_ids) == set(bm25_ids) == {c.chunk_id for c in chunks}` chạy xanh.

### M4 — Hybrid Retrieval & RRF

- Viết `src/hybridrag/retrieval.py`:
  - Hàm `search_bm25(query, top_n=8)`.
  - Hàm `search_faiss(query, top_n=8)`.
  - Hàm `rrf_fuse(bm25_results, faiss_results, k=60, top_k=5)`.
  - Cung cấp 3 chế độ: `hybrid_rrf`, `bm25_only`, `faiss_only`.
- **DoD:** Truy vấn thử 3 câu hỏi mẫu, trả về Top-5 chunk kèm điểm RRF và vị trí trang/chương.

### M5 — Generation & Trích dẫn

- Viết `src/hybridrag/generator.py`:
  - LLM client kết nối qua API key từ `.env`.
  - Đóng gói ngữ cảnh `[S1]...[S5]` kèm nguồn tài liệu, chương/mục, số trang.
  - Prompt ràng buộc: Không suy đoán ngoài ngữ cảnh, từ chối khi không có dữ liệu, trích dẫn nguồn chuẩn.
- **DoD:** Chạy thử câu hỏi trong sách $\rightarrow$ trả lời đúng kèm citation. Chạy thử câu hỏi ngoài sách $\rightarrow$ từ chối lịch sự.

### M6 — Giao diện Demo Streamlit

- Viết `app.py`:
  - Sidebar: Chọn sách, chọn chiến lược Chunking (`fixed` / `recursive`), chọn chế độ tìm kiếm, chọn xem Top-K trước hay sau câu trả lời.
  - Chat / Form nhập câu hỏi + Gợi ý các câu hỏi mẫu (định nghĩa, so sánh, tìm theo trang/chương).
  - Khối hiển thị đáp án + Citation chuẩn.
  - Bảng thống kê: Thời gian truy vấn, số chunk nạp, chi tiết các chunk Top-K.
- **DoD:** Chạy `streamlit run app.py` mượt mà trên trình duyệt.

### M7 — Đánh giá & Hoàn thiện sản phẩm

- Tạo `data/eval/eval_questions.csv` gồm đúng 15 câu (5 từ khóa, 5 diễn đạt lại, 5 tổng hợp).
- Viết `evaluate.py`: Chạy 15 câu trên cả 2 chiến lược chunking và 3 chế độ retrieval, đo đạc tự động **Precision@3** và **Recall@3**, lưu ra `data/eval/results.csv`.
- Ghi nhận 3 case điển hình (1 thành công, 1 thất bại, 1 khác biệt giữa 2 chiến lược) để đưa vào Báo cáo & Slide.
- **DoD:** Toàn bộ metrics được sinh tự động từ script thật; README hướng dẫn chạy từ A-Z.

---

## 4. Các lệnh thực thi nhanh (Makefile)

```bash
make setup     # Cài đặt môi trường & dependencies
make ingest    # Chạy M1: Đọc và làm sạch 2 file PDF
make chunk     # Chạy M2: Tạo chunks cho cả Fixed và Recursive
make index     # Chạy M3: Build chỉ mục FAISS và BM25
make app       # Chạy M6: Khởi chạy ứng dụng Streamlit demo
make eval      # Chạy M7: Chạy kiểm thử đánh giá 15 câu hỏi (P@3, R@3)
```
