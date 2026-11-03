# Trợ lý hỏi đáp giáo trình — Hybrid RAG

Ứng dụng Python cho Project 1, Chuyên đề CNTT, UTC. Tra cứu hai giáo trình song ngữ bằng **BM25 + FAISS exact cosine → RRF**, đối sánh **fixed** và **recursive**, trả lời từ context kèm nguồn và trang.

## Chạy từ đầu

Yêu cầu Python 3.10+, GPU NVIDIA có driver hỗ trợ CUDA 12.8, khoảng 4 GB VRAM và Internet ở lần cài thư viện/tải mô hình đầu tiên. Mô hình mặc định `intfloat/multilingual-e5-base` có trọng số khoảng 1,1 GB, chạy GPU với batch size 8. `make setup` ưu tiên `uv` nếu có, hoặc dùng `venv`/`pip`; cài PyTorch 2.9.1 CUDA 12.8 theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/). Có thể đổi phiên bản bằng biến `TORCH_VERSION` của Makefile.

```bash
make setup
cp .env.example .env
make pipeline     # ingest → chunk → index cho cả hai chiến lược
make app          # http://localhost:8501
```

Trong workspace đã cài thư viện và tạo chỉ mục, chỉ cần `make app`. Không cần kích hoạt virtualenv khi dùng Makefile.

Embedding khi tạo chỉ mục và mã hóa câu hỏi đều dùng GPU (`EMBEDDING_DEVICE=cuda` trong `.env`). BM25 và FAISS vẫn chạy CPU; LLM gọi API chạy tại provider, còn GPU cho Ollama do Ollama quản lý riêng. Chuyển CPU/GPU với cùng model không yêu cầu tạo lại chỉ mục. Sau khi đổi môi trường, dừng ứng dụng đang chạy và khởi động lại bằng `make app`.

Kiểm tra GPU và PyTorch sau khi cài:

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Nếu cần chạy CPU, đặt `EMBEDDING_DEVICE=cpu` trong `.env` và cài bằng `make setup TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.

Chạy từng bước hoặc không dùng `make`:

```bash
.venv/bin/python -m hybridrag.ingest
.venv/bin/python -m hybridrag.chunking
.venv/bin/python -m hybridrag.indexer
.venv/bin/python -m streamlit run app.py
.venv/bin/python evaluate.py
.venv/bin/python -m pytest -q
```

Hai PDF đầu vào có sẵn trong `books/`. Ingest tìm tại `data/raw/`, rồi `books/processed/`, rồi `books/`; tự sao chép vào `data/raw/`, kiểm tra đúng 60 và 48 trang. Không sửa PDF gốc. PDF ảnh không có text cần OCR riêng, ngoài phạm vi hai giáo trình này.

## Cách sử dụng

1. Chọn một hoặc cả hai sách, chiến lược chia đoạn và chế độ retrieval ở thanh bên.
2. Nhập câu hỏi, hoặc chọn câu hỏi mẫu. Có bộ lọc trang PDF/chương riêng để tra cứu vị trí chính xác.
3. Xem kết quả và nguồn, mở từng đoạn Top-5 để đối chiếu text, điểm BM25/cosine/RRF và tải PDF gốc. Có thể chọn hiển thị các đoạn trước hoặc sau đáp án.

Ví dụ: “Tập phát triển và tập kiểm tra nên có cùng phân phối vì sao?”, “What is a candidate key?”, “So sánh độ chệch và phương sai và cách khắc phục”.

Mặc định `LLM_PROVIDER=dry-run`: hiển thị **nguyên văn đoạn xếp hạng đầu**, ghi rõ chưa sinh đáp án; toàn bộ Top-5 nằm ở các khối mở rộng. Retrieval/embedding vẫn chạy thật; không thay embedding bằng vector ngẫu nhiên khi thiếu mô hình.

Để sinh đáp án, sửa `.env`, chọn provider và model hiện có trong tài khoản hoặc máy của bạn:

```dotenv
LLM_PROVIDER=gemini
LLM_API_KEY=your-key
LLM_MODEL=your-available-model
```

Các provider: `gemini`, `openai`, `groq`, `ollama`. Gemini dùng `generateContent`; ba provider còn lại dùng `/chat/completions` tương thích OpenAI. Với Ollama, không cần API key; đặt `LLM_MODEL` đúng tên model đã tải và `LLM_BASE_URL=http://localhost:11434/v1`. Với endpoint tùy chỉnh, `LLM_BASE_URL` là URL gốc **chưa có** `/chat/completions`. Không có tên model LLM cố định để tránh phụ thuộc model đã ngừng cung cấp. Khởi động lại ứng dụng khi đổi `.env`.

`LLM_TIMEOUT` mặc định 60 giây. Lỗi API hiện thông báo, không giả lập câu trả lời. Khi chọn provider từ xa, câu hỏi và các đoạn context được gửi tới provider đó. `.env`, API key, PDF sao chép, cache mô hình và chỉ mục được gitignore.

## Pipeline và các quyết định triển khai

- `config.py`, `schemas.py`: cấu hình YAML/env và metadata Pydantic. Các module nghiệp vụ giữ đúng danh sách trong ba tài liệu thiết kế; không dùng framework RAG hay agent phân rã câu hỏi.
- `ingest.py`: đọc PyMuPDF **theo từng trang**, chuẩn hóa khoảng trắng, nối từ ngắt dòng, bỏ footer theo vị trí. Phân biệt footer số trang với số trong bảng. Nhận diện chương, heading qua regex và tên/kích thước font, gắn `content`, `key_terms`, `exercises`, `attribution`.
- `chunking.py`: fixed 256 từ, overlap 35, có thể đi qua câu/chương nhưng không qua tài liệu. Recursive ưu tiên chương → heading → đoạn → câu, target 300, max 450 từ. Hai tài liệu thiết kế dùng “token/từ” chưa thống nhất; bản này dùng **từ tách khoảng trắng**, không gọi đó là subword token của embedding/LLM. Heading ngắn tạo chunk ngắn; không ép đủ 300 bằng cách gộp qua ranh giới mục. Nếu một câu riêng lẻ dài hơn max, giữ nguyên câu và ghi `oversized_sentences` trong thống kê thay vì cắt mất câu; hai PDF hiện tại không có trường hợp này.
- Mỗi chunk có `page`, `page_label`, `section`, `source`, `doc_id`, `chunk_id`, `strategy`, `block_type` và `pages` chứa **mọi trang PDF thực sự đóng góp text**. Metadata chính lấy ở đầu chunk. Citation liệt kê các trang đóng góp; ML cộng 3 để ra trang in, DB giữ nguyên. Chunk fixed có thể đi qua mục khác nên mục trong citation là mục bắt đầu.
- `indexer.py`: SentenceTransformer normalize L2, E5 có prefix `passage:`/`query:`, FAISS `IndexIDMap(IndexFlatIP)` + BM25Okapi. Chunk ID là số nguyên, cùng thứ tự ở cả hai index. Lưu corpus BM25 dạng JSON và dựng lại khi nạp, tránh pickle. Manifest lưu model, dimension, count và SHA-256 nội dung chunk; nạp sai ID hoặc metadata sẽ báo lỗi. Model truy vấn lấy từ manifest của index.
- `retrieval.py`: lấy Top-8 mỗi nhánh, `RRF(c)=Σ 1/(60+rank(c))`, rank bắt đầu từ 1, giữ Top-5. Các mode `bm25_only` và `faiss_only` không gọi nhánh còn lại. Bộ lọc sách/trang/chương áp dụng trước khi lấy Top-N. FAISS exact scan toàn corpus nhỏ để lọc chính xác. Không loại exercises hay thêm trọng số sau RRF để giữ phép so sánh đúng công thức; block type được hiển thị để phân tích.
- `generator.py`: evidence gate dùng **cosine thật** khi có, dùng ngưỡng BM25 ở mode lexical; không dùng điểm RRF làm độ tin cậy. Prompt chỉ cho phép context, yêu cầu nhãn `[S1]…`; thiếu nhãn hoặc nhãn không tồn tại sẽ từ chối. Nguồn được định dạng từ metadata, không nhờ LLM tự bịa số trang.

Gate là heuristic, không chứng minh ngữ cảnh đủ căn cứ. Ngưỡng mặc định nằm trong `configs/default.yaml`, phụ thuộc embedding và cần hiệu chỉnh bằng tập validation riêng. Chỉ kiểm tra được tính hợp lệ của nhãn citation bằng code; độ đúng, đủ và mức độ hỗ trợ từng khẳng định vẫn cần người chấm. Dry-run không thay thế kiểm tra generation bằng LLM thật.

Mô hình lưu ở `.cache/huggingface/`. Sau lần tải đầu, có thể chạy offline bằng `HF_HUB_OFFLINE=1 make app`/`make eval`. Nếu đổi model hoặc tham số chunking, chạy lại `make chunk index`; nếu thay PDF, chạy lại `make pipeline`. Không so sánh kết quả sinh từ các phiên bản index khác nhau.

Độ dài chunk tính bằng từ khác với giới hạn tokenizer của mô hình. E5-base giới hạn đầu vào theo cấu hình SentenceTransformer; phần cuối chunk dài có thể bị truncate khi embedding, còn toàn bộ text vẫn được giữ cho BM25 và context. Cần tính đến giới hạn này khi diễn giải so sánh chunking hoặc thay đổi model.

## Đánh giá tự động

```bash
make test
make eval
# Chỉ kiểm tra lexical, không cần nạp mô hình embedding:
.venv/bin/python evaluate.py --modes bm25_only --output data/eval/bm25_results.csv
```

`data/eval/eval_questions.csv` có đúng **15 câu**: 5 keyword, 5 paraphrase, 5 synthesis; cả 5 câu synthesis có căn cứ ở nhiều trang/chunk. Hai câu ngoài phạm vi nằm riêng ở `data/eval/out_of_scope.csv`. Ground truth được gán theo **trang PDF từ 1**, không dùng số trang in. Đây là nhãn ban đầu để người làm báo cáo duyệt lại, không phải benchmark đã được thẩm định độc lập.

Để tính metric chuẩn ở mức chunk, script ánh xạ mỗi ground truth `doc_id + relevant_pages` sang tất cả chunk chạm một trang liên quan trong từng strategy:

```text
R = tập chunk có ít nhất một trang liên quan trong đúng tài liệu
H = tập ID truy xuất ở Top-3
Precision@3 = |H ∩ R| / 3
Recall@3    = |H ∩ R| / |R|
```

Tìm kiếm trên **cả hai sách**, không dùng `doc_id` gold làm bộ lọc. Chạy 15 × 2 strategies × 3 modes = **90 lượt**. Số chunk liên quan khác nhau giữa hai chiến lược, nên mẫu số Recall cũng khác; nhãn trang có thể tính cả chunk chỉ chạm một phần nội dung trang. Cần xem đoạn trích/case study khi diễn giải, nhất là câu tổng hợp. Không tự chấm tính đúng của đáp án bằng metric retrieval.

Đầu ra của `make eval`:

- `data/eval/results.csv`: từng câu, P@3/R@3, ID và trang truy xuất, latency sau warm-up, tên model.
- `results_summary.csv`: macro average theo strategy/mode.
- `results_gate.csv`: kiểm tra 2 câu ngoài phạm vi × 6 cấu hình; không gọi API và không phát sinh phí LLM.
- `results_cases.md`: tự chọn kết quả tốt nhất, yếu nhất và chênh lệch hai chiến lược từ số liệu thật, kèm các trích đoạn. Nếu không có thất bại hoàn toàn, không tự tạo một trường hợp thất bại.

Đánh giá generation bổ sung: chọn provider thật, chạy 15 câu, chấm thủ công **Đúng / Đầy đủ / Bám nguồn / Citation**; đánh dấu câu tổng hợp bị thiếu căn cứ dù retrieval có hit. Thử cả hai câu ngoài phạm vi và xem `results_gate.csv` trước khi quyết định ngưỡng.

### Kết quả chạy ban đầu

Đã chạy 90 lượt bằng E5-base trên hai PDF có sẵn; số liệu dưới đây lấy từ `results_summary.csv`, không phải số liệu giả định:

| Chunking | Retrieval | P@3 | R@3 |
| --- | --- | ---: | ---: |
| fixed | hybrid_rrf | 0,6222 | 0,3484 |
| fixed | bm25_only | 0,5111 | 0,2794 |
| fixed | faiss_only | 0,6000 | 0,3262 |
| recursive | hybrid_rrf | 0,4889 | 0,2437 |
| recursive | bm25_only | 0,4222 | 0,1788 |
| recursive | faiss_only | 0,4667 | 0,2121 |

Fixed tốt hơn recursive ở lần chạy này; không kết luận recursive luôn tốt hơn chỉ vì có cấu trúc. Với hai câu ngoài phạm vi, hybrid/FAISS từ chối 8/8 lượt, BM25 từ chối 0/4 lượt. Điểm lexical có thể cao chỉ do trùng từ phổ biến, nên gate BM25 chưa đáng tin cậy để chứng minh đủ căn cứ; prompt LLM vẫn phải kiểm tra nội dung. Hai câu ngoài phạm vi chưa đủ đánh giá tỷ lệ từ chối tổng quát. API của bốn provider được kiểm tra bằng transport mô phỏng; chưa có kiểm chứng generation với API key thật.

## Kiểm thử và tham chiếu API

Test bao gồm số trang/offset của hai PDF thật, footer, provenance qua trang, overlap, ranh giới chương/câu, ID index và dữ liệu hỏng, lọc trước Top-N, công thức RRF, gate, citation và công thức metric. Encoder nhỏ trong test chỉ dùng để kiểm tra thuật toán/index, không dùng trong chạy demo hay tính kết quả đánh giá.

API triển khai được đối chiếu với [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), [Gemini generateContent](https://ai.google.dev/api/generate-content) và [Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx).
