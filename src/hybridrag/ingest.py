"""Đọc từng trang PDF, nhận diện chương/mục từ text và font."""
import re
import shutil
import statistics

import fitz

from .config import ROOT, Document, load_config
from .schemas import PageRecord, TextBlock, write_jsonl

FOOTER = re.compile(r"^(?:\d+|\d+\s*\|\s*Chapter\b.*|Chapter\b.*\|\s*\d+)$", re.I)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\u00ad", "").splitlines()]
    # Số đứng riêng giữa trang có thể là ô bảng; chỉ lọc footer có chữ ở đây.
    lines = [line for line in lines if not ("|" in line and FOOTER.fullmatch(line))]
    text = "\n".join(lines)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_document(path, document: Document) -> list[PageRecord]:
    records = []
    chapter, section, block_type = "Mở đầu", "Mở đầu", "content"
    with fitz.open(path) as pdf:
        if len(pdf) != document.expected_pages:
            raise ValueError(f"{path.name}: cần {document.expected_pages} trang, nhận {len(pdf)}")
        for number, page in enumerate(pdf, start=1):
            raw = page.get_text("dict", sort=True)["blocks"]
            sizes = [s["size"] for b in raw for line in b.get("lines", []) for s in line["spans"] if s["text"].strip()]
            body_size = statistics.median(sizes) if sizes else 12
            blocks = []
            page_text = clean_text(page.get_text())
            page_lines = page_text.splitlines()
            if page_lines and page_lines[-1] == str(number + document.page_offset):
                page_text = "\n".join(page_lines[:-1]).strip()
            # Các trang phân cách Phần có tiêu đề lớn bị PDF chia làm nhiều block.
            is_part_page = page_text.startswith("Phần ") and len(page_text.split()) < 30
            if is_part_page:
                chapter = section = " ".join(page_text.split())
                block_type = "content"
                blocks = [TextBlock(text=page_text, section=section)]
            for block in raw:
                if is_part_page:
                    break
                spans = [s for line in block.get("lines", []) for s in line["spans"]]
                text = clean_text("\n".join("".join(s["text"] for s in line["spans"]) for line in block.get("lines", [])))
                if not text:
                    continue
                flat = " ".join(text.split())
                if block["bbox"][1] >= page.rect.height * 0.90 and FOOTER.fullmatch(flat):
                    continue
                chapter_match = re.match(r"^Chapter\s+\d+\b", flat)
                ml_match = document.doc_id == "ml_yearning_vi" and re.match(r"^\d+\.\s+\D", flat) and max(s["size"] for s in spans) > body_size * 1.12
                if chapter_match or ml_match or (flat.startswith("Phần ") and len(flat) < 120):
                    chapter = flat if not ml_match else "Chương " + flat
                    section, block_type = chapter, "content"
                elif flat.lower() in {"key terms", "exercises", "attribution"}:
                    block_type = {"key terms": "key_terms", "exercises": "exercises", "attribution": "attribution"}[flat.lower()]
                    section = f"{chapter} > {flat}"
                elif len(flat.split()) <= 18 and spans and (
                    max(s["size"] for s in spans) > body_size * 1.12
                    # Font flags của PDF ML đánh dấu cả chữ thường là bold.
                    # Tên font đáng tin cậy hơn cho hai giáo trình này.
                    or (all("bold" in s["font"].lower() for s in spans if s["text"].strip())
                        and (document.doc_id != "ml_yearning_vi" or len(flat.split()) >= 5))
                ) and not flat.isupper():
                    section, block_type = f"{chapter} > {flat}", "content"
                blocks.append(TextBlock(text=text, section=section, block_type=block_type))
            records.append(PageRecord(doc_id=document.doc_id, source=document.filename, page=number,
                                      page_label=str(number + document.page_offset),
                                      section=blocks[0].section if blocks else section,
                                      text="\n\n".join(b.text for b in blocks), blocks=blocks))
    return records


def main() -> None:
    config = load_config()
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    for doc in config.documents:
        target = ROOT / "data/raw" / doc.filename
        if not target.exists():
            candidates = [ROOT / "books/processed" / doc.filename, ROOT / "books" / doc.filename]
            source = next((p for p in candidates if p.exists()), None)
            if source is None:
                raise FileNotFoundError(f"Không tìm thấy PDF {doc.filename} trong books/ hoặc data/raw/")
            shutil.copy2(source, target)
        pages = extract_document(target, doc)
        write_jsonl(ROOT / "data/interim" / f"{doc.doc_id}_pages.jsonl", pages)
        print(f"{doc.doc_id}: {len(pages)} trang, {sum(bool(p.text) for p in pages)} trang có text")


if __name__ == "__main__":
    main()
