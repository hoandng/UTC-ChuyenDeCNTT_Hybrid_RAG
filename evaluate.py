"""Đánh giá retrieval thật; ground truth là trang PDF, ánh xạ sang chunk mỗi strategy."""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from hybridrag.config import ROOT, load_config
from hybridrag.generator import generate_answer
from hybridrag.indexer import Encoder
from hybridrag.retrieval import MODES, Retriever


def relevant_ids(chunks, doc_id: str, pages: set[int]) -> set[int]:
    return {c.chunk_id for c in chunks if c.doc_id == doc_id and pages.intersection(c.pages)}


def precision_recall(retrieved: list[int], relevant: set[int], k: int = 3) -> tuple[float, float]:
    if not relevant:
        raise ValueError("Ground truth không ánh xạ được tới chunk nào")
    found = len(set(retrieved[:k]).intersection(relevant))
    return found / k, found / len(relevant)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_report(rows: list[dict], details: dict) -> str:
    best = max(rows, key=lambda r: (r["precision_at_3"], r["recall_at_3"]))
    worst = min(rows, key=lambda r: (r["precision_at_3"], r["recall_at_3"]))
    pairs = []
    for row in rows:
        other = next((r for r in rows if r["id"] == row["id"] and r["mode"] == row["mode"] and r["strategy"] != row["strategy"]), None)
        if other:
            pairs.append((abs(row["precision_at_3"] - other["precision_at_3"]) + abs(row["recall_at_3"] - other["recall_at_3"]), row, other))
    lines = ["# Các trường hợp lấy từ kết quả chạy thật", "",
             "Ground truth gán ở mức trang PDF; cần duyệt lại đoạn trích để đánh giá độ đầy đủ của đáp án. Không có điểm chất lượng LLM tự bịa.", ""]
    cases = [("Kết quả tốt nhất", best), ("Kết quả yếu nhất / chưa đầy đủ", worst)]
    if pairs:
        delta, a, b = max(pairs, key=lambda p: p[0])
        lines += [f"Chênh lệch chunking lớn nhất: {a['id']} / {a['mode']}, Δ(P@3 + R@3)={delta:.4f}.",
                  f"- {a['strategy']}: P@3={a['precision_at_3']:.4f}, R@3={a['recall_at_3']:.4f}",
                  f"- {b['strategy']}: P@3={b['precision_at_3']:.4f}, R@3={b['recall_at_3']:.4f}", ""]
        cases += [("Đối sánh chunking A", a), ("Đối sánh chunking B", b)]
    for title, row in cases:
        key = (row["id"], row["strategy"], row["mode"])
        lines += [f"## {title}", "", f"{row['id']} · {row['strategy']} · {row['mode']}",
                  row["question"], "", f"P@3={row['precision_at_3']:.4f}, R@3={row['recall_at_3']:.4f}", ""]
        for hit in details[key]:
            lines += [f"- Chunk {hit.chunk.chunk_id}, {hit.chunk.section}, PDF {hit.chunk.pages}: {hit.chunk.text[:400].replace(chr(10), ' ')}"]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/results.csv")
    args = parser.parse_args()
    config = load_config()
    with (ROOT / "data/eval/eval_questions.csv").open(encoding="utf-8", newline="") as stream:
        questions = list(csv.DictReader(stream))
    if len(questions) != 15 or Counter(q["category"] for q in questions) != {"keyword": 5, "paraphrase": 5, "synthesis": 5}:
        raise ValueError("Cần đúng 15 câu: 5 keyword / 5 paraphrase / 5 synthesis")
    with (ROOT / "data/eval/out_of_scope.csv").open(encoding="utf-8", newline="") as stream:
        outside = list(csv.DictReader(stream))
    rows, gate_rows, details, encoders = [], [], {}, {}
    for strategy in ("fixed", "recursive"):
        retriever = Retriever(ROOT / "indexes" / strategy, config.retrieval, device=config.embedding.device)
        if any(mode != "bm25_only" for mode in args.modes):
            model = retriever.manifest["model"]
            if model not in encoders:
                embedding_config = config.embedding.model_copy(update={"model": model})
                encoders[model] = Encoder(embedding_config)
            retriever.encoder = encoders[model]
        # Warm-up tách khỏi đo truy vấn, không tính thời gian tải model.
        retriever.search("database", args.modes[0])
        for mode in args.modes:
            for question in questions:
                gold_pages = {int(p) for p in question["relevant_pages"].split(";")}
                gold = relevant_ids(retriever.chunks, question["doc_id"], gold_pages)
                start = time.perf_counter()
                # Tìm trên cả hai sách: doc_id chỉ dùng chấm điểm, không tiết lộ gold cho retrieval.
                hits = retriever.search(question["question"], mode, top_k=3)
                elapsed = (time.perf_counter() - start) * 1000
                ids = [h.chunk.chunk_id for h in hits]
                precision, recall = precision_recall(ids, gold)
                rows.append(dict(id=question["id"], category=question["category"], question=question["question"],
                                 strategy=strategy, mode=mode, model=retriever.manifest["model"],
                                 precision_at_3=precision, recall_at_3=recall, latency_ms=elapsed,
                                 relevant_chunks=len(gold), retrieved_ids=json.dumps(ids),
                                 retrieved_pages=json.dumps([{"doc_id": h.chunk.doc_id, "pages": h.chunk.pages} for h in hits])))
                details[(question["id"], strategy, mode)] = hits
            for question in outside:
                hits = retriever.search(question["question"], mode)
                answer = generate_answer(question["question"], hits, config.retrieval, dry_run=True)
                gate_rows.append(dict(id=question["id"], strategy=strategy, mode=mode, refused=answer.refused,
                                      max_cosine=max((h.dense_score for h in hits if h.dense_score is not None), default=""),
                                      max_bm25=max((h.bm25_score for h in hits if h.bm25_score is not None), default="")))
    write_csv(args.output, rows)
    write_csv(args.output.with_name(args.output.stem + "_gate.csv"), gate_rows)
    args.output.with_name(args.output.stem + "_cases.md").write_text(case_report(rows, details), encoding="utf-8")
    summary = []
    for strategy in ("fixed", "recursive"):
        for mode in args.modes:
            selected = [r for r in rows if r["strategy"] == strategy and r["mode"] == mode]
            summary.append(dict(strategy=strategy, mode=mode, questions=len(selected),
                                precision_at_3=sum(r["precision_at_3"] for r in selected) / len(selected),
                                recall_at_3=sum(r["recall_at_3"] for r in selected) / len(selected),
                                latency_ms=sum(r["latency_ms"] for r in selected) / len(selected)))
    write_csv(args.output.with_name(args.output.stem + "_summary.csv"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Đã ghi {len(rows)} lượt đánh giá: {args.output}")


if __name__ == "__main__":
    main()
