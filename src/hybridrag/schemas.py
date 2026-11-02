"""Metadata trang và chunk; pages ghi toàn bộ trang chứa nội dung chunk."""
import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

Strategy = Literal["fixed", "recursive"]
BlockType = Literal["content", "key_terms", "exercises", "attribution"]


class TextBlock(BaseModel):
    text: str = Field(min_length=1)
    section: str
    block_type: BlockType = "content"


class PageRecord(BaseModel):
    doc_id: str
    source: str
    page: int = Field(ge=1)
    page_label: str
    section: str
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: int = Field(ge=0)
    text: str = Field(min_length=1)
    source: str
    doc_id: str
    page: int = Field(ge=1)
    page_label: str
    section: str
    strategy: Strategy
    block_type: BlockType = "content"
    pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pages(self):
        if not self.text.strip():
            raise ValueError("Chunk không được rỗng")
        if not self.pages:
            self.pages = [self.page]
        if self.pages != sorted(set(self.pages)) or self.pages[0] != self.page:
            raise ValueError("pages phải tăng dần và bắt đầu tại page")
        return self


T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, schema: type[T]) -> list[T]:
    return [schema.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
