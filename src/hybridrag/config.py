"""Cấu hình YAML và biến môi trường, độc lập với thư mục chạy lệnh."""
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


class Document(BaseModel):
    doc_id: str
    title: str
    filename: str
    expected_pages: int = Field(gt=0)
    page_offset: int = 0


class ChunkingConfig(BaseModel):
    fixed_size: int = Field(default=256, gt=0)
    fixed_overlap: int = Field(default=35, ge=0)
    recursive_target: int = Field(default=300, gt=0)
    recursive_max: int = Field(default=450, gt=0)

    @model_validator(mode="after")
    def validate_sizes(self):
        if self.fixed_overlap >= self.fixed_size:
            raise ValueError("fixed_overlap phải nhỏ hơn fixed_size")
        if self.recursive_target > self.recursive_max:
            raise ValueError("recursive_target phải <= recursive_max")
        return self


class EmbeddingConfig(BaseModel):
    model: str = "intfloat/multilingual-e5-base"
    device: str = "cuda"
    batch_size: int = Field(default=8, gt=0)


class RetrievalConfig(BaseModel):
    candidates: int = Field(default=8, gt=0)
    top_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    min_dense_similarity: float = Field(default=0.78, ge=-1, le=1)
    min_bm25_score: float = Field(default=1.0, ge=0)


class Settings(BaseModel):
    documents: list[Document]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def load_config(path: Path | None = None) -> Settings:
    import os

    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.model_validate(yaml.safe_load((path or ROOT / "configs/default.yaml").read_text()))
    settings.embedding.model = os.getenv("EMBEDDING_MODEL", settings.embedding.model)
    settings.embedding.device = os.getenv("EMBEDDING_DEVICE", settings.embedding.device)
    return settings
