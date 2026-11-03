import sys
from types import SimpleNamespace

import pytest
import torch

from hybridrag.config import EmbeddingConfig
from hybridrag.indexer import Encoder


def test_cuda_unavailable_reports_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="EMBEDDING_DEVICE=cpu"):
        Encoder(EmbeddingConfig(device="cuda"))


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_encoder_passes_selected_device_to_model(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device != "cpu")
    calls = []

    def model_factory(model, *, device):
        calls.append((model, device))
        return object()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=model_factory))
    config = EmbeddingConfig(device=device)
    Encoder(config)
    assert calls == [(config.model, device)]
