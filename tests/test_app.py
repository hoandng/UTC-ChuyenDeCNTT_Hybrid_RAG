import shutil

from streamlit.testing.v1 import AppTest

from hybridrag.indexer import build_index
from hybridrag.schemas import Chunk
from test_pipeline import ToyEncoder


def test_streamlit_query_and_rerun_preserve_results(tmp_path, monkeypatch):
    import hybridrag.config as configuration

    root = configuration.ROOT
    (tmp_path / "configs").mkdir()
    shutil.copy2(root / "configs/default.yaml", tmp_path / "configs/default.yaml")
    chunks = [Chunk(chunk_id=i, text="database information general content " * 5 if i else "candidate key uniquely identifies a record",
                    source="book.pdf", doc_id="db_design_en", page=37, page_label="37", section="Chapter 8 > Keys", strategy="recursive") for i in range(20)]
    build_index(chunks, tmp_path / "indexes/recursive", ToyEncoder(), "test-only")
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "dry-run")
    app = AppTest.from_file(str(root / "app.py")).run(timeout=20)
    assert not app.exception
    app.selectbox[1].select("bm25_only").run()
    app.text_area[0].set_value("candidate key")
    app.button[-1].click().run(timeout=20)
    assert not app.exception
    assert any("Dry-run" in element.value for element in app.markdown)
    assert len(app.metric) == 4
    app.radio[1].set_value("Trước câu trả lời").run()
    assert not app.exception
    assert any("Dry-run" in element.value for element in app.markdown)
