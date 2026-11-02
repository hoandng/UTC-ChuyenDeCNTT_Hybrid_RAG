PYTHON ?= .venv/bin/python

.PHONY: setup ingest chunk index pipeline app eval test
setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --allow-existing .venv && \
		uv pip install --python $(PYTHON) torch --index-url https://download.pytorch.org/whl/cpu && \
		uv pip install --python $(PYTHON) -r requirements.txt; \
	else \
		python3 -m venv .venv && \
		$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi
ingest:
	$(PYTHON) -m hybridrag.ingest
chunk:
	$(PYTHON) -m hybridrag.chunking
index:
	$(PYTHON) -m hybridrag.indexer
pipeline: ingest chunk index
app:
	$(PYTHON) -m streamlit run app.py
eval:
	$(PYTHON) evaluate.py
test:
	$(PYTHON) -m pytest -q
