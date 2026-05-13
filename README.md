# Visual Product Search Engine

Hybrid text search over the catalog: **query normalization** → **fuzzy (RapidFuzz)** → **semantic (Sentence-Transformers + FAISS)** → **score merge**.

## Layout

| Path | Role |
|------|------|
| `app/config.py` | Paths, model name, thresholds |
| `app/main.py` | Flask API + UI |
| `app/search/` | Query normalization + `SearchService` |
| `app/retrieval/engine.py` | Embeddings, FAISS, fuzzy, semantic, hybrid |
| `app/indexing/build_index.py` | Build FAISS + metadata from `data/products.json` |
| `app/indexing/image_documents.py` | Scan `images/`, write `processed/rag/documents.json`, export products |
| `data/products.json` | Product records for search |
| `data/vectors/` | FAISS index + metadata mirror |

## Setup

```bash
pip install -r requirements.txt
python -m app.indexing.build_index
python -m app.main
```

Windows: run `run.bat` (installs deps, starts `python -m app.main`).

## Endpoints

- `GET /` — Web UI
- `GET /health` — Search readiness
- `GET /search?q=...&top_k=10` — JSON search
- `POST /search` — JSON body `{ "q": "...", "top_k": 10 }`
- `GET /catalog-images` — Items with `image_url` for the UI image matcher
- `POST /reindex` — Rebuild `documents.json` from `images/`, refresh `data/products.json`, rebuild FAISS

## Images

Place files under `images/<category>/...` (see `POST /reindex`).
