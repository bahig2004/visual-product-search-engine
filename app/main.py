from __future__ import annotations

import json
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from app.config import BASE_DIR, IMAGES_DIR, PRODUCTS_FILE, RAG_DOCUMENTS_FILE
from app.indexing.build_index import main as rebuild_text_index
from app.indexing.image_documents import ImageCatalogStore, documents_to_products
from app.search.search_service import get_search_service, reset_search_service

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def _to_public_image_url(image_path: str) -> str | None:
    image_abs = (BASE_DIR / image_path).resolve()
    try:
        relative = image_abs.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        return None
    return url_for("serve_image", image_path=relative.as_posix())


def _build_context(results: list[dict]) -> str:
    chunks: list[str] = []
    for item in results:
        tags = item.get("tags") or []
        tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        chunks.append(
            f"Product: {item.get('name', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Tags: {tag_str}\n"
        )
    return "\n---\n".join(chunks)


def _format_hybrid_results(raw: list[dict], top_k: int) -> tuple[list[dict], str]:
    out: list[dict] = []
    for rank, item in enumerate(raw, start=1):
        row = dict(item)
        row["rank"] = rank
        row["score"] = float(row.get("final_score", row.get("semantic_score", 0.0)))
        image_path = row.get("image_path")
        row["image_url"] = _to_public_image_url(image_path) if isinstance(image_path, str) else None
        if "filename" not in row and row.get("name"):
            row.setdefault("filename", str(row.get("name", "")))
        out.append(row)

    if not out:
        return [], "No matching products found for this query."
    preview = _build_context(out[: min(5, len(out))])
    answer = f"Hybrid (semantic + fuzzy) matches for your query.\n\n{preview[:2000]}"
    return out[:top_k], answer


@app.get("/")
def home() -> Any:
    query = request.args.get("q", "")
    results: list = []
    if query:
        results = get_search_service().search(query, top_k=10)
    return render_template("index.html", query=query, results=results)


@app.get("/images/<path:image_path>")
def serve_image(image_path: str) -> Any:
    return send_from_directory(IMAGES_DIR, image_path)


@app.get("/health")
def health() -> Any:
    try:
        get_search_service()
    except Exception as exc:
        return jsonify({"status": "not_ready", "error": str(exc)}), 503
    return jsonify({"status": "ok", "search": {"mode": "hybrid", "products_file": str(PRODUCTS_FILE)}})


@app.post("/reindex")
def reindex() -> Any:
    """Rebuild documents from images, refresh products.json, rebuild FAISS, reset search cache."""
    try:
        catalog = ImageCatalogStore(RAG_DOCUMENTS_FILE)
        rag_summary = catalog.rebuild_from_images_folder(IMAGES_DIR)
        products = documents_to_products(catalog.documents)
        PRODUCTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PRODUCTS_FILE, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2)
        reset_search_service()
        rebuild_text_index()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"message": "catalog and text index rebuilt", "rag": rag_summary}), 201


@app.get("/search")
def search_get() -> Any:
    query = (request.args.get("q") or "").strip()
    top_k_raw = request.args.get("top_k", 10)
    try:
        top_k = int(top_k_raw)
        if top_k <= 0 or top_k > 200:
            raise ValueError("top_k out of range")
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be a positive integer (max 200)"}), 400

    if not query:
        return jsonify({"query": query, "count": 0, "results": []})

    try:
        raw = get_search_service().search(query, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": f"search failed: {exc}"}), 500

    results, _answer = _format_hybrid_results(raw, top_k)
    return jsonify({"query": query, "count": len(results), "results": results, "mode": "hybrid", "top_k": top_k})


@app.post("/search")
def search_post() -> Any:
    payload = request.get_json(silent=True) or {}
    query = (payload.get("q") or payload.get("query") or "").strip()
    top_k_raw = payload.get("top_k", 10)
    if not query:
        return jsonify({"error": "Missing search query (q)."}), 400
    try:
        top_k = int(top_k_raw)
        if top_k <= 0 or top_k > 200:
            raise ValueError("top_k out of range")
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be a positive integer (max 200)"}), 400

    try:
        raw = get_search_service().search(query, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": f"search failed: {exc}"}), 500

    results, answer = _format_hybrid_results(raw, top_k)
    return jsonify(
        {
            "query": query,
            "count": len(results),
            "results": results,
            "answer": answer,
            "mode": "hybrid",
            "top_k": top_k,
        }
    )


@app.get("/catalog-images")
def catalog_images() -> Any:
    try:
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            products = json.load(f)
    except Exception as exc:
        return jsonify({"error": f"failed to load catalog: {exc}"}), 500

    items: list[dict[str, Any]] = []
    for row in products:
        image_path = row.get("image_path")
        image_url = _to_public_image_url(image_path) if isinstance(image_path, str) else None
        if not image_url:
            continue
        items.append(
            {
                "doc_id": str(row.get("id", "")),
                "text": row.get("description") or row.get("name") or "",
                "category": row.get("category"),
                "colors": row.get("tags") or [],
                "image_colors": row.get("tags") or [],
                "filename": row.get("filename") or row.get("name"),
                "image_path": image_path,
                "image_url": image_url,
            }
        )
    return jsonify({"count": len(items), "items": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
