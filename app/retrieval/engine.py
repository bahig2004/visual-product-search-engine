"""FAISS semantic search, fuzzy match, and hybrid merge (single module)."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_FILE,
    FUZZY_THRESHOLD,
    METADATA_FILE,
    PRODUCTS_FILE,
    TOP_K,
)
from app.search.query_normalizer import normalize_query


class FaissManager:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = None

    def build(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        if self.index is None:
            self.load()

        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        return scores[0], indices[0]


class TextEmbeddingPipeline:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        )
        return np.array(embeddings).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding).astype("float32")


def fuzzy_search(query: str, products: list[dict], threshold: int = 75) -> list[dict]:
    query = normalize_query(query)
    results = []

    for product in products:
        text = " ".join(
            [
                str(product.get("name", "")),
                str(product.get("category", "")),
                str(product.get("description", "")),
                " ".join(product.get("tags") or []),
            ]
        )

        text = normalize_query(text)
        score = fuzz.partial_ratio(query, text)

        if score >= threshold:
            item = product.copy()
            item["fuzzy_score"] = score
            results.append(item)

    results.sort(key=lambda x: x["fuzzy_score"], reverse=True)
    return results


class SemanticSearch:
    def __init__(self):
        self.embedder = TextEmbeddingPipeline()
        self.faiss_manager = FaissManager(FAISS_INDEX_FILE)
        self.faiss_manager.load()

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        query = normalize_query(query)

        query_embedding = self.embedder.encode_query(query)
        scores, indices = self.faiss_manager.search(query_embedding, top_k)

        results = []
        rank = 0
        for idx, score in zip(indices, scores):
            if int(idx) < 0 or int(idx) >= len(self.metadata):
                continue
            rank += 1
            product = self.metadata[int(idx)].copy()
            product["rank"] = rank
            product["semantic_score"] = float(score)
            results.append(product)

        return results


class HybridSearch:
    def __init__(self):
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            self.products = json.load(f)

        self.semantic = SemanticSearch()

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        semantic_results = self.semantic.search(query, top_k=top_k)
        fuzzy_results = fuzzy_search(query, self.products, threshold=FUZZY_THRESHOLD)

        merged = {}

        for item in semantic_results:
            product_id = str(item.get("id", item.get("image_id", item.get("name"))))
            item["final_score"] = item.get("semantic_score", 0) * 100
            merged[product_id] = item

        for item in fuzzy_results:
            product_id = str(item.get("id", item.get("image_id", item.get("name"))))

            if product_id in merged:
                merged[product_id]["final_score"] += item.get("fuzzy_score", 0)
                merged[product_id]["fuzzy_score"] = item.get("fuzzy_score", 0)
            else:
                item["semantic_score"] = 0
                item["final_score"] = item.get("fuzzy_score", 0)
                merged[product_id] = item

        results = list(merged.values())
        results.sort(key=lambda x: x["final_score"], reverse=True)

        return results[:top_k]
