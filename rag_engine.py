from __future__ import annotations

from pathlib import Path
from typing import Any

from generator import RagDocument, RagGenerator
from indexer import RagIndexer
from retriever import RagRetriever


class ProductRagEngine:
    """Orchestrates RAG generator, indexer, and retriever."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.generator = RagGenerator()
        self.indexer = RagIndexer(store_path=store_path)
        self.retriever = RagRetriever()
        self.documents: list[RagDocument] = []

    def rebuild_from_images_folder(self, images_dir: Path) -> dict[str, Any]:
        documents = self.generator.build_from_images_folder(images_dir)
        self.indexer.save(documents)
        self.documents = documents
        self.retriever.set_documents(documents)
        return {"doc_count": len(documents), "store_path": str(self.store_path)}

    def load(self) -> None:
        self.documents = self.indexer.load()
        self.retriever.set_documents(self.documents)

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self.retriever.answer(query=query, top_k=top_k)
