from app.retrieval.engine import HybridSearch


class SearchService:
    def __init__(self):
        self.hybrid_search = HybridSearch()

    def search(self, query: str, top_k: int = 10):
        if not query or not query.strip():
            return []

        return self.hybrid_search.search(query, top_k=top_k)


_service: SearchService | None = None


def get_search_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


def reset_search_service() -> None:
    """Drop singleton so the next request reloads products + FAISS (after reindex)."""
    global _service
    _service = None
