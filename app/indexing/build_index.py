import json

from app.config import FAISS_INDEX_FILE, METADATA_FILE, PRODUCTS_FILE
from app.retrieval.engine import FaissManager, TextEmbeddingPipeline


def product_to_text(product: dict) -> str:
    return " ".join(
        [
            str(product.get("name", "")),
            str(product.get("category", "")),
            str(product.get("description", "")),
            " ".join(product.get("tags") or []),
        ]
    )


def main():
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)

    texts = [product_to_text(product) for product in products]

    embedder = TextEmbeddingPipeline()
    embeddings = embedder.encode_texts(texts)

    faiss_manager = FaissManager(FAISS_INDEX_FILE)
    faiss_manager.build(embeddings)
    faiss_manager.save()

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)

    print("Index built successfully")
    print(f"Products indexed: {len(products)}")
    print(f"Index saved to: {FAISS_INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
