from typing import Any, Dict, List

import numpy as np

from indexer import FaissCosineImageIndexer


def get_query_stage_output() -> Dict[str, Any]:
    """
    PREVIOUS STAGE INTEGRATION POINT
    --------------------------------
    Replace this function with your real query-image feature extraction step.

    This function must return:
        {
            "query_vector": np.ndarray
        }

    Example real flow:
        1. user uploads an image
        2. feature extractor converts image -> vector
        3. that vector is returned here
    """

    # -------------------------------------------------------------
    # EXAMPLE ONLY
    # Replace this fake vector with the real query vector.
    # -------------------------------------------------------------
    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return {
        "query_vector": query_vector
    }


def send_results_to_next_stage(results: List[Dict[str, Any]]) -> None:
    """
    NEXT STAGE INTEGRATION POINT
    ----------------------------
    Replace this with your real backend / API / UI handling.

    Example next steps:
        - return JSON response in Flask/FastAPI
        - render search results in the frontend
        - map image paths to public URLs
        - log analytics
    """
    print("Top results:")
    for item in results:
        print(
            f"{item['rank']}. {item['image_path']} | "
            f"score={item['score']:.4f} | "
            f"image_id={item['image_id']} | "
            f"category={item.get('category', 'N/A')}"
        )


def main() -> None:
    """
    SEARCH FLOW
    -----------
    1. Load the saved FAISS index
    2. Get the query vector from the previous stage
    3. Search top-k most similar items
    4. Send results to the next stage
    """

    # IMPORTANT:
    # embedding_dim must match the dimension used when the dataset index was built.
    indexer = FaissCosineImageIndexer(
        embedding_dim=3,
        index_path="data/image_index.faiss",
        metadata_path="data/metadata.pkl",
        config_path="data/index_config.json",
    )

    # Load saved index artifacts
    indexer.load()

    # Get query vector from previous stage
    query_stage_output = get_query_stage_output()

    # Search top-k similar images
    results = indexer.search_from_query_stage_output(
        query_stage_output,
        top_k=5,
    )

    # Pass ranked results to the next stage
    send_results_to_next_stage(results)


if __name__ == "__main__":
    main()
