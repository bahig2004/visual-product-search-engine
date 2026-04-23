from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from .exceptions import ConfigurationError, IndexNotBuiltError, IndexPersistenceError
from .utils import (
    ensure_parent_dir,
    get_logger,
    read_json,
    validate_dataset_vectors,
    validate_metadata,
    validate_query_vector,
    write_json,
)


class FaissCosineImageIndexer:
    """
    Exact cosine-similarity image indexer backed by FAISS IndexFlatIP.

    -------------------------------------------------------------------
    PROJECT PIPELINE POSITION
    -------------------------------------------------------------------
    This class sits in the middle of the image retrieval pipeline.

    PREVIOUS STAGE:
        Feature extraction stage.
        That stage must produce:
            1) dataset vectors   -> shape: (num_images, embedding_dim)
            2) dataset metadata  -> list of dicts aligned with the vectors

            3) query vector      -> shape: (embedding_dim,) or (1, embedding_dim)

    CURRENT STAGE:
        This class:
            - validates vectors and metadata
            - normalizes vectors for cosine similarity
            - builds a FAISS index
            - saves/loads index artifacts
            - searches top-k similar items

    NEXT STAGE:
        System integration / backend / API / UI stage.
        That stage will:
            - call .search(...) with a query vector
            - receive ranked results with metadata
            - return results as JSON / API response / UI cards

    -------------------------------------------------------------------
    COSINE LOGIC
    -------------------------------------------------------------------
    Cosine similarity is implemented by:
        1) casting vectors to float32
        2) applying L2 normalization
        3) using FAISS inner product index (IndexFlatIP)

    For normalized vectors:
        inner product == cosine similarity
    """

    METRIC_NAME = "cosine"
    INDEX_TYPE = "IndexFlatIP"

    def __init__(
        self,
        embedding_dim: int,
        index_path: str,
        metadata_path: str,
        config_path: str,
        logger=None,
    ) -> None:
        """
        Parameters
        ----------
        embedding_dim:
            Dimension of the feature vectors produced by the feature extractor.
            Example: 512, 1024, 2048, etc.

        index_path:
            Path where the FAISS index file will be saved.
            This is one of the handoff artifacts between offline indexing
            and online query search.

        metadata_path:
            Path where aligned metadata will be saved.

        config_path:
            Path where config JSON will be saved.

        logger:
            Optional custom logger.

        INTEGRATION NOTE:
        The system integrator should keep these artifact paths stable so the
        online search service can load the same index built offline.
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")

        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        self.logger = logger or get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # INTEGRATION BRIDGE: DATASET SIDE
    # ------------------------------------------------------------------
    def build_index_from_extraction_output(self, extraction_output: Dict[str, Any]) -> None:
        """
        Build the index directly from the output of the previous stage.

        EXPECTED PREVIOUS STAGE OUTPUT FORMAT:
            {
                "vectors": np.ndarray of shape (num_images, embedding_dim),
                "metadata": [
                    {"image_id": "...", "image_path": "...", ...},
                    ...
                ]
            }

        WHY THIS METHOD EXISTS:
        This makes the handoff from feature extraction -> retrieval explicit.
        The feature extraction team can return one dictionary, and the
        retrieval/indexing stage can consume it directly.

        NEXT STEP AFTER THIS:
            call .save()

        Example:
            extraction_output = feature_extractor.extract_dataset(...)
            indexer.build_index_from_extraction_output(extraction_output)
            indexer.save()
        """
        if not isinstance(extraction_output, dict):
            raise ValueError(
                "extraction_output must be a dictionary with keys: 'vectors' and 'metadata'."
            )

        if "vectors" not in extraction_output:
            raise ValueError(
                "extraction_output is missing required key: 'vectors'.")
        if "metadata" not in extraction_output:
            raise ValueError(
                "extraction_output is missing required key: 'metadata'.")

        vectors = extraction_output["vectors"]
        metadata = extraction_output["metadata"]

        self.build_index(vectors, metadata)

    def build_index(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Build the FAISS index from dataset vectors + aligned metadata.

        PREVIOUS STAGE INPUTS:
            vectors:
                produced by the feature extraction stage
            metadata:
                produced during dataset preparation / extraction stage

        CURRENT STAGE ACTIONS:
            - validate vectors
            - validate metadata alignment
            - normalize vectors
            - create FAISS IndexFlatIP
            - add vectors to index
            - keep metadata in memory

        NEXT STAGE OUTPUT:
            After calling .save(), the next stage can load these artifacts:
                - image_index.faiss
                - metadata.pkl
                - index_config.json
        """
        self.logger.info("Starting FAISS index build.")

        validated_vectors = validate_dataset_vectors(
            vectors, self.embedding_dim)
        validated_metadata = validate_metadata(
            metadata, validated_vectors.shape[0])

        self.logger.info(
            "Validated %s vectors with embedding_dim=%s.",
            validated_vectors.shape[0],
            self.embedding_dim,
        )

        # IMPORTANT:
        # This normalization step is required for cosine similarity with FAISS.
        # Do not remove it unless you intentionally change the metric.
        faiss.normalize_L2(validated_vectors)
        self.logger.info("Vector normalization complete.")

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(validated_vectors)

        self.index = index
        self.metadata = validated_metadata

        self.logger.info(
            "Index build complete. Stored vectors=%s.",
            self.index.ntotal,
        )

    def save(self) -> None:
        """
        Save the built index artifacts to disk.

        OFFLINE / INDEXING SIDE:
            This is usually called once after dataset indexing.

        ONLINE / SEARCH SIDE:
            The backend later loads these same files using .load().

        ARTIFACTS SAVED:
            - FAISS index
            - metadata pickle
            - config JSON
        """
        self._assert_ready()

        try:
            ensure_parent_dir(self.index_path)
            ensure_parent_dir(self.metadata_path)
            ensure_parent_dir(self.config_path)

            faiss.write_index(self.index, str(self.index_path))

            with self.metadata_path.open("wb") as file_obj:
                pickle.dump(self.metadata, file_obj)

            config = {
                "metric": self.METRIC_NAME,
                "embedding_dim": self.embedding_dim,
                "vector_count": int(self.index.ntotal),
                "normalized": True,
                "index_type": self.INDEX_TYPE,
            }
            write_json(self.config_path, config)

            self.logger.info("Index, metadata, and config saved successfully.")
        except Exception as exc:
            raise IndexPersistenceError(
                f"Failed to save index artifacts: {exc}") from exc

    def load(self) -> None:
        """
        Load previously saved index artifacts from disk.

        USE CASE:
            This is the normal entry point for the online search service.

        INTEGRATION NOTE:
            In a backend, this should usually happen:
                - once at service startup, or
                - once when the search component initializes

            It should NOT usually rebuild the whole dataset index on every request.
        """
        missing_files = [
            str(path)
            for path in (self.index_path, self.metadata_path, self.config_path)
            if not path.exists()
        ]
        if missing_files:
            raise IndexPersistenceError(
                f"Cannot load index. Missing file(s): {', '.join(missing_files)}."
            )

        try:
            index = faiss.read_index(str(self.index_path))

            with self.metadata_path.open("rb") as file_obj:
                metadata = pickle.load(file_obj)

            config = read_json(self.config_path)
            self._validate_loaded_config(config, index, metadata)

            self.index = index
            self.metadata = metadata

            self.logger.info(
                "Index loaded successfully. vector_count=%s, embedding_dim=%s.",
                self.index.ntotal,
                self.embedding_dim,
            )
        except (IndexPersistenceError, ConfigurationError):
            raise
        except Exception as exc:
            raise IndexPersistenceError(
                f"Failed to load index artifacts: {exc}") from exc

    # ------------------------------------------------------------------
    # INTEGRATION BRIDGE: QUERY SIDE
    # ------------------------------------------------------------------
    def search_from_query_stage_output(
        self,
        query_stage_output: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search directly from the output of the query feature extraction stage.

        EXPECTED PREVIOUS STAGE OUTPUT FORMAT:
            {
                "query_vector": np.ndarray shape (embedding_dim,)
            }

        or, if your integrator prefers:
            {
                "vector": np.ndarray shape (embedding_dim,)
            }

        WHY THIS METHOD EXISTS:
        This makes the handoff from:
            uploaded image -> query feature extractor -> retrieval search
        very explicit.

        NEXT STAGE AFTER THIS:
            The returned results can be:
                - returned as API JSON
                - shown in the UI
                - passed into a ranking/post-processing layer
        """
        if not isinstance(query_stage_output, dict):
            raise ValueError(
                "query_stage_output must be a dictionary containing 'query_vector' or 'vector'."
            )

        if "query_vector" in query_stage_output:
            query_vector = query_stage_output["query_vector"]
        elif "vector" in query_stage_output:
            query_vector = query_stage_output["vector"]
        else:
            raise ValueError(
                "query_stage_output must contain 'query_vector' or 'vector'."
            )

        return self.search(query_vector, top_k=top_k)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search top-k most similar indexed images for the given query vector.

        PREVIOUS STAGE INPUT:
            query_vector:
                produced by the query image feature extractor

        CURRENT STAGE ACTIONS:
            - validate query vector
            - normalize query vector
            - search FAISS index
            - attach metadata to each matched row

        NEXT STAGE OUTPUT:
            Returns a list of ranked dictionaries.
            This format is already easy to serialize as JSON in a backend.

        RESULT FORMAT:
            [
                {
                    "rank": 1,
                    "index": 23,
                    "score": 0.9821,
                    "metric": "cosine",
                    "image_id": "img_0024",
                    "image_path": "dataset/shoes/red_shoe_24.jpg",
                    "category": "shoes"
                },
                ...
            ]
        """
        self._assert_ready()

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self.logger.info("Starting search with top_k=%s.", top_k)

        query = validate_query_vector(query_vector, self.embedding_dim)

        # IMPORTANT:
        # Query must also be normalized for cosine similarity.
        # Stored vectors and query vector must follow the same normalization rule.
        faiss.normalize_L2(query)

        safe_top_k = min(top_k, self.get_index_size())
        scores, indices = self.index.search(query, safe_top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            result_item = {
                "rank": rank,
                "index": int(idx),
                "score": float(score),
                "metric": self.METRIC_NAME,
            }

            # Metadata is attached here.
            # This is the point where vector-space results become real image results.
            # The next stage (API/UI) usually uses image_path, image_id, category, etc.
            result_item.update(self.metadata[int(idx)])
            results.append(result_item)

        self.logger.info(
            "Search complete. Returned %s result(s).", len(results))
        return results

    def is_loaded(self) -> bool:
        """Return True if the index is ready for search."""
        return self.index is not None and len(self.metadata) == int(self.index.ntotal)

    def get_index_size(self) -> int:
        """Return number of indexed vectors."""
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def get_runtime_summary(self) -> Dict[str, Any]:
        """
        Small helper for system integration / health checks.

        This can be used by:
            - backend startup logs
            - admin/debug endpoints
            - service health checks
        """
        return {
            "is_loaded": self.is_loaded(),
            "embedding_dim": self.embedding_dim,
            "index_size": self.get_index_size(),
            "metric": self.METRIC_NAME,
            "index_type": self.INDEX_TYPE,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "config_path": str(self.config_path),
        }

    def _assert_ready(self) -> None:
        if not self.is_loaded():
            raise IndexNotBuiltError(
                "FAISS index is not built or loaded. Build or load the index first."
            )

    def _validate_loaded_config(
        self,
        config: Dict[str, Any],
        index,
        metadata: List[Dict[str, Any]],
    ) -> None:
        required_keys = {
            "metric",
            "embedding_dim",
            "vector_count",
            "normalized",
            "index_type",
        }
        missing = required_keys - set(config.keys())

        if missing:
            raise ConfigurationError(
                f"Config file is missing required key(s): {', '.join(sorted(missing))}."
            )

        if config["metric"] != self.METRIC_NAME:
            raise ConfigurationError(
                f"Unsupported metric in config: {config['metric']}."
            )

        if int(config["embedding_dim"]) != self.embedding_dim:
            raise ConfigurationError(
                f"Embedding dimension mismatch. Config has {config['embedding_dim']}, "
                f"but indexer expects {self.embedding_dim}."
            )

        if config["index_type"] != self.INDEX_TYPE:
            raise ConfigurationError(
                f"Unsupported index type in config: {config['index_type']}."
            )

        if bool(config["normalized"]) is not True:
            raise ConfigurationError(
                "Config indicates vectors were not normalized.")

        if int(config["vector_count"]) != int(index.ntotal):
            raise ConfigurationError(
                f"Vector count mismatch between config ({config['vector_count']}) "
                f"and FAISS index ({index.ntotal})."
            )

        validated_metadata = validate_metadata(metadata, int(index.ntotal))
        if len(validated_metadata) != int(index.ntotal):
            raise ConfigurationError(
                "Loaded metadata is inconsistent with FAISS index.")
