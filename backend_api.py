from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from exceptions import RetrievalError


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
INDEX_DIR = PROCESSED_DIR / "index"

FEATURES_HDF5 = FEATURES_DIR / "image_features.h5"
FEATURES_MAPPING = FEATURES_DIR / "feature_mapping.json"
INDEX_PATH = INDEX_DIR / "image_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
CONFIG_PATH = INDEX_DIR / "index_config.json"

class QueryFeatureExtractor:
    """Shared query-side feature extractor using the same embedding model."""

    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torchvision.models import ResNet50_Weights, resnet50
        from PIL import Image

        self._torch = torch
        self._image_module = Image
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract_bytes(self, image_bytes: bytes) -> np.ndarray:
        img = self._image_module.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self.model(img_tensor).view(1, -1)
        return features.cpu().numpy().astype(np.float32)

    def extract(self, image_path: Path) -> np.ndarray:
        img = self._image_module.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self.model(img_tensor).view(1, -1)
        return features.cpu().numpy().astype(np.float32)


class SearchBackendService:
    """Coordinates index loading/building and upload->search flow."""

    def __init__(self) -> None:
        from indexer import FaissCosineImageIndexer

        self.indexer = FaissCosineImageIndexer(
            embedding_dim=2048,
            index_path=str(INDEX_PATH),
            metadata_path=str(METADATA_PATH),
            config_path=str(CONFIG_PATH),
        )
        self.extractor = QueryFeatureExtractor()
        self._result_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._result_cache_size = int(os.getenv("RESULT_CACHE_SIZE", "128"))
        self._ensure_index_ready()

    def _ensure_index_ready(self) -> None:
        if INDEX_PATH.exists() and METADATA_PATH.exists() and CONFIG_PATH.exists():
            self.indexer.load()
            return
        self.rebuild_index()

    def rebuild_index(self) -> Dict[str, Any]:
        if FEATURES_HDF5.exists() and FEATURES_MAPPING.exists():
            vectors, metadata = self._load_index_inputs_from_features_files()
        else:
            vectors, metadata = self._build_index_inputs_from_images_folder()

        self.indexer.build_index(vectors, metadata)
        self.indexer.save()
        self.indexer.load()
        return self.indexer.get_runtime_summary()

    def _load_index_inputs_from_features_files(
        self,
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        import h5py
        with h5py.File(FEATURES_HDF5, "r") as h5_file:
            vectors = np.array(h5_file["features"], dtype=np.float32)

        import json

        with FEATURES_MAPPING.open("r", encoding="utf-8") as file_obj:
            raw_mapping = json.load(file_obj)

        metadata: List[Dict[str, Any]] = []
        for item in raw_mapping:
            feature_idx = int(item["feature_idx"])
            metadata.append(
                {
                    "image_id": str(feature_idx),
                    "image_path": item["image_path"],
                    "category": item.get("category"),
                    "filename": item.get("filename"),
                }
            )
        return vectors, metadata

    def _build_index_inputs_from_images_folder(
        self,
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        if not IMAGES_DIR.exists():
            raise FileNotFoundError(
                f"Missing images folder database: {IMAGES_DIR}"
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted(
            [
                p
                for p in IMAGES_DIR.rglob("*")
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in database folder: {IMAGES_DIR}")

        vectors_list: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []

        for idx, image_path in enumerate(image_paths):
            try:
                vector = self.extractor.extract(image_path)[0]
            except Exception:
                # Skip unreadable or invalid images and continue indexing.
                continue

            relative_path = image_path.relative_to(BASE_DIR)
            category = image_path.parent.name
            vectors_list.append(vector.astype(np.float32))
            metadata.append(
                {
                    "image_id": str(idx),
                    "image_path": str(relative_path),
                    "category": category,
                    "filename": image_path.name,
                }
            )

        if not vectors_list:
            raise RuntimeError("No valid images could be indexed from the images folder.")

        vectors = np.vstack(vectors_list).astype(np.float32)
        return vectors, metadata

    def search_uploaded_image(self, image_file, top_k: int) -> List[Dict[str, Any]]:
        image_bytes = image_file.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
        # Reset stream position for compatibility if other middleware needs it.
        image_file.seek(0)

        cache_key = f"{top_k}:{hashlib.sha1(image_bytes).hexdigest()}"
        if cache_key in self._result_cache:
            self._result_cache.move_to_end(cache_key)
            return self._result_cache[cache_key]

        query_vector = self.extractor.extract_bytes(image_bytes)
        results = self.indexer.search(query_vector, top_k=top_k)

        self._result_cache[cache_key] = results
        if len(self._result_cache) > self._result_cache_size:
            self._result_cache.popitem(last=False)
        return results


app = Flask(__name__)
service: SearchBackendService | None = None
service_init_error: str | None = None


def get_service() -> SearchBackendService:
    global service, service_init_error
    if service is None:
        try:
            service = SearchBackendService()
            service_init_error = None
        except Exception as exc:
            service_init_error = str(exc)
            raise
    return service


def _to_public_image_url(image_path: str) -> str | None:
    image_abs = (BASE_DIR / image_path).resolve()
    try:
        relative = image_abs.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        return None
    return url_for("serve_image", image_path=relative.as_posix())


@app.get("/")
def home() -> Any:
    return render_template("index.html")


@app.get("/images/<path:image_path>")
def serve_image(image_path: str) -> Any:
    return send_from_directory(IMAGES_DIR, image_path)


@app.get("/health")
def health() -> Any:
    try:
        runtime = get_service().indexer.get_runtime_summary()
        return jsonify({"status": "ok", "index": runtime})
    except Exception:
        return jsonify({"status": "not_ready", "error": service_init_error}), 503


@app.post("/reindex")
def reindex() -> Any:
    summary = get_service().rebuild_index()
    return jsonify({"message": "index rebuilt", "index": summary}), 201


@app.post("/search")
def search() -> Any:
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "Missing required file field: image"}), 400

    top_k_raw = request.form.get("top_k", "5")
    try:
        top_k = int(top_k_raw)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
    except ValueError:
        return jsonify({"error": "top_k must be a positive integer"}), 400

    try:
        raw_results = get_service().search_uploaded_image(image_file=image_file, top_k=top_k)
    except RetrievalError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"search failed: {exc}"}), 500

    results: List[Dict[str, Any]] = []
    for item in raw_results:
        row = dict(item)
        image_path = row.get("image_path")
        if isinstance(image_path, str):
            row["image_url"] = _to_public_image_url(image_path)
        else:
            row["image_url"] = None
        results.append(row)

    return jsonify({"top_k": top_k, "count": len(results), "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
