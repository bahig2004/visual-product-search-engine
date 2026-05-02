from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend_api import SearchBackendService


def evaluate(
    sample_size: int,
    top_k: int,
    seed: int,
    output_path: Path | None,
) -> Dict[str, Any]:
    random.seed(seed)
    service = SearchBackendService()

    if not service.indexer.metadata:
        raise RuntimeError("No indexed metadata found. Build/load index first.")

    population = service.indexer.metadata
    eval_size = min(sample_size, len(population))
    samples = random.sample(population, eval_size)

    latencies_ms: List[float] = []
    hit_at_1 = 0
    hit_at_k = 0

    for item in samples:
        image_path = item.get("image_path")
        category = item.get("category")
        if not image_path:
            continue

        query_abs_path = (Path(__file__).resolve().parent.parent / image_path).resolve()
        if not query_abs_path.exists():
            continue

        t0 = time.perf_counter()
        query_vector = service.extractor.extract(query_abs_path)
        results = service.indexer.search(query_vector, top_k=top_k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        if not results:
            continue

        # Exact self-match
        if results[0].get("image_path") == image_path:
            hit_at_1 += 1

        # Category-level relevance
        if category is not None and any(r.get("category") == category for r in results):
            hit_at_k += 1

    evaluated = len(latencies_ms)
    if evaluated == 0:
        raise RuntimeError("No valid evaluation samples were processed.")

    total_ms = float(sum(latencies_ms))
    report = {
        "config": {
            "sample_size_requested": sample_size,
            "sample_size_evaluated": evaluated,
            "top_k": top_k,
            "seed": seed,
        },
        "accuracy": {
            "recall_at_1": hit_at_1 / evaluated,
            "recall_at_k_category": hit_at_k / evaluated,
        },
        "speed": {
            "latency_ms_mean": statistics.mean(latencies_ms),
            "latency_ms_median": statistics.median(latencies_ms),
            "latency_ms_min": min(latencies_ms),
            "latency_ms_max": max(latencies_ms),
            "throughput_qps": (1000.0 * evaluated / total_ms) if total_ms > 0 else 0.0,
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality and speed for the visual search system."
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/evaluation/evaluation_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
