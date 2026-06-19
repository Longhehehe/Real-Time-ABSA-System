"""
Micro-benchmark for PhoBERT ABSA inference latency.

Use this before and after changes such as max_length tuning, distillation,
ONNX export, quantization, or dynamic batching.
"""
import argparse
import os
import statistics
import sys
import time


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "app"))

from absa_predictor import PhoBERTPredictor  # noqa: E402


SAMPLES = [
    "San pham dung nhu mo ta, dong goi can than, giao hang nhanh.",
    "Chat luong kem, dung vai ngay da hong, shop ho tro cham.",
    "Gia tot nhung van chuyen hoi lau, hop bi mop nhe.",
    "May chay on, pin tot, nhung nong khi dung lau.",
    "Hang chinh hang, bao hanh ro rang, se mua tiep.",
]


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[idx]


def run_benchmark(batch_size, iterations, warmup):
    predictor = PhoBERTPredictor()
    if not predictor.load_model():
        raise SystemExit("Model could not be loaded")

    predictor.batch_size = batch_size
    batch = [SAMPLES[i % len(SAMPLES)] for i in range(batch_size)]

    for _ in range(warmup):
        predictor.predict_batch(batch, format="multipolarity")

    latencies = []
    total_items = 0
    start_all = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        predictor.predict_batch(batch, format="multipolarity")
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        total_items += len(batch)

    total_elapsed = time.perf_counter() - start_all

    print({
        "device": predictor.device,
        "batch_size": batch_size,
        "max_length": predictor.max_length,
        "iterations": iterations,
        "throughput_items_per_sec": round(total_items / total_elapsed, 2),
        "latency_ms_avg": round(statistics.mean(latencies) * 1000, 2),
        "latency_ms_p50": round(percentile(latencies, 50) * 1000, 2),
        "latency_ms_p95": round(percentile(latencies, 95) * 1000, 2),
        "latency_ms_p99": round(percentile(latencies, 99) * 1000, 2),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("ABSA_BATCH_SIZE", "32")))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(args.batch_size, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
