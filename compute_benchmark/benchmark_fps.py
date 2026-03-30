"""
benchmark_fps.py — Measure Deep3D pure inference FPS on the current device.

Benchmarks only the neural-network forward pass (no video I/O, no pre/post
processing).  Results are saved to a JSON file that can be fed directly into
compute_conversion.py.

Usage example:
    python benchmark_fps.py \\
        --model ../export/deep3d_v1.0_640x360_cuda.pt \\
        --gpu_id 0 --warmup 50 --iterations 500 \\
        --output rtx4090_benchmark.json
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch

# Allow importing from the parent project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Deep3D model inference speed (FPS)"
    )
    p.add_argument(
        "--model", required=True, type=str,
        help="Path to TorchScript .pt model file "
             "(filename must contain WIDTHxHEIGHT, e.g. deep3d_v1.0_640x360_cuda.pt)"
    )
    p.add_argument(
        "--gpu_id", default=0, type=int,
        help="GPU device ID; use -1 to force CPU"
    )
    p.add_argument(
        "--warmup", default=50, type=int,
        help="Number of warm-up iterations (not counted in results)"
    )
    p.add_argument(
        "--iterations", default=500, type=int,
        help="Number of timed iterations"
    )
    p.add_argument(
        "--output", default="", type=str,
        help="Output JSON file path "
             "(default: benchmark_<model_stem>_results.json in the same directory)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_resolution(model_path):
    """
    Parse (width, height) from the model filename.
    Expected format: *_WIDTHxHEIGHT_*.pt  (e.g. deep3d_v1.0_640x360_cuda.pt)
    """
    basename = os.path.basename(model_path)
    for part in basename.split("_"):
        if "x" in part:
            wh = part.split("x")
            if len(wh) == 2 and wh[0].isdigit() and wh[1].isdigit():
                return int(wh[0]), int(wh[1])
    raise ValueError(
        f"Cannot parse resolution from model filename: {model_path}\n"
        f"Expected format: deep3d_vX.X_WIDTHxHEIGHT_*.pt"
    )


def build_dummy_input(width, height, device_id, use_half):
    """
    Build a synthetic [1, 18, H, W] input tensor matching the model's format.

    The model receives 6 consecutive frames concatenated along the channel
    axis (6 × RGB = 18 channels), normalised to [0, 1].
    """
    dtype = torch.float16 if use_half else torch.float32
    frame = torch.zeros(3, height, width, dtype=dtype)
    if device_id >= 0:
        frame = frame.cuda(device_id)
    # [18, H, W] → [1, 18, H, W]
    return torch.cat([frame] * 6, dim=0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    print(f"Loading model: {args.model}")
    net = torch.jit.load(args.model)
    net.eval()

    # Device & precision
    cuda_available = torch.cuda.is_available() and args.gpu_id >= 0
    use_half = cuda_available and ("cuda" in args.model.lower())
    device_id = args.gpu_id if cuda_available else -1

    if device_id >= 0:
        net = net.cuda(device_id)
        if use_half:
            net = net.half()
        device_name = torch.cuda.get_device_name(device_id)
        device_str = f"GPU:{device_id} ({device_name})"
    else:
        device_str = "CPU"

    precision_str = "FP16" if use_half else "FP32"
    print(f"Device   : {device_str}")
    print(f"Precision: {precision_str}")

    # Input tensor
    width, height = get_model_resolution(args.model)
    print(f"Resolution (model input): {width}x{height}")
    dummy_input = build_dummy_input(width, height, device_id, use_half)

    def sync():
        if device_id >= 0:
            torch.cuda.synchronize(device_id)

    # ── Warm-up ────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} iterations)…")
    with torch.no_grad():
        for _ in range(args.warmup):
            net(dummy_input)
    sync()

    # ── Timed iterations ───────────────────────────────────────────────────
    print(f"Benchmarking ({args.iterations} iterations)…")
    latencies = []
    with torch.no_grad():
        for i in range(args.iterations):
            sync()
            t0 = time.perf_counter()
            net(dummy_input)
            sync()
            latencies.append(time.perf_counter() - t0)

            if (i + 1) % 100 == 0:
                recent = latencies[-100:]
                print(f"  [{i+1:4d}/{args.iterations}]  "
                      f"recent FPS: {100 / sum(recent):.1f}")

    lat = np.array(latencies)
    fps = 1.0 / lat

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Benchmark Results")
    print("=" * 55)
    print(f"  Device     : {device_str}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Precision  : {precision_str}")
    print(f"  Iterations : {args.iterations}")
    print(f"  FPS  mean  : {fps.mean():.2f}")
    print(f"  FPS  median: {np.median(fps):.2f}")
    print(f"  FPS  P5–P95: {np.percentile(fps, 5):.2f} – {np.percentile(fps, 95):.2f}")
    print(f"  Latency ms : {lat.mean()*1000:.2f} (mean)  "
          f"{np.percentile(lat, 95)*1000:.2f} (P95)")

    results = {
        "model": args.model,
        "device": device_str,
        "precision": precision_str,
        "resolution": f"{width}x{height}",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "fps": {
            "mean":   float(fps.mean()),
            "median": float(np.median(fps)),
            "std":    float(fps.std()),
            "min":    float(fps.min()),
            "max":    float(fps.max()),
            "p5":     float(np.percentile(fps, 5)),
            "p95":    float(np.percentile(fps, 95)),
        },
        "latency_ms": {
            "mean":   float(lat.mean() * 1000),
            "median": float(np.median(lat) * 1000),
            "p95":    float(np.percentile(lat, 95) * 1000),
        },
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = args.output
    if not out_path:
        stem = os.path.splitext(os.path.basename(args.model))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"benchmark_{stem}_results.json",
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark(parse_args())
