"""
compute_conversion.py — Estimate inference FPS on edge devices from a
measured RTX 4090 (or any source device) benchmark result.

Methodology: Roofline model
  The neural network inference speed is limited by the tighter of two
  ceilings:
    1. Compute ceiling   — peak FLOPs/TOPS of the chip
    2. Bandwidth ceiling — peak memory bandwidth of the chip

  conversion_factor = min(compute_ratio, bandwidth_ratio)   [conservative]
  conversion_factor = sqrt(compute_ratio × bandwidth_ratio) [moderate]
  conversion_factor = max(compute_ratio, bandwidth_ratio)   [optimistic]

  estimated_FPS_edge = measured_FPS_RTX4090 × conversion_factor

Usage examples:
  # From a saved benchmark JSON:
  python compute_conversion.py \\
      --benchmark_json rtx4090_benchmark.json \\
      --src "RTX 4090" --dst "Edge-4T-INT8" --precision fp16

  # Direct FPS input:
  python compute_conversion.py --fps 312.5 --src "RTX 4090" --dst "RK3588-NPU"

  # List available hardware profiles:
  python compute_conversion.py --list_hardware
"""

import argparse
import json
import math
import os
import sys

# Import hardware database from the same package directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_specs import HARDWARE_DB


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

_PREC_KEY = {
    "fp16": "fp16_tflops",
    "fp32": "fp32_tflops",
    "int8": "int8_tops",
}


def compute_conversion_factors(src_hw: dict, dst_hw: dict,
                                precision: str = "fp16") -> dict:
    """
    Calculate theoretical FPS-conversion factors from src to dst hardware.

    Returns a dict with:
        compute_ratio       — dst_compute / src_compute
        bandwidth_ratio     — dst_bandwidth / src_bandwidth
        bottleneck          — "compute" or "bandwidth"
        conservative_factor — min(compute, bandwidth) ratio  [worst-case]
        moderate_factor     — geometric mean of both          [recommended]
        optimistic_factor   — max(compute, bandwidth) ratio  [best-case]
    """
    prec_key = _PREC_KEY[precision]
    compute_ratio   = dst_hw[prec_key]        / src_hw[prec_key]
    bandwidth_ratio = dst_hw["bandwidth_gbs"] / src_hw["bandwidth_gbs"]

    conservative = min(compute_ratio, bandwidth_ratio)
    moderate     = math.sqrt(compute_ratio * bandwidth_ratio)
    optimistic   = max(compute_ratio, bandwidth_ratio)
    bottleneck   = "compute" if compute_ratio < bandwidth_ratio else "bandwidth"

    return {
        "compute_ratio":      compute_ratio,
        "bandwidth_ratio":    bandwidth_ratio,
        "bottleneck":         bottleneck,
        "conservative_factor": conservative,
        "moderate_factor":     moderate,
        "optimistic_factor":   optimistic,
    }


def estimate_fps(src_fps: float, src_hw: dict, dst_hw: dict,
                 precision: str = "fp16") -> dict:
    """Return conversion factors plus estimated FPS values."""
    factors = compute_conversion_factors(src_hw, dst_hw, precision)
    return {
        **factors,
        "src_fps": src_fps,
        "estimated_fps": {
            "conservative": src_fps * factors["conservative_factor"],
            "moderate":     src_fps * factors["moderate_factor"],
            "optimistic":   src_fps * factors["optimistic_factor"],
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(src_name: str, dst_name: str,
                 src_fps: float, precision: str = "fp16") -> dict:
    src_hw = HARDWARE_DB[src_name]
    dst_hw = HARDWARE_DB[dst_name]
    result = estimate_fps(src_fps, src_hw, dst_hw, precision)

    pk = _PREC_KEY[precision]
    unit = "TOPS" if precision == "int8" else "TFLOPS"

    print("\n" + "=" * 62)
    print(f"  FPS Conversion:  {src_name}  →  {dst_name}")
    print("=" * 62)
    print(f"  源硬件 (Source) : {src_hw['description']}")
    print(f"  目标硬件 (Target): {dst_hw['description']}")
    print(f"  推理精度 (Precision): {precision.upper()}")
    print(f"  测量 FPS ({src_name}): {src_fps:.2f}")
    print()
    print("  换算因子 (Conversion factors):")
    print(f"    算力比 (Compute ratio)  : {result['compute_ratio']:.4f}"
          f"  ({dst_hw[pk]:.2f} / {src_hw[pk]:.1f} {unit})")
    print(f"    带宽比 (Bandwidth ratio): {result['bandwidth_ratio']:.4f}"
          f"  ({dst_hw['bandwidth_gbs']:.1f} / {src_hw['bandwidth_gbs']:.1f} GB/s)")
    print(f"    瓶颈   (Bottleneck)     : {result['bottleneck'].upper()}-bound")
    print()
    print("  预估端侧 FPS (Estimated edge FPS):")
    print(f"    保守 Conservative (瓶颈限制)     : "
          f"{result['estimated_fps']['conservative']:7.2f} FPS")
    print(f"    适中 Moderate     (几何平均)      : "
          f"{result['estimated_fps']['moderate']:7.2f} FPS  ← 推荐参考值")
    print(f"    乐观 Optimistic   (理论最优)      : "
          f"{result['estimated_fps']['optimistic']:7.2f} FPS")
    print()
    print("  注意事项 (Practical caveats):")
    print("    · 框架转换 (ONNX/TensorRT/厂商 SDK) 可能带来额外加速")
    print("    · INT8 量化可将 FP16 模型实际 FPS 提升约 1.5–2×")
    print("    · 端侧热降频可能使实际 FPS 降低 10–30%")
    print("    · CPU↔NPU 数据搬运延迟可能使实际 FPS 降低 20–40%")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    hw_names = list(HARDWARE_DB.keys())
    p = argparse.ArgumentParser(
        description="Estimate Deep3D inference FPS on edge devices from RTX 4090 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", default="RTX 4090", choices=hw_names,
                   help="Source hardware (where FPS was measured)")
    p.add_argument("--dst", default="Edge-4T-INT8", choices=hw_names,
                   help="Target hardware (where FPS is to be estimated)")
    p.add_argument("--fps", type=float, default=None,
                   help="Measured FPS on source hardware")
    p.add_argument("--benchmark_json", type=str, default=None,
                   help="Path to JSON produced by benchmark_fps.py "
                        "(reads fps.mean automatically)")
    p.add_argument("--precision", default="fp16",
                   choices=["fp16", "fp32", "int8"],
                   help="Inference precision of the model on source hardware "
                        "(Deep3D CUDA model uses fp16)")
    p.add_argument("--list_hardware", action="store_true",
                   help="Print all available hardware profiles and exit")
    p.add_argument("--output_json", type=str, default=None,
                   help="Save conversion result to a JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_hardware:
        print("\nAvailable hardware profiles:")
        print(f"  {'Name':<25}  Description")
        print("  " + "-" * 70)
        for name, hw in HARDWARE_DB.items():
            print(f"  {name:<25}  {hw['description']}")
        return

    # Resolve source FPS
    src_fps = args.fps
    if src_fps is None and args.benchmark_json:
        with open(args.benchmark_json) as f:
            data = json.load(f)
        src_fps = data["fps"]["mean"]
        print(f"Loaded FPS from {args.benchmark_json}: {src_fps:.2f} FPS")

    if src_fps is None:
        print("Error: provide --fps <value> or --benchmark_json <path>")
        sys.exit(1)

    result = print_report(args.src, args.dst, src_fps, args.precision)

    if args.output_json:
        output = {
            "src_hardware": args.src,
            "dst_hardware": args.dst,
            "precision": args.precision,
            **result,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nConversion result saved → {args.output_json}")


if __name__ == "__main__":
    main()
