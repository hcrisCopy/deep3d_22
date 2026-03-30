#!/usr/bin/env bash
# run_benchmark_cpu.sh — 一键运行 CPU 版本 FPS 测试并输出端侧 4T 算力换算结果
#
# 适用场景:
#   源设备: CPU — Intel(R) Xeon(R) Platinum 8470Q (20 vCPU, Sapphire Rapids)
#   模型  : deep3d_v1.0_1280x720_cpu.pt (FP32 CPU TorchScript 模型)
#   目标  : 端侧 4T 算力（AI 芯片 / NPU，或通用处理器 / 移动 GPU）
#
# 用法 / Usage:
#   bash run_benchmark_cpu.sh [model_path] [warmup] [iterations]
#
# 示例 / Examples:
#   bash run_benchmark_cpu.sh
#   bash run_benchmark_cpu.sh /root/autodl-tmp/deep3d_22/export/deep3d_v1.0_1280x720_cpu.pt
#   bash run_benchmark_cpu.sh /root/autodl-tmp/deep3d_22/export/deep3d_v1.0_1280x720_cpu.pt 5 30

set -e

MODEL="${1:-/root/autodl-tmp/deep3d_22/export/deep3d_v1.0_1280x720_cpu.pt}"
WARMUP="${2:-5}"
ITERS="${3:-30}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_JSON="$SCRIPT_DIR/cpu_benchmark.json"

echo "============================================================"
echo " Deep3D CPU FPS Benchmark & Edge-Device Compute Conversion"
echo "============================================================"
echo " Source CPU : Intel Xeon Platinum 8470Q (20 vCPU)"
echo " Model      : $MODEL"
echo " Warm-up    : $WARMUP  iterations"
echo " Benchmark  : $ITERS iterations"
echo "------------------------------------------------------------"
echo " NOTE: CPU inference is slower than GPU. With 1280x720 input"
echo "       each forward pass may take several seconds."
echo "       Total estimated time: ~$((WARMUP + ITERS)) forward passes."
echo "============================================================"

# ── Step 1: Benchmark on CPU ─────────────────────────────────────────────────
echo ""
echo "[Step 1/3]  Measuring inference FPS on CPU (--gpu_id -1) …"
python "$SCRIPT_DIR/benchmark_fps.py" \
    --model      "$MODEL"    \
    --gpu_id     -1          \
    --warmup     "$WARMUP"   \
    --iterations "$ITERS"    \
    --output     "$RESULT_JSON"

# ── Step 2: Convert to Edge-4T-INT8 (AI chip / NPU) ─────────────────────────
echo ""
echo "[Step 2/3]  Converting → 端侧 Edge-4T-INT8 (AI 芯片 / NPU)  [precision=fp32] …"
python "$SCRIPT_DIR/compute_conversion.py" \
    --benchmark_json "$RESULT_JSON"       \
    --src  "Xeon-8470Q-20vCPU"            \
    --dst  "Edge-4T-INT8"                 \
    --precision fp32                      \
    --output_json "$SCRIPT_DIR/cpu_conversion_Edge-4T-INT8.json"

# ── Step 3: Convert to Edge-4T-FP32 (general-purpose / mobile GPU) ──────────
echo ""
echo "[Step 3/3]  Converting → 端侧 Edge-4T-FP32 (通用处理器 / 移动 GPU)  [precision=fp32] …"
python "$SCRIPT_DIR/compute_conversion.py" \
    --benchmark_json "$RESULT_JSON"       \
    --src  "Xeon-8470Q-20vCPU"            \
    --dst  "Edge-4T-FP32"                 \
    --precision fp32                      \
    --output_json "$SCRIPT_DIR/cpu_conversion_Edge-4T-FP32.json"

echo ""
echo "============================================================"
echo " All done!  Output files:"
echo "   $RESULT_JSON"
echo "   $SCRIPT_DIR/cpu_conversion_Edge-4T-INT8.json"
echo "   $SCRIPT_DIR/cpu_conversion_Edge-4T-FP32.json"
echo "============================================================"
