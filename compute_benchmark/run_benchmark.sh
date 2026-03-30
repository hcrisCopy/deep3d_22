#!/usr/bin/env bash
# run_benchmark.sh — 一键运行 FPS 测试并输出端侧换算结果
#
# 用法 / Usage:
#   bash run_benchmark.sh <model_path> [gpu_id] [warmup] [iterations]
#
# 示例 / Example:
#   bash run_benchmark.sh ../export/deep3d_v1.0_640x360_cuda.pt 0 50 500

set -e

MODEL="${1:-../export/deep3d_v1.0_640x360_cuda.pt}"
GPU_ID="${2:-0}"
WARMUP="${3:-50}"
ITERS="${4:-500}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_JSON="$SCRIPT_DIR/rtx4090_benchmark.json"

echo "============================================================"
echo " Deep3D FPS Benchmark & Edge-Device Compute Conversion"
echo "============================================================"
echo " Model      : $MODEL"
echo " GPU ID     : $GPU_ID"
echo " Warm-up    : $WARMUP  iterations"
echo " Benchmark  : $ITERS iterations"
echo "------------------------------------------------------------"

# ── Step 1: Benchmark on current device ─────────────────────────────────────
echo ""
echo "[Step 1/3]  Measuring inference FPS on current device …"
python "$SCRIPT_DIR/benchmark_fps.py" \
    --model    "$MODEL"       \
    --gpu_id   "$GPU_ID"      \
    --warmup   "$WARMUP"      \
    --iterations "$ITERS"     \
    --output   "$RESULT_JSON"

# ── Step 2: Convert to Edge-4T-INT8 (AI chip / NPU) ────────────────────────
echo ""
echo "[Step 2/3]  Converting → 端侧 Edge-4T-INT8 (AI 芯片 / NPU)  [precision=fp16] …"
python "$SCRIPT_DIR/compute_conversion.py" \
    --benchmark_json "$RESULT_JSON" \
    --src  "RTX 4090"     \
    --dst  "Edge-4T-INT8" \
    --precision fp16      \
    --output_json "$SCRIPT_DIR/conversion_Edge-4T-INT8.json"

# ── Step 3: Convert to Edge-4T-FP32 (general-purpose / mobile GPU) ─────────
echo ""
echo "[Step 3/3]  Converting → 端侧 Edge-4T-FP32 (通用处理器 / 移动 GPU)  [precision=fp32] …"
python "$SCRIPT_DIR/compute_conversion.py" \
    --benchmark_json "$RESULT_JSON" \
    --src  "RTX 4090"     \
    --dst  "Edge-4T-FP32" \
    --precision fp32      \
    --output_json "$SCRIPT_DIR/conversion_Edge-4T-FP32.json"

echo ""
echo "============================================================"
echo " All done!  Output files:"
echo "   $RESULT_JSON"
echo "   $SCRIPT_DIR/conversion_Edge-4T-INT8.json"
echo "   $SCRIPT_DIR/conversion_Edge-4T-FP32.json"
echo "============================================================"
