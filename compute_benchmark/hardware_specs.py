"""
Hardware specifications database for FPS conversion calculations.

Specs used in the Roofline model:
  - fp16_tflops  : FP16 dense TFLOPS (Tensor Core peak, no sparsity)
  - fp32_tflops  : FP32 TFLOPS
  - int8_tops    : INT8 TOPS
  - bandwidth_gbs: memory bandwidth in GB/s
  - memory_gb    : total memory capacity in GB

"Edge-4T-INT8"  assumes the edge spec "4T算力" refers to 4 TOPS INT8 (AI-chip / NPU common unit).
"Edge-4T-FP32"  assumes it refers to 4 TFLOPS FP32 (general-purpose processor / mobile GPU unit).
"""

HARDWARE_DB = {
    # ── NVIDIA server / consumer GPUs ─────────────────────────────────────────
    "RTX 4090": {
        "fp32_tflops":   82.6,
        "fp16_tflops":  165.2,   # dense Tensor Core (non-sparse)
        "int8_tops":    661.0,
        "bandwidth_gbs": 1008.0,
        "memory_gb":      24.0,
        "description": "NVIDIA GeForce RTX 4090 (Ada Lovelace, 24 GB GDDR6X)",
    },
    "RTX 3090": {
        "fp32_tflops":   35.6,
        "fp16_tflops":   71.2,
        "int8_tops":    142.0,
        "bandwidth_gbs": 936.0,
        "memory_gb":      24.0,
        "description": "NVIDIA GeForce RTX 3090 (Ampere, 24 GB GDDR6X)",
    },
    "RTX 2080 Ti": {
        "fp32_tflops":   13.4,
        "fp16_tflops":   26.9,
        "int8_tops":    107.9,
        "bandwidth_gbs": 616.0,
        "memory_gb":      11.0,
        "description": "NVIDIA GeForce RTX 2080 Ti (Turing, 11 GB GDDR6)",
    },

    # ── Generic edge-device presets ───────────────────────────────────────────
    # 端侧 4T 算力 — AI 专用芯片 / NPU（INT8 计量，常见于 AI 芯片规格书）
    "Edge-4T-INT8": {
        "fp32_tflops":   1.0,   # 4 TOPS INT8 / 4 (bit-width factor)
        "fp16_tflops":   2.0,   # 4 TOPS INT8 / 2
        "int8_tops":     4.0,
        "bandwidth_gbs":  30.0, # typical low-power edge LPDDR4
        "memory_gb":       4.0,
        "description": "端侧 4 TOPS INT8 算力 (AI 芯片/NPU), ~30 GB/s LPDDR4",
    },
    # 端侧 4T 算力 — 通用处理器 / 移动 GPU（FP32 计量）
    "Edge-4T-FP32": {
        "fp32_tflops":   4.0,
        "fp16_tflops":   8.0,   # assuming native FP16 = 2× FP32
        "int8_tops":    16.0,
        "bandwidth_gbs":  51.2,
        "memory_gb":       8.0,
        "description": "端侧 4 TFLOPS FP32 算力 (移动 GPU/DSP), ~51.2 GB/s LPDDR5",
    },

    # ── Real edge platforms ───────────────────────────────────────────────────
    "RK3588-NPU": {
        "fp32_tflops":   0.75,
        "fp16_tflops":   1.5,
        "int8_tops":     6.0,
        "bandwidth_gbs":  51.2,
        "memory_gb":       8.0,
        "description": "Rockchip RK3588 NPU (6 TOPS INT8, LPDDR4X 51.2 GB/s)",
    },
    "Jetson-Orin-NX-8G": {
        "fp32_tflops":   4.9,
        "fp16_tflops":   9.8,   # Ampere architecture: FP16 = 2× FP32
        "int8_tops":    39.0,
        "bandwidth_gbs":  68.0,
        "memory_gb":       8.0,
        "description": "NVIDIA Jetson Orin NX 8G (39 TOPS INT8, 9.8 TFLOPS FP16, 68 GB/s LPDDR5)",
    },
    "Jetson-Nano": {
        "fp32_tflops":   0.472,
        "fp16_tflops":   0.472,  # Maxwell architecture: no native FP16, same speed as FP32
        "int8_tops":     0.944,
        "bandwidth_gbs":  25.6,
        "memory_gb":       4.0,
        "description": "NVIDIA Jetson Nano (472 GFLOPS, Maxwell arch — no native FP16 acceleration, 25.6 GB/s LPDDR4)",
    },
}
