# Deep3D 算力换算工具  
**RTX 4090 → 端侧 4T 算力  推理速度（FPS）换算标准**

---

## 目录结构

```
compute_benchmark/
├── benchmark_fps.py       # 在 RTX 4090（或任意 CUDA/CPU 设备）上测量纯推理 FPS
├── compute_conversion.py  # 基于 Roofline 模型将 RTX 4090 FPS 换算到端侧设备
├── hardware_specs.py      # 硬件规格数据库（可自行扩展）
├── run_benchmark.sh       # 一键脚本：测速 + 换算
└── README.md              # 本文档
```

---

## 环境配置

与主项目保持相同的 PyTorch 环境：

```bash
conda create -n deep3d python=3.8
conda activate deep3d

# PyTorch 1.7.1 + CUDA 11.0（与作者 1.txt 保持一致）
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install -r ../requirements.txt
pip install psutil
```

> **注意**：`benchmark_fps.py` 仅依赖 `torch` 和 `numpy`，与主项目共享同一环境即可。

---

## 快速开始（一键运行）

```bash
cd compute_benchmark
bash run_benchmark.sh ../export/deep3d_v1.0_640x360_cuda.pt
```

脚本会依次执行：
1. 在当前 GPU（RTX 4090）上测量推理 FPS，结果写入 `rtx4090_benchmark.json`  
2. 换算到 **端侧 4T INT8**（AI 芯片 / NPU），结果写入 `conversion_Edge-4T-INT8.json`  
3. 换算到 **端侧 4T FP32**（通用处理器 / 移动 GPU），结果写入 `conversion_Edge-4T-FP32.json`

---

## 步骤详解

### Step 1：在 RTX 4090 上测量 FPS

```bash
python benchmark_fps.py \
    --model      ../export/deep3d_v1.0_640x360_cuda.pt \
    --gpu_id     0        \
    --warmup     50       \
    --iterations 500      \
    --output     rtx4090_benchmark.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | *必填* | `.pt` 文件路径，文件名须含 `WIDTHxHEIGHT`（如 `640x360`） |
| `--gpu_id` | `0` | GPU 编号，`-1` 强制使用 CPU |
| `--warmup` | `50` | 预热轮次（不计入统计） |
| `--iterations` | `500` | 正式计时轮次 |
| `--output` | 自动生成 | 结果 JSON 路径 |

**输出示例：**
```
Device   : GPU:0 (NVIDIA GeForce RTX 4090)
Resolution (model input): 640x360
Precision: FP16

Benchmark Results
═══════════════════════════════════════════════════════
  FPS  mean  : 312.54
  FPS  median: 315.02
  FPS  P5–P95: 298.12 – 324.67
  Latency ms : 3.20 (mean)  3.36 (P95)
```

---

### Step 2：端侧算力换算

```bash
# 从上一步生成的 JSON 读取 FPS（推荐）
python compute_conversion.py \
    --benchmark_json rtx4090_benchmark.json \
    --src "RTX 4090"     \
    --dst "Edge-4T-INT8" \
    --precision fp16

# 或直接指定 FPS 数值
python compute_conversion.py --fps 312.54 --src "RTX 4090" --dst "Edge-4T-INT8"

# 查看所有支持的硬件配置
python compute_conversion.py --list_hardware
```

**输出示例：**
```
══════════════════════════════════════════════════════════════
  FPS Conversion:  RTX 4090  →  Edge-4T-INT8
══════════════════════════════════════════════════════════════
  源硬件 (Source) : NVIDIA GeForce RTX 4090 (Ada Lovelace, 24 GB GDDR6X)
  目标硬件 (Target): 端侧 4 TOPS INT8 算力 (AI 芯片/NPU), ~30 GB/s LPDDR4
  推理精度 (Precision): FP16
  测量 FPS (RTX 4090): 312.54

  换算因子 (Conversion factors):
    算力比 (Compute ratio)  : 0.0121  (2.00 / 165.2 TFLOPS)
    带宽比 (Bandwidth ratio): 0.0298  (30.0 / 1008.0 GB/s)
    瓶颈   (Bottleneck)     : COMPUTE-bound

  预估端侧 FPS (Estimated edge FPS):
    保守 Conservative (瓶颈限制)     :    3.78 FPS
    适中 Moderate     (几何平均)      :    5.99 FPS  ← 推荐参考值
    乐观 Optimistic   (理论最优)      :    9.32 FPS
```

---

### 保存换算结果到 JSON

```bash
python compute_conversion.py \
    --benchmark_json rtx4090_benchmark.json \
    --src "RTX 4090" --dst "Edge-4T-INT8" \
    --precision fp16 \
    --output_json conversion_Edge-4T-INT8.json
```

---

## 支持的硬件配置

| 名称 | 描述 |
|------|------|
| `RTX 4090` | NVIDIA GeForce RTX 4090, 165.2 TFLOPS FP16, 1008 GB/s |
| `RTX 3090` | NVIDIA GeForce RTX 3090, 71.2 TFLOPS FP16, 936 GB/s |
| `RTX 2080 Ti` | NVIDIA GeForce RTX 2080 Ti, 26.9 TFLOPS FP16, 616 GB/s |
| `Edge-4T-INT8` | 端侧 4 TOPS INT8（AI 芯片 / NPU），30 GB/s |
| `Edge-4T-FP32` | 端侧 4 TFLOPS FP32（通用处理器 / 移动 GPU），51.2 GB/s |
| `RK3588-NPU` | Rockchip RK3588 NPU，6 TOPS INT8，51.2 GB/s |
| `Jetson-Orin-NX-8G` | NVIDIA Jetson Orin NX 8G，39 TOPS，68 GB/s |
| `Jetson-Nano` | NVIDIA Jetson Nano，472 GFLOPS FP32，25.6 GB/s |

如需添加自定义设备，直接编辑 `hardware_specs.py` 中的 `HARDWARE_DB` 字典即可。

---

## 换算原理

### Roofline 模型

```
算术强度 (AI) = FLOPs / 访存字节数

若 AI > 机器平衡点  →  计算瓶颈  →  使用算力比
若 AI < 机器平衡点  →  带宽瓶颈  →  使用带宽比
```

对于 Deep3D（基于 UNet 的 2D 卷积网络），在 RTX 4090 上通常是**计算瓶颈**；  
在内存带宽受限的端侧设备上可能转变为**带宽瓶颈**。  
工具会自动判断并标注瓶颈类型。

### 三档估算说明

| 档位 | 计算公式 | 使用建议 |
|------|----------|----------|
| **保守 Conservative** | `min(算力比, 带宽比)` | 实际最坏情况，考虑最紧的瓶颈 |
| **适中 Moderate** | `√(算力比 × 带宽比)` | 综合考量，**推荐作为参考值** |
| **乐观 Optimistic** | `max(算力比, 带宽比)` | 理论最优，假设只受一种瓶颈限制 |

### 精度参数选择

Deep3D 的 CUDA 模型（`*_cuda.pt`）在 GPU 上以 **FP16** 运行，  
因此与 RTX 4090 对比时使用 `--precision fp16`。

端侧设备若使用 INT8 量化推理，可改用 `--precision int8`，但需将 FP16 FPS 先换算到 INT8 FPS（通常 ×1.5–2）。

### "端侧 4T 算力"的两种解读

| 配置 | 适用场景 | 含义 |
|------|----------|------|
| `Edge-4T-INT8` | AI 专用芯片 / NPU | 4 TOPS INT8（AI 芯片规格书常用单位） |
| `Edge-4T-FP32` | 通用处理器 / 移动 GPU | 4 TFLOPS FP32（通用算力常用单位） |

---

## 注意事项

1. **框架转换加速**：端侧部署通常需要将模型转为 ONNX → TensorRT / 厂商专用格式，经过 INT8 量化后实际 FPS 可能**高于**本工具的 FP16 估算值。  
2. **热降频**：端侧设备长时间运行时，CPU/NPU 可能降频，实际 FPS 下降 **10–30%**。  
3. **搬运开销**：CPU↔NPU 的数据传输延迟在端侧较显著，实际 FPS 可能比理论值低 **20–40%**。  
4. **本 benchmark 测量的是纯推理时间**（不含视频读写、resize 等 IO 操作），与完整流水线的 FPS 会有差异。
