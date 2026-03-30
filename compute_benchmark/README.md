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

### 1. 从延迟公式推导换算比值

推理一帧的**理论延迟下界**由两个硬件上限共同决定：

```
延迟_计算 = 模型 FLOPs  / 硬件峰值算力          （单位：s）
延迟_带宽 = 模型访存量  / 硬件峰值内存带宽      （单位：s）
理论延迟  = max(延迟_计算, 延迟_带宽)           （取较大者，即瓶颈）
理论 FPS  = 1 / 理论延迟
```

> **FLOPs 消去原理**：将同一模型分别部署在"源硬件（src）"和"目标硬件（dst）"上，
> 对同一帧的 FLOPs 和访存量完全相同，因此在两端的比值中自动消去：
>
> ```
> FPS_dst    1/延迟_dst    理论算力_dst / 模型FLOPs     算力_dst
> ─────── = ──────────── = ────────────────────────── = ─────────
> FPS_src    1/延迟_src    理论算力_src / 模型FLOPs     算力_src
> ```
>
> **结论：FPS 换算比等于两端硬件规格之比，与模型本身的 FLOPs 无关。**  
> 这正是本工具不需要预先 profiling 模型 FLOPs 的数学依据。

---

### 2. Roofline 模型完整推导

Roofline 模型用**算术强度（Arithmetic Intensity, AI）**描述模型对硬件的需求：

```
算术强度 AI = 模型 FLOPs / 模型访存字节数     （单位：FLOP/Byte）
```

硬件的**机器平衡点（Ridge Point）**：

```
机器平衡点 = 峰值算力（FLOPS）/ 峰值内存带宽（B/s）   （单位：FLOP/Byte）
```

| 情形 | 判断条件 | 瓶颈 | 理论 FPS ∝ |
|------|----------|------|-----------|
| 计算受限（Compute-bound） | `AI > 机器平衡点` | 芯片算力 | 算力比 |
| 带宽受限（Bandwidth-bound） | `AI < 机器平衡点` | 内存带宽 | 带宽比 |

**RTX 4090 的机器平衡点示例：**

```
机器平衡点 = 165.2 TFLOPS / 1008 GB/s ≈ 164 FLOP/Byte
```

若 Deep3D（640×360）的算术强度 > 164 FLOP/Byte → 计算瓶颈；  
若 < 164 FLOP/Byte → 带宽瓶颈（在 RTX 4090 上较少见）。

**Roofline 示意图（对数坐标）：**

```
  FPS
   │                                  ╱──────────── 计算上限（算力/FLOPs）
   │                              ╱──
   │                          ╱──
   │    带宽上限（BW/访存量）╱
   │   ──────────────────────/ ← 机器平衡点（Ridge Point）
   │
   └────────────────────────────────► 算术强度 (FLOP/Byte)
```

---

### 3. 换算因子的计算步骤

以 **RTX 4090 → Edge-4T-INT8（精度 FP16）** 为例，逐步说明：

**① 读取两端硬件规格（来自 `hardware_specs.py`）：**

| 规格 | RTX 4090 | Edge-4T-INT8 |
|------|----------|--------------|
| FP16 峰值算力 | 165.2 TFLOPS | 2.0 TFLOPS |
| 内存带宽 | 1008 GB/s | 30 GB/s |

**② 计算两个比值：**

```
算力比 = dst_fp16_tflops / src_fp16_tflops = 2.0 / 165.2 = 0.0121

带宽比 = dst_bandwidth   / src_bandwidth   = 30  / 1008  = 0.0298
```

**③ 判断瓶颈方向：**

```
算力比 (0.0121) < 带宽比 (0.0298)
→ 目标设备算力更紧张 → COMPUTE-bound
```

**④ 三档换算因子：**

```
保守 conservative = min(0.0121, 0.0298) = 0.0121   ← 取最紧瓶颈
适中 moderate     = √(0.0121 × 0.0298) = 0.0190   ← 几何平均
乐观 optimistic   = max(0.0121, 0.0298) = 0.0298   ← 假设只受一种瓶颈
```

**⑤ 乘以实测 FPS（以 312.54 FPS 为例）：**

```
保守：312.54 × 0.0121 = 3.78 FPS
适中：312.54 × 0.0190 = 5.94 FPS   ← 推荐参考值
乐观：312.54 × 0.0298 = 9.31 FPS
```

以上逻辑完整对应 `compute_conversion.py` 中的 `compute_conversion_factors()` 函数。

---

### 4. 三档估算的数学含义

| 档位 | 公式 | 物理含义 |
|------|------|----------|
| **保守 Conservative** | `FPS × min(算力比, 带宽比)` | 两种瓶颈**同时**存在时，取限制最紧的一个，代表真实最坏情况 |
| **适中 Moderate** | `FPS × √(算力比 × 带宽比)` | 两种瓶颈的**几何平均**，对"部分计算受限 + 部分带宽受限"的混合场景建模；**推荐作为工程参考值** |
| **乐观 Optimistic** | `FPS × max(算力比, 带宽比)` | 假设模型**只受其中一种**瓶颈限制（另一种资源充裕），代表理论最优上界 |

> **几何平均的直觉**：若真实利用率在计算与带宽之间均匀分布，则几何平均比算术平均
> 更接近实际情况，因为 FPS 对比值的响应是乘性（multiplicative）而非加性的。

---

### 5. 精度参数对换算的影响

不同精度对应 `hardware_specs.py` 中不同的规格字段：

| `--precision` | 使用字段 | 典型场景 |
|---------------|----------|---------|
| `fp16` | `fp16_tflops` | Deep3D CUDA 模型（`*_cuda.pt`）在 GPU 上的默认精度 |
| `fp32` | `fp32_tflops` | CPU 推理，或不支持 FP16 的硬件（如 Jetson Nano Maxwell 架构） |
| `int8` | `int8_tops` | INT8 量化后的模型，适用于 NPU / TensorRT INT8 模式 |

**FP16 → INT8 额外加速估算：**

若端侧部署经过 INT8 量化，实际 FPS 通常高于 FP16 估算值：

```
INT8 估算 FPS ≈ FP16 估算 FPS × 量化加速系数（一般 1.5–2.0×）
```

可以直接使用 `--precision int8` 并将源 FPS 先换算到 INT8 基准，
或在换算结果上手动乘以加速系数。

---

### 6. "端侧 4T 算力"的两种解读

产品规格书中"4T 算力"的单位并不统一，对应两种完全不同的预设：

| 配置 | 适用场景 | FP16 等效算力 | 含义 |
|------|----------|---------------|------|
| `Edge-4T-INT8` | AI 专用芯片 / NPU | 2.0 TFLOPS | **4 TOPS INT8**；INT8 换 FP16 需除以 2（位宽因子） |
| `Edge-4T-FP32` | 通用处理器 / 移动 GPU | 8.0 TFLOPS | **4 TFLOPS FP32**；FP32 换 FP16 通常乘以 2（硬件原生 FP16） |

> 拿到具体芯片规格书时，请确认"T"代表 TOPS（整型）还是 TFLOPS（浮点），  
> 再选择对应预设或直接在 `hardware_specs.py` 中添加自定义条目。

---

## 注意事项

1. **框架转换加速**：端侧部署通常需要将模型转为 ONNX → TensorRT / 厂商专用格式，经过 INT8 量化后实际 FPS 可能**高于**本工具的 FP16 估算值。  
2. **热降频**：端侧设备长时间运行时，CPU/NPU 可能降频，实际 FPS 下降 **10–30%**。  
3. **搬运开销**：CPU↔NPU 的数据传输延迟在端侧较显著，实际 FPS 可能比理论值低 **20–40%**。  
4. **本 benchmark 测量的是纯推理时间**（不含视频读写、resize 等 IO 操作），与完整流水线的 FPS 会有差异。
