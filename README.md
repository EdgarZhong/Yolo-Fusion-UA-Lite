# YOLO-Fusion-UA-Lite

双模态 RGB-IR 轻量化 OBB 车辆检测研究项目。
基于 YOLOv8n，面向 DroneVehicle 数据集，目标 ≈6M 参数，投稿 Remote Sensing（SCI Q1）。

**当前最优：** M5（FA-Concat + COCO 预训练），mAP50 = **0.762**，mAP50-95 = 0.615
**当前阶段：** 注意力机制对比实验（Exp-0/A/B/C），详见 `docs/research/研究备忘录_v7.md` 第五节

---

## 环境激活

```powershell
& "C:\DevLib\miniconda3\Scripts\conda.exe" "shell.powershell" "hook" | Out-String | Invoke-Expression
conda activate .\.conda\ultra82-py312
```

版本锁定：`ultralytics==8.2.103`（editable install）、`numpy==1.26.4`、`torch==2.6.0+cu124`（CUDA 12.4）

**框架唯一性约束：** 必须使用 `ultralytics-8.2/` editable install，不得混用 pip 版本，混用会产生静默错误。
实际源码路径：`ultralytics-8.2/ultralytics/`

---

## 实验结果总览

| 模型代号 | 描述 | mAP50 | mAP50-95 | Recall | Precision | 状态 |
|---------|------|-------|----------|--------|-----------|------|
| M0 IR-only Scratch | 单模态基准（无预训练） | 0.662 | — | 0.622 | 0.678 | 完成 |
| IR-only COCO | 单模态 + COCO 预训练 | 0.707 | 0.533 | 0.644 | 0.738 | 完成 |
| M3 FA-Concat Scratch | 引入 Neck 后跳升 | 0.730 | — | 0.677 | — | 完成 |
| **M5 FA-Concat COCO** | **当前最优** | **0.762** | **0.615** | **0.704** | **0.785** | 完成 |
| M6 正则化实验 | 确认 Recall 瓶颈非过拟合 | 0.746 | — | — | — | 完成 |
| CM-FA-220 | CrossModalSE 三点融合 | 0.690 | — | 0.606 | 0.734 | 完成 |
| Exp-0（进行中） | 单点 P3 FA-Concat 基准 | — | — | — | — | 进行中 |

> 评测标准统一为：`imgsz=640, conf=0.25, iou=0.75, max_det=500`，test set，官方划分。
> 详细逐类指标和混淆矩阵见 `docs/research/研究备忘录_v7.md` 第三节。

---

## 研究文档索引

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究备忘录 v7 | `docs/research/研究备忘录_v7.md` | **项目最高层级决策文档**，实验记录、结论、工程规范、时间规划 |
| 点子库与文献引用 v4 | `docs/research/点子库与文献引用_v4.md` | 所有研究方案索引（20 个点子）、分类、文献引用 [L1]-[L20] |
| 阶段复盘与注意力机制分析 v3 | `docs/research/阶段复盘与注意力机制分析_v3_2026-03-17.md` | FA-Concat vs CM-FA 对比分析、注意力实验设计、速查表 |

---

## 工程约束

以下约束来自代码库实际实现和实验验证，当前仍然生效：

| 约束 | 说明 |
|------|------|
| 初始化：`yolov8n.pt` | 新架构必须从 COCO 预训练初始化，不得用 M5 权重（迁移效果极差） |
| 模态顺序：ch1-3=RGB，ch4-6=IR | 代码约定，改动会导致静默错误 |
| 数据集：只用 `data_croped/` | 去白边版本（640×512），已完成预处理 |
| Inception 输入 `c1 % 4 == 0` | 仅约束直接输入 Inception 的张量 |
| Inception 输出 `.contiguous()` | 已实现 |
| SiLU `inplace=False` | SE/CrossModalSE 已实现；新增含 SiLU 的模块同样适用 |
| YAML 融合接口：输入 `list[Tensor,Tensor]`，输出 `[B,2C,H,W]` | 解析器要求 |
| `max_det=500` | 数据集最高 246 实例，留余量 |

**关于训练超参：** 训练超参随实验阶段推进而变化，不在此处维护。当前阶段统一超参见 `docs/research/研究备忘录_v7.md` 第 5.2 节。

---

## 目录结构

```
YOLO-Fusion-UA-Lite/
├── data/                          # 原始数据集（有白边，不用于训练）
│   ├── train/ val/ test/
│   ├── classes.txt
│   ├── mismatch_obb.txt           # 329 对已剔除的跨模态标注不一致样本记录
│   └── 数据集预处理逻辑.md
├── data_croped/                   # 裁切后数据集（去白边，640×512，用于所有训练）
│   ├── train/ val/ test/          # 各含 <subset>img/（RGB）、<subset>imgr/（IR）、<subset>labels_yolo_obb/
│   └── crop_meta.json             # 裁切元数据，统一 x100_y100_w640_h512
├── models/
│   ├── baseline/
│   │   └── IR-Only-Pretrained/    # IR-only COCO 基准模型权重
│   ├── formal/
│   │   ├── dualbackbone-easy-obb-formal6/    # 早期简单拼接基线
│   │   ├── dualbackbone-FA-Concat-obb/       # FA-Concat（无 Neck）
│   │   ├── FA-Concat-FPN-PAN-neck/           # M3：FA-Concat + Neck
│   │   ├── CM-FA-Concat-FPN-PAN-neck/        # CM-FA 正式训练
│   │   └── CM-FA-Transferred/                # CM-FA 迁移训练
│   ├── fusion-attention/          # 快速实验轮次（CM-FA quick 系列等）
│   ├── IR-YOLOv8n/
│   │   └── from_scrach/train/     # M0：IR-only 从零训练
│   └── posttrain/
│       ├── FA-Concat_FPN-PAN_tuned/          # M5：COCO 预训练迁移，当前最优
│       ├── CM-FA-Transferred/                # CM-FA 迁移实验
│       ├── CM-FA-Transferred-220/            # CM-FA-220（220 轮）
│       ├── Final_Polish_800_FlipScale/       # 高分辨率微调（已完成，非最优）
│       └── Final_Recall_640_Regularized/     # 正则化实验（M6）
├── result/                        # 各模型评测结果（JSON/CSV/混淆矩阵图）
│   ├── baseline-100epoch/
│   ├── IR-YOLOv8n/
│   ├── dualbackbone-FA-Concat-100epoch/
│   ├── FA-Concat-FPN-PAN-neck-100epoch/
│   ├── FA-Concat_FPN-PAN_tuned/
│   ├── CM-FA_FPN-PAN_neck/
│   ├── CM-FA-Transferred/
│   ├── CM-FA-Transferred2/
│   ├── Final_Polish_800_FlipScale/
│   ├── Final_Recall_640_Regularized/
│   ├── modal_dropout/
│   ├── benchmark_speed_subset.csv
│   ├── 多模型结果汇总.csv
│   ├── 训练曲线.png
│   └── radar_chart_final.png
├── src/
│   ├── cfg/
│   │   ├── datasets/
│   │   │   ├── dual_obb_dronevehicle.yaml          # 双模态数据集配置（指向 data_croped/）
│   │   │   ├── dual_obb_dronevehicle_origin.yaml   # 原始数据集配置（有白边）
│   │   │   ├── dual_obb_dronevehicle_train_plus_val.yaml
│   │   │   ├── ir_obb_dronevehicle.yaml            # IR 单模态配置
│   │   │   └── ir_obb_dronevehicle_raw.yaml
│   │   └── model/
│   │       ├── FA_Concat_FPN-PAN_neck.yaml         # M5 架构配置（当前基线）
│   │       ├── CM-FA_Concat_FPN-PAN_neck.yaml      # CM-FA 架构配置
│   │       ├── dualbackbone_FA-Concat.yaml
│   │       ├── dualbackbone_CM-FA-Concat.yaml
│   │       ├── dualbackbone_easy_obb.yaml
│   │       └── dualbackbone_fusionattention_obb.yaml
│   ├── dataset_preprocess/
│   │   ├── preprocess_obb.py       # 标签预处理：四角点→OBB 6列格式，双阶段几何匹配
│   │   ├── crop_white_borders.py   # 白边裁切（已完成，输出 data_croped/）
│   │   ├── clean_mismatch.py       # 删除 mismatch_obb.txt 记录的不一致样本
│   │   └── verify_obb_preview.py   # OBB 标签可视化预览
│   ├── testing/
│   │   ├── test_general.py         # 主评测脚本（双模态 OBB，支持 --weights/--model-name 等）
│   │   ├── test_stability.py       # 模态缺失稳定性测试
│   │   ├── view_inference.py       # 推理可视化
│   │   └── archive/                # 旧版测试脚本
│   ├── tools/
│   │   ├── build_confusion_matrix_best.py   # 生成 best epoch 混淆矩阵
│   │   ├── count_dataset_instances.py       # 数据集实例分布统计
│   │   ├── plot_single_model_metrics.py     # 单模型训练曲线绘制
│   │   ├── plot_training.py                 # 训练曲线绘制
│   │   ├── transfer_dual_weights.py         # COCO 预训练权重双主干映射迁移
│   │   └── archive/                         # 旧版工具脚本
│   └── trainning/
│       ├── train_manager.py                     # 训练任务管理器
│       ├── ir_only_pretrained_baseline.py       # IR-only COCO 基准训练（全量）
│       ├── ir_only_pretrained_baseline_quick.py # IR-only 快速验证（fraction=0.05）
│       ├── cm_fa_transfer_train.py              # CM-FA 迁移训练
│       ├── crossmodal_fusion_attention_quick_train.py  # CM-FA 快速验证
│       ├── resume_train.py                      # 断点续训
│       └── archive/                             # 旧版训练脚本
├── docs/
│   ├── research/                  # 权威研究文档（见上方索引表）
│   └── archive/                   # 归档的历史实施文档
├── ultralytics-8.2/               # 固定版本框架源码（editable install）
│   └── ultralytics/               # 实际使用的源码路径，含自定义模块修改
├── .archive/                      # 项目级归档（git 不追踪）
├── .conda/ultra82-py312/          # 项目本地 conda 环境
└── .gitignore
```

---

## 训练使用模式

### Train Manager（统一训练入口）

所有正式训练通过 `train_manager.py` 启动，而不是直接执行训练脚本。它提供：
- **自动断点续训**：检测 `last.pt` 是否存在，崩溃后自动以 resume 模式重启
- **训练完成检测**：读取 `results.csv` 中的 epoch 数，避免重复训练
- **自动重试**：异常退出后等待 `--cooldown` 秒后重试，最多 `--max-retries` 次

```bash
# 启动训练（统一用法）
python src/trainning/train_manager.py --script src/trainning/<训练脚本>.py

# 选项
# --max-retries 100    最大重试次数（默认 100）
# --cooldown 5         异常退出后冷却秒数（默认 5）
# --launch-blocking    开启 CUDA_LAUNCH_BLOCKING=1（调试用）
```

每个训练脚本通过实现 `get_train_manager_spec()` 函数对外暴露配置，包含 `train_cmd`、`resume_cmd`、`run_dir`、`total_epochs` 等字段。

**当前可用训练脚本：**

| 脚本 | 说明 |
|------|------|
| `ir_only_pretrained_baseline.py` | IR 单模态 COCO 预训练基准（全量，160 epoch） |
| `ir_only_pretrained_baseline_quick.py` | IR 单模态快速冒烟验证（fraction=0.05） |
| `cm_fa_transfer_train.py` | CM-FA 从 yolov8n.pt 迁移训练 |
| `crossmodal_fusion_attention_quick_train.py` | CM-FA 快速冒烟验证 |
| `resume_train.py` | 通用断点续训脚本（`--resume <run_dir>`） |

> **约定**：新训练脚本必须实现 `get_train_manager_spec()` 接口，才能接入 train_manager 的自动重试与完成检测。

### 模态随机失活（Modality Dropout）

Dropout 功能已集成至 `ultralytics-8.2/ultralytics/models/yolo/detect/train.py`，通过训练参数控制，不依赖外部 Hook：

| 参数 | 说明 | 当前阶段默认值 |
|------|------|--------------|
| `use_test_as_val=True` | 训练期以测试集作为验证集，best 权重对测试集最优 | **必须开启** |
| `drop_prob_rgb=0.2` | RGB 模态丢弃概率（互斥，不同时丢弃两路） | 0.2 |
| `drop_prob_ir=0.2` | IR 模态丢弃概率 | 0.2 |
| `close_dropout=16` | 训练尾期关闭 dropout 的轮次数 | 16 |

60% 概率保留双模态，20% 概率仅保留 IR，20% 概率仅保留 RGB，不会同时丢弃两路。

> 当前阶段统一超参（epochs/freeze/close_mosaic 等）见 `docs/research/研究备忘录_v7.md` 第 5.2 节。

---

## 评测脚本用法

`src/testing/test_general.py` — 双模态 OBB 评测。脚本使用 `OBBValidator` 在测试集评估，固定 `imgsz=640`、`rect=True`；参数统一由数据集 YAML 和脚本顶部路径宏控制，推荐直接修改脚本顶部路径宏后运行。

**参数列表：**
- `--device cpu|<gpu_index>`：运行设备，默认 `0`（GPU）
- `--weights <pt 或目录>`：权重文件或训练输出目录（未提供时按脚本宏自动查找）
- `--model-name <str>`：结果目录名，输出到 `result/<model-name>/`
- `--result-dir <path>`：结果输出根目录，默认 `result`
- `--test-ratio <0-1>`：测试集比例，小于 1.0 时按比例构建子集
- `--test-aug`：开启测试增强（多尺度/翻转）
- `--iou <float>`：NMS IoU 阈值，默认 `0.75`

**输出文件：**
- `result/<model-name>/<model-name>.json` 与 `.csv`
- `result/<model-name>/predictions.json`（Ultralytics 约定）

**示例命令：**
```bash
python src/testing/test_general.py --device 0 --model-name Final_Recall_640_Regularized --result-dir result
python src/testing/test_general.py --weights models/posttrain/FA-Concat_FPN-PAN_tuned --model-name M5
python src/testing/test_general.py --weights models/baseline/IR-Only-Pretrained/weights/best.pt --model-name IR-Only-COCO --device 0
python src/testing/test_general.py --device cpu --test-ratio 0.2 --model-name quick-sample
```

**数据集约定：**
- 标签格式：6 列 OBB `class cx cy w h angle`（归一化，角度弧度），参考样例 `data/train/trainlabels_yolo_obb/00001.txt`
- 模态顺序：ch1-3 = RGB（`<subset>img/`），ch4-6 = IR（`<subset>imgr/`）
- 训练阶段：`imgsz=640, rect=False`（随机打乱）
- 验证/测试：`imgsz=640, rect=True`（640×512 矩形分桶）
- 数据增强：正式训练开启 `mosaic=1.0`；**颜色增强全部关闭**（`hsv_h/s/v=0`），以保证 IR 图像数据分布的真实性；验证阶段不使用任何增强

---

## 数据集规模

DroneVehicle，清洗后（删除 329 对跨模态标注不一致样本）：

| 子集 | 清洗前 | 清洗后 |
|------|--------|--------|
| train | 17990 | **17789** |
| val | 1469 | **1445** |
| test | 8980 | **8876** |

5 类：car / truck / bus / van / freight_car。极端类别不均衡，car 占绝对多数。

---

## 架构速查

### 融合模块（`ultralytics-8.2/ultralytics/nn/modules/fusion.py`）

| 类名 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `Inception` | `[B,C,H,W]`，要求 `c1%4==0` | `[B,C,H,W]` | 多分支特征提取（1×1 / 3×3 / 5×5 / 池化） |
| `SEBlock` | `[B,C,H,W]` | `[B,C,H,W]` | 独立通道注意力（GAP → MLP → Sigmoid） |
| `FeatureAttentionConcat` | `[x_rgb, x_ir]` 各 `[B,C,H,W]` | `[B,2C,H,W]` | **M5 当前基线**：两路各自 Inception+SE 后 concat |
| `CrossModalSE` | `x_rgb, x_ir` 各 `[B,C,H,W]` | `(x_rgb_w, x_ir_w)` | 跨模态联合权重，CM-FA 内部使用 |
| `CrossModalFusionAttention` | `[x_rgb, x_ir]` | `[B,2C,H,W]` | CM-FA：Inception + 跨模态SE + concat |
| `FusionAttention` | `[x_rgb, x_ir]` | `[B,C,H,W]`（相加） | 早期加法融合，**已弃用** |

### 当前基线架构层索引（`FA_Concat_FPN-PAN_neck.yaml`，nano 规格）

YAML 配置通道（width=0.25 缩放后实际通道 / 融合后）：

| 融合点 | RGB 层索引 | IR 层索引 | nano 单路通道 | concat 后通道 |
|--------|-----------|----------|--------------|--------------|
| P3 | 7 | 17 | 64 | 128 |
| P4 | 9 | 19 | 128 | 256 |
| P5 | 12 | 22 | 256 | 512 |

Neck 输出（经 FPN+PANet 后）：Neck-P3 → 层34，Final-P4 → 层37，Final-P5 → 层40。

---

## 工程历史说明

**框架源码已修改：** `ultralytics-8.2/ultralytics/` 中的标签映射和数据集加载逻辑已经过修改以适配双模态 6 通道输入约定。若未来升级框架版本，需将这些修改迁移。

**已废弃方案（勿重复踩坑）：**
- 直接依赖 `site-packages` 中的高版本 `ultralytics 8.3.x`——版本差异导致行为不一致
- 训练阶段对库进行猴子补丁——已改为直接修改 editable install 源码

---

## 常见问题

- **"Mean of empty slice"**：检查是否用 HBB 任务读取了 OBB 标签，确保使用 OBB 任务与 `labels_yolo_obb/` 目录。
- **pin_memory 线程异常**：将 `workers` 降至 0 或 2。
