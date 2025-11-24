## 阶段一工作记录（双模态 OBB，双主干模型）

### 2025-11-24 变更记录（NMS 阈值与测试脚本重构）

- 验证阶段 NMS 置信度阈值下限提升到 `0.33`，减少低分候选框，缓解 NxN/NxM 概率 IoU 峰值与耗时。
  - 位置：`ultralytics-8.2/ultralytics/models/yolo/obb/val.py:39-61`
  - **务必注意：为了更好的验证效果，请保证此项仅在训练时验证起效：验证时务必在验证代码中包含conf参数设置为0.001，以将置信度下限阈值0.4的逻辑覆盖掉**
- 禁用模型的程序化构建，强制通过配置文件或字典提供模型结构以保证可复现。
  - 位置：`ultralytics-8.2/ultralytics/nn/tasks.py:425-436, 444-452`
- 重构通用测试脚本，不保存任何推理图片，只输出总体与每类指标（JSON/CSV）。
  - 默认使用 GPU（`--device 0`），支持指定权重路径/模型名称/结果输出路径与测试集比例。
  - 位置：`src/testing/test_baseline.py:1-36, 80-136, 140-189, 324-346`
- 自动验证：
  - 已运行快速训练（1 epoch）并完成评估，结果保存至 `result/quick-baseline`。
  - 已使用正式权重目录 `models/formal/baseline/dualbackbone-easy-obb-formal-3` 完成推理评估，结果保存至 `result/formal-3`。

示例命令：

```bash
# 快速训练（1 epoch, fraction=0.05）
python src/trainning/baseline_quick_train.py

# 使用快速训练权重评估（测试集 10%）
python src/testing/test_general.py --device 0 \
  --weights models/baseline/dualbackbone-easy-obb-baseline11 \
  --model-name quick-baseline --result-dir result --test-ratio 0.1

# 使用正式权重目录评估（测试集 20%）
python src/testing/test_general.py --device 0 \
  --weights models/formal/baseline/dualbackbone-easy-obb-formal-3 \
  --model-name formal-3 --result-dir result --test-ratio 0.2
```

评估产物：
- `result/<model_name>.json` 与 `result/<model_name>.csv`（含总体与每类指标）
- `result/<model_name>/predictions.json`（验证器自动生成）

---

- 日期：2025-11-19
- 目标：在 `ultralytics-8.2/ultralytics` 官方源码基础上，完成双模态数据加载（RGB+IR，6 通道）与双主干 OBB 模型的构建与训练脚本，实现快速验证与正式训练所需的基础能力。

### 数据模块改造
- 新增双路数据集类 `YOLODualDataset`，约定目录结构下自动配对 RGB 与 IR：
  - 类定义：`ultralytics-8.2/ultralytics/data/dataset.py:283`
  - 自动配对与路径列表（支持 'rgb|ir' 条目）：`ultralytics-8.2/ultralytics/data/dataset.py:296-349`
  - 标签路径映射与缓存加载：`ultralytics-8.2/ultralytics/data/dataset.py:351-395`
  - 构建 OBB 标签缓存（6 列：class cx cy w h angle），并将旋转框转换为多边形段：`ultralytics-8.2/ultralytics/data/dataset.py:396-436`
  - 解析 'rgb|ir' 与 OBB 标签的校验逻辑：`ultralytics-8.2/ultralytics/data/dataset.py:438-492`
- 目录约定与行为：
  - 训练集：`data/train/trainimg/`（RGB）与同级 `trainimgr/`（IR）
  - 验证集：`data/val/valimg/` 与 `valimgr/`
  - 测试集：`data/test/testimg/` 与 `testimgr/`
  - 标签：`data/<subset>/{subset}labels_yolo_obb/` 与图像同名 `.txt`
- 数据集选择逻辑：当传入目录名为 `trainimg/valimg/testimg` 时自动使用 `YOLODualDataset`：`ultralytics-8.2/ultralytics/data/build.py:84-115`

### 图像读取与缓存（6 通道）
- `load_image` 增强，支持 'rgb|ir' 输入串并在通道维拼接为 6 通道；IR 缺失时按零阵填充：`ultralytics-8.2/ultralytics/data/base.py:157-189`
- 读取后的图像尺寸处理与缓冲逻辑：`ultralytics-8.2/ultralytics/data/base.py:192-210`
- 构建 `.npy` 加速缓存时，将 'rgb|ir' 合并保存为 6 通道数组：`ultralytics-8.2/ultralytics/data/base.py:230-247`

### 变换管线与像素处理
- LetterBox 在通道数大于 4（例如 6 通道）时使用 `np.pad` 做常量填充，避免 `cv2.copyMakeBorder` 的通道限制：`ultralytics-8.2/ultralytics/data/augment.py:1587-1593`
- `_format_img` 对 6 通道图像只对前 3 通道（RGB）进行 BGR↔RGB 通道翻转，IR 通道保持不变：`ultralytics-8.2/ultralytics/data/augment.py:2063-2099`

### 模型模块改造（双主干 OBB）
- 新增基础模块：
  - `IdentityInput`：恒等输入，占位/分支起点：`ultralytics-8.2/ultralytics/nn/modules/block.py:699-712`
  - `ModalitySelector`：从 6 通道输入选择指定模态（RGB=1 / IR=2）：`ultralytics-8.2/ultralytics/nn/modules/block.py:715-749`
  - 在模块导出中注册上述组件：`ultralytics-8.2/ultralytics/nn/modules/__init__.py:45-46,147-148`
- `parse_model` 解析器增强：识别 `ModalitySelector` 并计算输出通道（6→3），正确传递参数：`ultralytics-8.2/ultralytics/nn/tasks.py:1148-1191`（关键分支：`1166-1171`）
- 双主干 OBB 模型类：
  - `DualBackboneOBBModel` 类定义与程序化构建逻辑（两路主干，P3/P4/P5 基础拼接 + C2f 规整，接 OBB 头）：`ultralytics-8.2/ultralytics/nn/tasks.py:402-536`
  - 基于官方 `yolov8-obb.yaml` 自动生成双主干配置的实现说明：`ultralytics-8.2/ultralytics/nn/tasks.py:442-524`
- 训练器接入：
  - OBB 训练器默认使用双主干模型并设置 `ch=6`：`ultralytics-8.2/ultralytics/models/yolo/obb/train.py:25-37`
  - 验证器在 warmup 阶段对 OBB 任务使用 6 通道输入：`ultralytics-8.2/ultralytics/engine/validator.py:155-158`

### 配置文件（数据与模型）
- 数据集配置（DroneVehicle）：`src/cfg/datasets/dual_obb_dronevehicle.yaml:1-31`
  - 指向 RGB 目录；IR 目录由数据集类自动推断为同级 `*imgr/`
  - 通道顺序：`RGB(前3)` + `IR(后3)`；标签格式：`class cx cy w h angle`（归一化坐标，角度为弧度）
- 模型配置（Easy-level Feature Fusion）：`src/cfg/model/dualbackbone_easy_obb.yaml:1-74`
  - 前端使用 `IdentityInput` + 两个 `ModalitySelector` 切分模态（`22-31`）
  - 两套对称主干，P3/P4/P5 在颈部做 `Concat + C2f` 基础融合，接三尺度 OBB 头（`56-74`）

### 训练脚本与策略
- 快速验证（真实数据，禁用增强）：`src/trainning/baseline_quick_train.py:27-74`
  - 训练 1 epoch，`fraction=0.05`，`imgsz=832`，训练阶段 `rect=False`，验证测试阶段矩形评估由内部逻辑启用。
  - 输出路径：`models/baseline/dualbackbone-easy-obb-baseline/`（打印与参数见 `76-81`）。
- 正式训练脚本（宏定义集中）：`src/trainning/train_formal.py:19-52,74-110`
  - 顶部统一宏：`EPOCHS/BATCH/WORKERS/DEVICE/FRACTION/IMG_SIZE/RECT_TRAIN/PATIENCE` 等（`19-52`）。
  - 训练策略：训练 `rect=False`、随机打乱；验证/测试 `imgsz=832 + rect=True`；禁用所有增强（可通过宏显式控制）
  - 早停：`patience=10`（`74-110`）。
  - 注意：`save_dir` 打印中包含反斜杠转义（`118`），不影响功能，后续可统一为正斜杠以消除警告。

### 数据输入策略确认（与实现对齐）
- 训练：`imgsz=832`、`rect=False`、`shuffle=True`、禁用 `mosaic` 等增强；不做裁切与额外标签调整。
- 验证/测试：`imgsz=832`、`rect=True`；无额外操作。
- 真实数据集优先：已移除样例数据集的回退逻辑，`baseline_quick_train.py` 直接使用 `dual_obb_dronevehicle.yaml`。

### 关键接口与约束
- 模态顺序固定：1 为 RGB（前 3 通道），2 为 IR（后 3 通道）。数据集在 `load_image` 中将两路合并为 6 通道，模型前端用 `ModalitySelector` 切分。
- 标签格式固定：6 列 OBB（`class cx cy w h angle`），在缓存中转换为多边形段以便后续绘制与评估。
- 解析器适配：`parse_model` 正确处理 `ModalitySelector` 的通道输出，使双主干与融合层的通道数对齐。

### 文件改动汇总
- 数据模块：
  - `ultralytics-8.2/ultralytics/data/dataset.py`（新增 `YOLODualDataset` 与 OBB 缓存解析）
  - `ultralytics-8.2/ultralytics/data/build.py`（根据目录名自动选择 `YOLODualDataset`）
  - `ultralytics-8.2/ultralytics/data/base.py`（支持 'rgb|ir' 输入与 6 通道 `.npy` 缓存）
  - `ultralytics-8.2/ultralytics/data/augment.py`（LetterBox 通道>4 的填充与 6 通道 BGR 处理）
- 模型模块：
  - `ultralytics-8.2/ultralytics/nn/modules/block.py`（新增 `IdentityInput`、`ModalitySelector`）
  - `ultralytics-8.2/ultralytics/nn/modules/__init__.py`（模块导出注册）
  - `ultralytics-8.2/ultralytics/nn/tasks.py`（新增 `DualBackboneOBBModel`；`parse_model` 适配）
  - `ultralytics-8.2/ultralytics/models/yolo/obb/train.py`（训练器默认实例化双主干 OBB 模型）
  - `ultralytics-8.2/ultralytics/engine/validator.py`（OBB 任务 warmup 使用 6 通道）
- 配置与脚本：
  - `src/cfg/datasets/dual_obb_dronevehicle.yaml`（双路目录、通道顺序与标签格式说明）
  - `src/cfg/model/dualbackbone_easy_obb.yaml`（双主干 + 基础融合的模型配置）
  - `src/trainning/baseline_quick_train.py`（快速验证脚本，真实数据）
  - `src/trainning/train_formal.py`（正式训练脚本，宏定义与早停）

---

### Git 分支修复与工作区恢复流程记录-11.19

**场景描述：**

当本地 `main` 分支与远程 `origin/main` 分支出现历史分叉（Diverged History）时，如何安全地用本地的稳定版本覆盖远程的错误版本，并同时保留在 `stash` 中的、基于错误版本进行的后续开发工作。

**问题根源：**

-   **本地 `main`**：拥有一个稳定、正确的提交（例如 `CPU方法解决验证爆显存问题`）。
-   **远程 `origin/main`**：拥有一个不想要、有问题的提交（例如 `尝试修复验证阶段爆显存问题`）。
-   **开发中的工作**：基于远程的错误提交进行了一些新的修改，并存放在 `stash` 中。

直接 `push` 会失败，直接 `pull` 会产生不必要的合并，污染历史记录。

**解决方案（三步法）：**

1.  **强制推送，修复主干**：使用更安全的强制推送命令，将本地正确的 `main` 分支状态覆盖到远程。这会重写远程历史，使其与本地保持一致。
    ```bash
    git push --force-with-lease origin main
    ```
    *结果：远程 `main` 分支被修复，恢复到已知的稳定状态。*

2.  **创建特性分支，恢复工作**：将在 `stash` 中的工作恢复到一个专门的特性分支上，以隔离开发。
    ```bash
    # 检查 stash 列表
    git stash list
    # 从 stash 创建新分支并应用修改
    git stash branch <新分支名> # 例如：feature/gpu-validation-fix
    ```
    *结果：所有在 `stash` 中的修改都被安全地转移到一个新的、隔离的分支上，`stash` 列表被清空。*

3.  **变基，对齐历史**：将新创建的特性分支变基（Rebase）到修复后的 `main` 分支上，确保提交历史呈线性，便于未来合并。
    ```bash
    # 切换到新分支（上一步已自动切换）
    # git switch feature/gpu-validation-fix
    # 执行变基操作
    git rebase main
    ```
    *结果：特性分支的起点更新为 `main` 分支的最新提交，历史记录变得干净、线性。*

**最终状态：**

-   `main` 分支在本地和远程都处于稳定、可用的状态。
-   所有后续的开发工作都在一个独立的特性分支上进行，与主干完全隔离，可以安全地继续开发。

如需查看具体实现，请按上述代码引用定位到文件与行号（`file_path:line_number`）
