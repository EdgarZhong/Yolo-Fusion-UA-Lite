# YOLO-Fusion-UA-Lite

**最终目标：** 构建一个在DroneVehicle数据集上表现优异的、轻量级的YOLOv8-based融合检测模型，并探索基于不确定性感知的优化策略。

**技术栈：**
*   **环境：** mini-conda（在 PowerShell 先执行 `start-conda` 中的命令以启用） + Python 3.12
*   **框架：** PyTorch
*   **基础模型：** YOLOv8（固定使用本仓库随附的 `ultralytics-8.2` 源码）
*   **硬件：** 支持 CUDA 的 NVIDIA GPU（驱动版本需匹配 12.4 运行库）
*   **核心数据集：** DroneVehicle

## **双模态有向目标检测模型开发与验证实践路径**

**核心思想：** 以官方`ultralytics`库为不可变的外部依赖，通过“猴子补丁”或自定义扩展的方式，在其之上构建并验证一个双模态融合检测模型。此路径将工程解耦与代码可维护性置于最高优先级。

**最终目标：**
1.  构建一个基于`ultralytics` YOLOv8-OBB原生功能的、可工作的双路输入基线模型。
2.  在该基线之上，实现并集成`FusionAttention`（来自一篇暂时没有给出的论文，进入二阶段后再做详解）融合模块。
3.  通过在DroneVehicle数据集上的对比实验，量化验证`FusionAttention`模块的有效性。
---

### **双模态有向目标检测模型开发与验证最终实践路径**

#### **阶段零：环境与数据准备**

1.  **环境配置：**
    *   创建并激活一个隔离的虚拟环境。
    *   通过标准包管理器安装所有必需的、不可修改的外部库依赖。

2.  **数据与标签准备：**
    *   获取原始数据集。
    *   执行一个一次性的**标签预处理脚本**。该脚本的核心逻辑必须完成：
        a. **输入：** 原始多模态标注。
        b. **几何统一：** 将所有原始的非标准几何标注（如四角点）在内存中转换为两种标准表示：一种用于鲁棒匹配（轴对齐），另一种用于精确描述（原始顶点）。
        c. **多模态合并：** 执行一个两阶段的智能合并算法。第一阶段使用低精度但快速的几何表示进行初步匹配；第二阶段对未匹配项，使用高精度的几何表示和更严格的匹配准则进行二次匹配。合并操作必须基于能保留所有几何信息的原始顶点。
        d. **最终格式化：** 将合并后的几何真值，通过最小面积包围算法，转换为目标检测框架所要求的最终参数化格式。
        e. **输出：** 生成一个与原始图像一一对应的、格式统一的最终标签集。

---

#### **阶段一：构建并验证双路OBB基线框架**

**核心思想：** 直接在官方框架的基础上进行扩展，而不是通过“猴子补丁”或自定义扩展的方式。

1.  **编写自定义模块：**
    *   **数据模块：** 实现一个能够处理双路图像输入的数据处理类。
    *   **模型模块：** 实现一个能够程序化地、根据自定义配置文件构建双主干网络模型的类。该类应继承自官方框架的模型基类，并重写其前向传播逻辑以处理双路输入。

2.  **定义基线模型配置：**
    *   创建一个自定义的配置文件。该文件需描述一个双主干网络结构，并在颈部使用最基础的拼接操作进行特征融合。此处请参考`yolo-fuse/`中的`Easy-level-Feature-Fusion.yaml`配置文件。

3.  **编写主执行脚本：**
    *   创建一个主训练脚本。
    *   **核心逻辑：**
        a. 导入并初始化官方框架的训练引擎。
        b. **组件替换：**
            i.   在运行时，将训练引擎的数据处理组件替换为自己实现的双路数据处理类的实例。
            ii.  调用自己实现的模型构建模块，根据基线配置文件创建一个自定义的双主干模型实例。
            iii. 将训练引擎的模型组件替换为这个自定义模型实例。
        c. 启动训练引擎的训练方法。

4.  **训练与评估基线：**
    *   执行主训练脚本完成训练。
    *   编写并执行一个类似的验证脚本完成评估。
    *   记录所有实验配置和最终的性能指标，作为后续对比的基准。

---

#### **阶段二：实现与验证高级融合模块**

1.  **实现融合模块：**
    *   在一个独立的自定义模块文件中，实现高级融合策略的逻辑。

2.  **修改模型构建器：**
    *   扩展第一阶段编写的模型构建模块，使其能够解析并实例化这个新的高级融合模块。

3.  **定义新模型配置：**
    *   创建一个新的配置文件，仅将融合操作的类型从基础拼接修改为新的高级融合模块。

4.  **进行对比实验：**
    *   使用新的配置文件，在与第一阶段完全相同的实验条件下，重新执行主训练脚本和验证脚本。

5.  **分析与结论：**
    *   将新模型的性能指标与基线模型进行严格的量化比较和可视化分析，得出关于高级融合模块有效性的最终结论。

## 项目核心信息

   **当前阶段：阶段一**
- 任务类型：YOLO‑OBB（旋转框检测），标签为 6 列格式 `class cx cy w h angle`（归一化，角度为弧度），样例见 `data/train/trainlabels_yolo_obb/00001.txt`
- 输入模态与顺序约定：
  - 目录命名：`img` 存放 RGB，`imgr` 存放 IR
  - 通道顺序：始终为 “1: img(RGB), 2: imgr(IR)”；
- 分辨率策略：数据集图片默认的原生矩形（840*712）裁切到 `height=704, width=832`（各边裁 4 像素，避免白边），不缩放；保持 stride=32 对齐
- 数据增强：默认禁用（mosaic/mixup/copy_paste/erasing/flip/HSV 等均关闭），保证原生分布

## 环境与激活
- 初始化 Conda（PowerShell）：
  - `& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression`
- 激活项目环境：
  - `conda activate .\\.conda\\yolo-fusion-lite`
- 版本锁定：`ultralytics==8.3.228`、`numpy==2.2.6`、`opencv-python==4.12.0.88`、`torch==2.6.0+cu124`（CUDA 12.4）

## 数据与目录结构
- 图像目录（统一约定）：
  - `data/train/trainimg/`（RGB） 与 `data/train/trainimgr/`（IR）
  - `data/val/valimg/`（RGB）   与 `data/val/valimgr/`（IR）
  - `data/test/testimg/`（RGB） 与 `data/test/testimgr/`（IR）
- 标签目录：
  - `data/<subset>/{subset}labels_yolo_obb/`（例如 `data/train/trainlabels_yolo_obb/`），与图像同名 `.txt`
- 目录快照：
  - `data/` 内含 `classes.txt`、`mismatch_obb.txt` 与 `数据集预处理逻辑.md`
  - 预处理脚本：`src/dataset_preprocess/preprocess_obb.py`、`verify_obb_preview.py`、`clean_mismatch.py`
  - 数据集预处理已完成√

## 训练入口与关键代码
- 训练脚本：`src/train/train_concat_baseline.py`
  - IR 路径推断与顺序约定（RGB 在前，IR 在后）：`src/train/train_concat_baseline.py:62-71, 85-93`
  - 矩形裁切：`src/train/train_concat_baseline.py:98-111`
  - 运行时生成数据集配置（使用 img 目录作为基准）：`src/train/train_concat_baseline.py:200-212`
  - 训练调用（矩形尺寸、禁用增强、早停）：`src/train/train_concat_baseline.py:215-249`
  - 默认模型配置：`src/configs/models/baseline_dual_fuse_obb.yaml`
- 运行时补丁：`ultralytics/__init__.py`
  - 双输入分流模块注册（ModalSplit）：`ultralytics/__init__.py:92-101`
  - 路径与通道堆叠顺序修正（RGB 在前，IR 在后）：`ultralytics/__init__.py:64-65`
  - 首层输入通道修补兜底（存在分流时跳过）：`ultralytics/__init__.py:108-118, 150-159`
- 模型配置：`src/configs/models/baseline_dual_fuse_obb.yaml`
  - 首层通过分流模块将双模态分别进入两个 3 通道主干；在 P3/P4/P5 处进行简洁融合（Concat + C2f），最终接入 OBB 检测头

## 训练与验证
 **基线模型务必存放在 `models/baseline/` 目录中**
- 快速验证（1 轮次）：
  - `python src/train/train_concat_baseline.py --model src/configs/models/baseline_dual_fuse_obb.yaml --imgsz_h 704 --imgsz_w 832 --rect True --epochs 1 --batch 16 --device 0 --workers 2 --optimizer SGD --fraction 1.0 --cache disk --patience 5`
- 完整训练（早停法）：
  - `python src/train/train_concat_baseline.py --model src/configs/models/baseline_dual_fuse_obb.yaml --imgsz_h 704 --imgsz_w 832 --rect True --epochs 100 --batch 16 --device 0 --workers 2 --optimizer SGD --fraction 1.0 --cache disk --patience 20`
- 清理缓存：
  - `Remove-Item -Force -ErrorAction SilentlyContinue .\\data\\train\\trainimg.cache, .\\data\\train\\trainimgr.cache, .\\data\\val\\valimg.cache, .\\data\\val\\valimgr.cache, .\\data\\test\\testimg.cache, .\\data\\test\\testimgr.cache`
- 权重输出：`models/baseline_obb/weights/best.pt`

## 常见问题与说明
- 若验证阶段出现“Mean of empty slice / invalid value encountered in divide”：
  - 原因是使用 HBB 检测任务读取 OBB 标签或标签路径不匹配；请确保使用 OBB 任务与 `labels_yolo_obb` 目录
- 若出现 pin_memory 线程异常：
  - 将 `workers` 降至 0/2，保证缓存与并发稳定
- 若日志显示输入为正方形：
  - 确认传参使用 `imgsz=[704,832]` 与 `rect=True`；脚本启动时会打印 `[Train][OBB] 使用矩形输入尺寸: height=704, width=832, rect=True`

## 参考与附属
- `yolo-fuse/` 目录：**作为参考与范例，不直接参与本项目训练**；其中包含多模态融合模块与配置，可用于对照研究
- 研究记录：`yolo_fuse_invest.md`（融合方法调研摘要）
- Conda 激活命令位于：`start-conda`

### 环境搭建（基于 ultralytics-8.2，推荐稳定路径）

- 初始化 Conda（PowerShell）：
  - `& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression`

- 创建并激活独立环境（Python 3.12）：
  - `conda create -y -p .\.conda\ultra82-py312 python=3.12`
  - `conda activate .\.conda\ultra82-py312`

- 安装依赖（固定在与 8.2 兼容的版本区间）：
  - `python -m pip install -q numpy==1.26.4 matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop`
  - `python -m pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124`

- 安装本地 ultralytics-8.2（可编辑模式，便于源码级修改）：
  - 进入 `ultralytics-8.2/` 目录执行：`python -m pip install -q -e . --no-deps`

- 验证 GPU：
  - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

**固定版本说明（针对 ultralytics-8.2）**
- `numpy < 2.0.0`（建议 `1.26.4`）以避免 CLI 版本校验失败
- `torch/vision` 对应 `cu124` 发行包；无需系统 CUDA Toolkit（`nvcc`）
- 驱动要求：`nvidia-smi` 的 `CUDA Version` 建议不低于 12.4

### 数据集清洗与规模变更

- 清洗依据：处理脚本在第二/第三阶段 OBB 匹配中发现跨模态类别不一致时，会记录到 `data/mismatch_obb.txt`
- 清洗脚本：`python clean_mismatch.py data` 删除记录样本的两侧图像、两份原 XML 与对应 OBB 标签
- 删除前后数据组数量（每组一个样本对）：
  - train：17990 → 17789
  - val：1469 → 1445
  - test：8980 → 8876
  - 总计剔除：329 组
- 压缩包：已重新压缩并覆盖 `data/train/trainlabels_yolo_obb.zip`、`data/val/vallabels_yolo_obb.zip`、`data/test/testlabels_yolo_obb.zip`，与清洗后的标签一致

### 项目目录扫描快照

- 根目录：
  - `data/`
    - `train/`
      - `trainlabels_yolo_obb.zip`
    - `val/`
      - `vallabels_yolo_obb.zip`
    - `test/`
      - `testlabels_yolo_obb.zip`
    - `classes.txt`
    - `mismatch_obb.txt`
    - `数据集预处理逻辑.md`
  - `src/`
    - `dataset_preprocess/`
      - `preprocess_obb.py`
      - `verify_obb_preview.py`
      - `clean_mismatch.py`
  - `ultralytics-8.2/`（随仓库附带的稳定版 YOLOv8 源码，建议以此为基）
  - `.conda/`（建议将所有项目环境创建在此目录下）
  - `start-conda`
  - `.gitignore`

### 废除项与最新实践

- 丢弃：直接依赖 `site-packages` 中的高版本 `ultralytics (8.3.x)`，以及在训练阶段对库进行猴子补丁的方案。
- 最新实践：将 `ultralytics-8.2` 作为稳定基础，按上述步骤在独立环境中以可编辑模式安装；所有代码修改围绕本地源码开展，确保可控与可复现。

### 训练与验证入口（建议）

- 读取数据与标签：保持 `data/<subset>/{subset}img` 与 `data/<subset>/{subset}imgr`，标签为 `{subset}labels_yolo_obb`（OBB 六列或多边形）
- 声明任务类型：数据 YAML 中明确写入 `task: obb` 与 `channels: 6`
- 训练命令示例（Python API，不依赖 CLI）：
  - `python -c "from ultralytics import YOLO; m=YOLO('src/configs/models/baseline_dual_fuse_obb.yaml'); m.train(data='src/configs/datasets/drone_dual_runtime.yaml', imgsz=832, epochs=100, batch=16, device=0)"`

### 清理与重启（当遇到环境不兼容）

- 卸载旧环境：
  - `conda env remove -p .\.conda\yolo-fusion-lite -y`（如存在）
  - 删除残留目录：`Remove-Item -Force -Recurse .\.conda\yolo-fusion-lite`（PowerShell）
- 重新创建：参考本节的 Python 3.12 路径重新搭建