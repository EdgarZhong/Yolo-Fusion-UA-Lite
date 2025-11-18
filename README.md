# **项目名称：YOLO-Fusion-UA-Lite - 轻量级RGB-红外融合检测模型的实现与优化**

**最终目标：** 构建一个在DroneVehicle数据集上表现优异的、轻量级的YOLOv8-based融合检测模型，并探索基于不确定性感知的优化策略。

**技术栈：**
*   **环境：** mini-conda(**在powershell运行`./start-conda`文件中的命令**以启用) + python-3.13
*   **框架：** PyTorch
*   **基础模型：** YOLOv8 (推荐使用`ultralytics`库)
*   **硬件：** 支持CUDA的NVIDIA GPU
*   **核心数据集：** DroneVehicle

---

### **第一阶段：基石搭建——双路YOLO的复现与验证 (硬性目标)**

**目标：** 成功搭建一个双路输入的YOLOv8模型，使用简单的拼接融合方式，并在DroneVehicle数据集上跑通训练、验证全流程，获得基线性能。

*   **Week 1: 环境配置与数据探索**
    *   **任务1.1 - 环境搭建：**
        *   创建独立的Conda虚拟环境。
        *   安装PyTorch (CUDA版本)及`ultralytics`库 (`pip install ultralytics`)。
        *   以`./yolo-fuse`下的项目作为参考，但主要依赖`ultralytics`官方库进行修改。
    *   **任务1.2 - 数据集下载与分析：**
        *   下载DroneVehicle数据集，解压并仔细分析其目录结构。
        *   重点分析RGB图像、红外图像、RGB标注(XML)、红外标注(XML)之间的文件名对应关系。
    *   **任务1.3 - 编写数据预处理脚本 (`preprocess.py`):**
        *   **输入：** DroneVehicle原始数据集路径。
        *   **核心逻辑：**
            1.  遍历所有成对的RGB/红外图像。
            2.  读取对应的两个XML标注文件。
            3.  **标注融合策略：** 将两套标注框合并并去重（可设置一个IoU阈值判断是否为重复框），生成一套统一的标注。
            4.  将统一后的标注转换为YOLO格式的`.txt`文件。
            5.  创建并运行标签处理脚本文件，完成数据集对yolo的适配
*   **Week 2: 模型改造与首次训练**
    *   **任务2.1 - 创建自定义数据加载器 (`custom_dataset.py`):**
        *   继承`ultralytics`的`BaseDataset`类。
        *   重写`__getitem__`方法。该方法除了加载RGB图像外，还要根据RGB图像的文件名，**动态地找到并加载对应的红外图像**。
        *   将红外图像（单通道）复制成三通道，使其尺寸与RGB图像一致。
        *   **输出：** 返回一个字典，例如`{'rgb': rgb_img, 'ir': ir_img, 'labels': labels}`。
    *   **任务2.2 - 改造YOLOv8模型架构 (`custom_yolo.py`):**
        *   复制一份`ultralytics/nn/tasks.py`中的`DetectionModel`类。
        *   修改其`forward`方法，使其能够接收一个字典作为输入。
        *   **融合实现：** 参考yolo-fuse项目`./yolo-fuse`仓库，移植其各融合实现逻辑，实现不同融合模块可以通过配置文件进行切换。
    *   **任务2.3 - 配置文件与训练：**
        *   创建数据集配置文件`dronevehicle.yaml`。
        *   创建模型配置文件`yolov8-fuse-concat.yaml`，在其中指定你的自定义模型结构。
        *   使用命令行启动训练，密切监控损失曲线，确保模型能正常收敛。
    *   **产出：** 获得第一批基线模型权重和性能报告。

---

### **第二阶段：核心升级——FusionAttention模块的实现与超越 (硬性目标)**

**目标：** 实现YOLO-Fusion论文中的`FusionAttention`模块，替换简单的拼接融合，并证明其相对于基线模型的优越性。

*   **Week 3: 模块实现与集成**
    *   **任务3.1 - 编写`FusionAttention`模块 (`fusion_modules.py`):**
        *   创建一个新的Python文件，在其中定义`class FusionAttention(nn.Module):`。
        *   分步实现内部组件：并行的多尺度卷积块、SE通道注意力模块。
        *   用随机张量进行单元测试，确保模块的输入输出形状正确。
    *   **任务3.2 - 集成到YOLO模型中：**
        *   回到`custom_yolo.py`，复制一份Week 2的模型代码。
        *   将`torch.cat`操作替换为调用你实现的`FusionAttention`模块。仔细核对通道数的变化，可能需要在模块前后加入1x1卷积来调整通道数。
        *   创建新的模型配置文件`yolov8-fuse-attention.yaml`。

*   **Week 4: 对比实验与分析**
    *   **任务4.1 - 训练新模型：**
        *   使用新的配置文件，在完全相同的实验设置下（数据集、超参数等）训练`FusionAttention`模型。
    *   **任务4.2 - 性能对比与分析：**
        *   在验证集上评估新模型的性能。
        *   制作一个表格，清晰地对比`Concat-Baseline`和`Attention-Fusion`两个模型在mAP、各类AP、参数量(Params)、计算量(GFLOPs)上的差异。
        *   **可视化分析：** 找出一些基线模型检测失败（漏检、误检）而新模型成功的例子，定性地分析性能提升的原因。
    *   **产出：** 性能更强的模型权重 (`attention_fusion.pt`) 和一份详细的对比分析报告。

---

### **第三阶段：探索创新——不确定性感知优化 (选做目标)**

**目标：** 设计并实现一个轻量级的、包含可学习参数的不确定性感知模块，进一步提升模型在复杂光照条件下的鲁棒性。

*   **Week 5: 可学习调制模块的设计与实现**
    *   **任务5.1 - 理论设计 (这是创新的核心！):**
        *   **模块名称：** `Learnable Brightness Modulator (LBM)`。
        *   **输入：** RGB图像的全局特征（例如，从backbone的最后输出中进行全局平均池化得到的一个特征向量）。
        *   **结构：** 一个非常迷你的神经网络。例如：`全局平均池化 -> 全连接层1 -> ReLU -> 全连接层2 -> Sigmoid`。
        *   **输出：** 一个在(0, 1)之间的标量值，即**调制因子 (Modulation Factor)**。
        *   **原理：** 我们不再手动设定亮度规则，而是让这个迷你网络自己去**学习**“什么样的图像特征（不仅仅是亮度，还可能包括对比度、色彩分布等）”对应一个“多大的惩罚/增强因子”。Sigmoid的输出特性天然适合做权重。
    *   **任务5.2 - 模块实现与集成：**
        *   在`custom_yolo.py`中实现`LBM`模块。
        *   改造你的`FusionAttention`模型：
            1.  在`forward`中，从RGB图像的backbone输出中提取全局特征，并送入`LBM`模块，得到调制因子`mod_factor`。
            2.  **应用方式一（训练时）：** 将这个`mod_factor`乘到**最终的总损失函数**上。`loss = mod_factor * original_loss`。这会让网络在认为RGB图像质量差时，自动减小反向传播的梯度，降低学习力度。
            3.  **应用方式二（推理时）：** 就像之前讨论的，将`mod_factor`乘到**预测框的置信度**上。

*   **Week 6: 最终实验与项目总结**
    *   **任务6.1 - 训练与评估最终模型：**
        *   训练集成了`LBM`的最终模型。这可能需要调整学习率等超参数。
        *   进行最终的性能评估，与第二阶段的模型进行对比。
    *   **任务6.2 - 项目总结与报告撰写：**
        *   整理所有实验结果，制作清晰的图表。
        *   撰写一份完整的项目报告，内容包括：
            *   **引言：** 项目背景和目标。
            *   **方法：** 详细介绍你的基线模型、`FusionAttention`模型和创新的`LBM`模块。
            *   **实验：** 描述数据集、实验设置和评估指标。
            *   **结果与分析：** 展示所有实验结果，并进行深入的定量和定性分析。
            *   **结论：** 总结你的工作和未来的展望。
    *   **任务6.3 - 代码整理与归档：**
        *   将你的代码整理干净，添加注释，上传到GitHub。这是一个展示你能力和项目经验的绝佳方式。

**风险与应对：**
*   **风险1：** 数据预处理耗时过长。**应对：** 先用一小部分数据跑通脚本，确保逻辑正确，再对完整数据集进行处理。
*   **风险2：** 模型训练不收敛。**应对：** 从最简单的模型开始，检查数据加载器是否正确，调低学习率，确保梯度能够正常传播。
*   **风险3：** 创新模块效果不佳。**应对：** 这是科研的常态。即使效果没有提升，只要你能清晰地分析出可能的原因，这个探索过程本身就是有价值的。

这份计划为你铺平了道路。现在，就从第一周的任务开始，一步一个脚印地去实现它吧！祝你项目顺利，收获满满！

### 环境快速配置与GPU方案

- PowerShell 初始化 Conda：执行 `& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression`，注意**路径需根据实际安装位置调整**。
- 两种可选方案，均基于 `python 3.13`：
  - 方案A（跨设备复现，推荐）：使用 Conda 的 CUDA 运行库
    - 文件：`environment-gpu-conda.yml`
    - 创建：`conda env create -n yolo-fusion-lite -f environment-gpu-conda.yml`
    - 激活：`conda activate yolo-fusion-lite`
    - 验证：`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
  - 方案B（当前设备已验证）：在 Conda 环境中用 pip 安装 `cu124` 轮子
    - 文件：`environment-gpu-pip-cu124.yml`
    - 创建：`conda env create -p .\\.conda\\yolo-fusion-lite -f environment-gpu-pip-cu124.yml`
    - 激活：`conda activate .\\.conda\\yolo-fusion-lite`
    - 安装 GPU 轮子：`python -m pip install --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124`
    - 验证：`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

**固定版本说明**
- 为保证可复现与兼容：`ultralytics==8.3.228`、`numpy==2.2.6`、`opencv-python==4.12.0.88`、`torch/vision/audio` 与 `CUDA 12.4` 匹配
- `pytorch-cuda=12.4`（Conda 方案）或 `cu124` pip 轮子均内置 CUDA 运行库；无需系统 CUDA Toolkit（`nvcc`）
- 可用驱动要求：`nvidia-smi` 显示的 `CUDA Version` 应不低于目标运行库版本（例如 12.4）

**重要提示**
- PyTorch 官方二进制不会自动复用系统已安装的 CUDA；必须使用带 CUDA 的发行包（Conda 的 `pytorch-cuda` 或 pip 的 `cuXXX` 轮子）
- 若目标设备 Conda 求解失败或被降为 CPU 版，可改用 pip `cuXXX` 方案；跨设备时可准备 `12.1/12.4` 两份配置以适配不同驱动

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
  - `yolo-fuse/`（参考与实验脚本集）
  - `environment-gpu-conda.yml`
  - `environment-gpu-pip-cu124.yml`
  - `start-conda`
  - `.gitignore`