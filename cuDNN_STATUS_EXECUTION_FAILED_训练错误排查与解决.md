# 训练时出现“cuDNN error: CUDNN_STATUS_EXECUTION_FAILED”排查与解决指南

本文档针对在训练过程中于 `Conv2d -> F.conv2d -> cuDNN` 处崩溃的错误进行系统分析与提供可操作的解决方案。

## 症状与典型堆栈
- 触发位置一般位于 `torch.nn.Conv2d` 前向（`ultralytics-8.2/ultralytics/nn/modules/conv.py`），栈中常见调用链：
  - `DetectionModel.train -> model(batch)`
  - `block.Conv.forward -> F.conv2d`
  - 最终抛出 `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`

## 可能原因（按发生概率排序）
- GPU 显存不足或瞬时碎片化导致内核执行失败（即使未显式报 OOM，也可能以 cuDNN 执行失败表现）。
- CUDA/cuDNN/PyTorch 版本与显卡驱动不匹配，算法选择或内核调度异常。
- AMP/FP16 下某些算子数值不稳定（过大/NaN），引发低层内核异常。
- 算法选择不稳定（`cudnn.benchmark` 开启时），在特定输入尺寸触发不兼容实现。
- 罕见：张量形状越界或非法（通常会有更明确的 size mismatch 报错，但也可能被掩盖为执行失败）。

## 快速定位建议（不修改业务逻辑）
- 确认环境版本：
  - 运行 `nvidia-smi` 检查驱动版本与显存占用。
  - 运行：
    - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"`
  - 推荐：驱动版本与本地 CUDA 兼容，PyTorch 与 CUDA/cuDNN版本对应官方发行矩阵。
- 打开更精确的错误定位：
  - 设置环境变量：`CUDA_LAUNCH_BLOCKING=1`（阻塞式执行，报错定位更准确）。
  - 在训练入口临时加入：
    - `torch.autograd.set_detect_anomaly(True)`（梯度异常检测）。
- 排除 cuDNN 特定实现问题：
  - 临时关闭：`torch.backends.cudnn.enabled = False`，若错误消失，表明为 cuDNN 内核相关问题。
  - 关闭基准算法选择：`torch.backends.cudnn.benchmark = False`，避免不稳定算法切换。
- 排除数值或显存压力：
  - 临时关闭混合精度/半精度：将训练参数中的 `half=False` 或关闭 AMP。
  - 降低 `imgsz`、`batch_size`（如从 16/8 下调到 4/2），观察是否稳定。

## 建议的修复路径（从易到难）
1. 降低显存压力：
   - 调整训练超参：`batch_size` 下调、`imgsz` 适当降低；必要时减少 `workers`。
   - 监控显存：`nvidia-smi -l 1` 持续观察显存曲线是否在崩溃前达到峰值。
2. 稳定数值精度：
   - 关闭 AMP/半精度：在 Ultralytics 训练脚本中关闭 `half`，以 `FP32` 检查是否稳定。
   - 启用 TF32（Ampere+）：
     - `torch.backends.cuda.matmul.allow_tf32 = True`
     - `torch.backends.cudnn.allow_tf32 = True`
3. 固定 cuDNN 行为：
   - 强制：`torch.backends.cudnn.benchmark = False`，避免在不同尺寸/批次切换算法。
   - 必要时禁用 cuDNN（仅用于定位或救急）：`torch.backends.cudnn.enabled = False`（速度会下降）。
4. 校验软件栈与驱动：
   - 升级显卡驱动到与 CUDA 版本兼容的推荐版本。
   - 重新安装与当前 Python 版本匹配的 PyTorch 发行包（确保 `torch.version.cuda` 与本机 CUDA 一致）。

## 与本仓库改动的相关性评估
- 当前 `CrossModalFusionAttention`/`FeatureAttentionConcat` 使用标准 `Conv2d` 与 `MaxPool2d`，无 `groups` 或深度可分离卷积特殊配置，形态合法，通常不会单独引发 cuDNN 执行失败。
- 由于 `Concat` 后通道数成倍增长，后续 `Conv2d` 的特征图更宽，显存占用显著上升，更易触发显存压力相关问题。若错误在加入 CM‑FA‑Concat 或完整 FPN‑PAN 颈部后出现，优先考虑显存与数值稳定性。

## 实操排查步骤（按序执行）
1. 基线回归：在同数据与脚本下，切回不含 CM‑FA‑Concat 的配置，验证是否稳定；若稳定，说明新增模块提高了显存/数值压力。
2. 下调超参：将 `batch_size` 与 `imgsz` 同步下调一个档位，观察是否恢复；同时关闭 AMP 验证是否与半精度相关。
3. 固定 cuDNN：在训练入口添加如下调试代码（仅定位，不长期保留）：
   ```python
   # 在训练主函数最前部加入（调试用）
   import torch
   torch.backends.cudnn.enabled = True  # 先明确开启
   torch.backends.cudnn.benchmark = False  # 禁用基准算法选择，避免不稳定切换
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   # 更精确定位
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   torch.autograd.set_detect_anomaly(True)
   ```
4. 版本核查：记录 `torch`、`cuda`、`cudnn`、驱动版本，依据官方兼容矩阵进行升级或降级至稳定组合。

## 验证与回归标准
- 在上述调整下，训练不再崩溃，且 `results.csv` 能持续写入。
- 逐步恢复原始超参（如 AMP/half、较大的 `batch_size`），确认崩溃不复现。
- 如仅在 AMP 下触发，考虑长期保留 `half=False` 或在关键算子处规避半精度（需要进一步代码层面实施，本文不做实施）。

## 备注
- 本文仅提供排查与修复建议，不对仓库代码进行变更实施。若需将上述调试代码固化到训练脚本，请另行提出实施需求。

