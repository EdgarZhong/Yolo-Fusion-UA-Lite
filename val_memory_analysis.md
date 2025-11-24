验证阶段显存溢出问题分析（GPU约20G占用）

一、现象与触发点
- 训练第 1~3 个 epoch 后，进入验证阶段（val）显存开始持续上升，最终报错 `CUDA error: unknown error`，同步时触发（参考 `ultralytics-8.2/ultralytics/utils/ops.py:61`）。
- 多次出现 `WARNING ⚠️ NMS time limit 2.600s exceeded`，表明后处理（NMS）耗时显著，候选框数量较多。

二、核心热点与内存占用来源
- 旋转 NMS 的 NxN 概率 IoU：
  - 位置：`ultralytics-8.2/ultralytics/utils/ops.py:142-160` 的 `nms_rotated`
  - 关键：调用 `batch_probiou(boxes, boxes)` 计算 NxN 的相似度矩阵，并取上三角 `triu_`，当每图候选框 N 很大（例如低阈值导致数千/上万候选），矩阵规模 O(N^2) 会造成短期内显存峰值。
  - 例如：N=10000 时，矩阵元素 1e8；FP32 约 400MB；若多步链式计算（包含 clamp、sqrt、log）在 GPU 保留中间结果，会暂时达到数 GB 级占用。

- OBB 验证匹配的 NxM 概率 IoU：
  - 位置：`ultralytics-8.2/ultralytics/models/yolo/obb/val.py:79-80` 调用 `batch_probiou(gt_bboxes, pred_rboxes)`；匹配流程在 `ultralytics-8.2/ultralytics/engine/validator.py:220-260`
  - 关键：IoU 矩阵规模为 GroundTruth 数量 × 预测数量；若预测极多，NxM 同样成倍增加显存占用。

- 精度与类型：
  - 若验证期未开启 `half`，大部分数学运算以 FP32 执行，张量体积翻倍；在 GPU 上进行 `sqrt/log` 等操作时会产生临时张量，增加峰值。

三、易致泄漏或峰值放大因素
- 多张量保留与未及时释放：
  - 若未使用 `torch.no_grad()`，或者中间张量参与 autograd 图，验证阶段虽然不反向，但也可能保留计算图引用，延迟释放。
  - 设备来回搬运：IoU 在 GPU 上计算，随后频繁 `.cpu()`/`.numpy()`（参考 `ultralytics-8.2/ultralytics/engine/validator.py:236-239`），若时机不当会在 GPU 累积内存直到下一次 GC。

四、与本仓代码路径的对应关系
- 概率 IoU实现在：`ultralytics-8.2/ultralytics/utils/metrics.py:239-276`（`batch_probiou`）
- 旋转 NMS 在：`ultralytics-8.2/ultralytics/utils/ops.py:142-160`
- OBB 验证匹配在：`ultralytics-8.2/ultralytics/models/yolo/obb/val.py:79-80` 与 `ultralytics-8.2/ultralytics/engine/validator.py:220-260`

五、问题成因总结
- 验证阶段在 GPU 上进行大规模（NxN / NxM）概率 IoU 计算与匹配，候选框数量大时瞬时显存需求急剧上升。
- 若缺少 `no_grad`/`detach`、类型为 FP32、以及设备频繁转换，会进一步放大显存峰值或延迟释放，导致见到接近 20G 的占用。

六、建议的解决方向（不改代码，仅记录）
- 限制候选框数量：适度提高 `conf_thres`，减少 NMS 输入规模；或在推理后先筛掉低分候选。
- 在验证环节显式 `no_grad` + 避免创建计算图；尽量一次性在 GPU 完成掩码相乘再转 CPU。
- 对 NxN/NxM IoU 采用分块计算（blockwise）控制峰值；或将最重的步骤（如 `sqrt/log`）在 CPU 上执行，仅保留必要张量在 GPU。
- 根据设备情况启用 `half` 验证以降低张量体积，同时评估数值稳定性。