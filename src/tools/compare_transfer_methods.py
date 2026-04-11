"""
对比两种权重迁移方式的验证脚本
- 方式A：当时的迁移 (transfer_dual_weights.py) - 逐 Module 拷贝
- 方式B：现在的迁移 (m5_fa_concat_rerun_train.py) - state_dict key 级拷贝
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

import torch
from ultralytics.nn.tasks import DualBackboneOBBModel, attempt_load_one_weight

MODEL_CFG = ROOT / "src" / "cfg" / "model" / "FA_Concat_FPN-PAN_neck.yaml"
PRETRAINED_WEIGHTS = ROOT / "yolov8n.pt"

# 映射配置（两种方式使用相同的逻辑映射）
SINGLE_TO_DUAL_BACKBONE_MAP = {
    0: (3, 13),
    1: (4, 14),
    2: (5, 15),
    3: (6, 16),
    4: (7, 17),
    5: (8, 18),
    6: (9, 19),
    7: (10, 20),
    8: (11, 21),
    9: (12, 22),
}


def deep_copy_module_params(src: torch.nn.Module, dst: torch.nn.Module) -> int:
    """当时的迁移方式：逐 Module 深拷贝"""
    sc = src.state_dict()
    dc = dst.state_dict()
    copied = 0
    for k, v in sc.items():
        if k in dc and dc[k].shape == v.shape:
            dc[k].copy_(v)
            copied += 1
    dst.load_state_dict(dc, strict=False)
    return copied


def transfer_method_a():
    """
    方式A：当时的迁移逻辑 (transfer_dual_weights.py)
    """
    print("=" * 60)
    print("[方式A] 当时的迁移逻辑 - 逐 Module 拷贝")
    print("=" * 60)

    # 加载源权重
    src_model, _ = attempt_load_one_weight(str(PRETRAINED_WEIGHTS), device=None, inplace=True, fuse=False)

    # 构建目标模型
    tgt_model = DualBackboneOBBModel(cfg=str(MODEL_CFG), ch=6, nc=5, verbose=False)

    # 定义索引映射（与当时脚本一致）
    src_idx = list(range(0, 10))
    rgb_idx = list(range(3, 13))
    ir_idx = list(range(13, 23))

    # 逐层拷贝（RGB）
    copied_rgb = 0
    for si, ti in zip(src_idx, rgb_idx):
        copied_rgb += deep_copy_module_params(src_model.model[si], tgt_model.model[ti])
    print(f"[方式A] RGB 分支拷贝完成，复制张量数: {copied_rgb}")

    # 逐层拷贝（IR）
    copied_ir = 0
    for si, ti in zip(src_idx, ir_idx):
        copied_ir += deep_copy_module_params(src_model.model[si], tgt_model.model[ti])
    print(f"[方式A] IR 分支拷贝完成，复制张量数: {copied_ir}")

    total_copied = copied_rgb + copied_ir
    print(f"[方式A] 总计拷贝张量数: {total_copied}")

    return tgt_model, total_copied, (copied_rgb, copied_ir)


def transfer_method_b():
    """
    方式B：现在的迁移逻辑 (m5_fa_concat_rerun_train.py)
    """
    print("\n" + "=" * 60)
    print("[方式B] 现在的迁移逻辑 - state_dict key 级拷贝")
    print("=" * 60)

    # 加载源权重
    source_model, _ = attempt_load_one_weight(str(PRETRAINED_WEIGHTS))
    source_sd = source_model.float().state_dict()

    # 构建目标模型
    target_model = DualBackboneOBBModel(cfg=str(MODEL_CFG), ch=6, nc=5, verbose=False)
    target_sd = target_model.state_dict()

    copied = 0
    skipped = 0
    source_hits = 0

    for source_idx, target_pair in SINGLE_TO_DUAL_BACKBONE_MAP.items():
        source_prefix = f"model.{source_idx}."
        for key, value in source_sd.items():
            if not key.startswith(source_prefix):
                continue
            source_hits += 1
            suffix = key[len(source_prefix):]
            for target_idx in target_pair:
                target_key = f"model.{target_idx}.{suffix}"
                if target_key in target_sd and target_sd[target_key].shape == value.shape:
                    target_sd[target_key].copy_(value)
                    copied += 1
                else:
                    skipped += 1

    print(f"[方式B] source_hits={source_hits}, copied={copied}, skipped={skipped}")

    # 加载修改后的 state_dict
    target_model.load_state_dict(target_sd, strict=False)

    return target_model, copied, (source_hits, skipped)


def compare_models(model_a, model_b):
    """对比两个模型的权重是否完全一致"""
    print("\n" + "=" * 60)
    print("[对比结果] 两种迁移方式的权重差异")
    print("=" * 60)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    all_keys = set(sd_a.keys()) | set(sd_b.keys())
    common_keys = set(sd_a.keys()) & set(sd_b.keys())

    print(f"总key数: {len(all_keys)}")
    print(f"共同key数: {len(common_keys)}")
    print(f"仅在A中: {len(sd_a.keys() - sd_b.keys())}")
    print(f"仅在B中: {len(sd_b.keys() - sd_a.keys())}")

    # 对比每个key的数值
    diff_keys = []
    max_diff = 0.0

    # 只检查主干层的key
    backbone_layers = list(range(3, 13)) + list(range(13, 23))
    backbone_keys = [k for k in common_keys if any(f"model.{i}." in k for i in backbone_layers)]

    for key in common_keys:
        if sd_a[key].shape != sd_b[key].shape:
            print(f"[形状不同] {key}: A={sd_a[key].shape}, B={sd_b[key].shape}")
            continue

        if sd_a[key].dtype != torch.float32 or sd_b[key].dtype != torch.float32:
            continue

        diff = (sd_a[key] - sd_b[key]).abs().max().item()
        if diff > 1e-6:
            diff_keys.append((key, diff))
        max_diff = max(max_diff, diff)

    # 主干层差异
    backbone_diff = [(k, d) for k, d in diff_keys if any(f"model.{i}." in k for i in backbone_layers)]

    print(f"\n数值差异 > 1e-6 的key数: {len(diff_keys)}")
    print(f"其中属于主干层(3-12,13-22)的key数: {len(backbone_diff)}")
    print(f"最大数值差异: {max_diff:.2e}")

    if diff_keys:
        print("\n[差异最大的10个key]:")
        diff_keys.sort(key=lambda x: x[1], reverse=True)
        for key, diff in diff_keys[:10]:
            layer_type = "主干" if any(f"model.{i}." in key for i in backbone_layers) else "其他"
            print(f"  [{layer_type}] {key}: max_diff={diff:.2e}")

    return len(diff_keys) == 0 and max_diff < 1e-6


def analyze_key_coverage():
    """分析两种方式的key覆盖情况"""
    print("\n" + "=" * 60)
    print("[Key覆盖分析] 详细对比")
    print("=" * 60)

    source_model, _ = attempt_load_one_weight(str(PRETRAINED_WEIGHTS))
    source_sd = source_model.float().state_dict()

    # 获取源模型的主干key（0-9层）
    source_backbone_keys = set()
    for idx in range(10):
        prefix = f"model.{idx}."
        for key in source_sd.keys():
            if key.startswith(prefix):
                source_backbone_keys.add(key)

    print(f"\n源模型(yolov8n)主干层(0-9)的key数: {len(source_backbone_keys)}")

    # 方式B应该生成的目标key
    expected_target_keys = set()
    for source_idx, target_pair in SINGLE_TO_DUAL_BACKBONE_MAP.items():
        prefix = f"model.{source_idx}."
        for key in source_backbone_keys:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                for target_idx in target_pair:
                    expected_target_keys.add(f"model.{target_idx}.{suffix}")

    print(f"方式B期望生成的目标key数: {len(expected_target_keys)}")

    # 构建目标模型查看实际的key
    target_model = DualBackboneOBBModel(cfg=str(MODEL_CFG), ch=6, nc=5, verbose=False)
    target_sd = target_model.state_dict()

    # 检查期望key是否都在目标模型中
    missing_in_target = expected_target_keys - set(target_sd.keys())
    if missing_in_target:
        print(f"\n[警告] 期望key在目标模型中缺失 ({len(missing_in_target)}个):")
        for key in sorted(missing_in_target)[:5]:
            print(f"  - {key}")

    # 检查目标模型中的主干层key
    target_backbone_keys = set()
    for idx in list(range(3, 13)) + list(range(13, 23)):
        prefix = f"model.{idx}."
        for key in target_sd.keys():
            if key.startswith(prefix):
                target_backbone_keys.add(key)

    print(f"\n目标模型主干层(3-12, 13-22)的key数: {len(target_backbone_keys)}")

    # 对比
    extra_in_target = target_backbone_keys - expected_target_keys
    if extra_in_target:
        print(f"\n目标模型中额外的key ({len(extra_in_target)}个):")
        for key in sorted(extra_in_target)[:5]:
            print(f"  - {key}")


def main():
    print("开始对比两种权重迁移方式...")
    print(f"模型配置: {MODEL_CFG}")
    print(f"预训练权重: {PRETRAINED_WEIGHTS}")

    # 执行两种迁移
    model_a, total_a, stats_a = transfer_method_a()
    model_b, total_b, stats_b = transfer_method_b()

    # 对比结果
    identical = compare_models(model_a, model_b)

    # 详细分析
    analyze_key_coverage()

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print(f"方式A (当时) 拷贝张量数: {total_a}")
    print(f"方式B (现在) 拷贝张量数: {total_b}")
    print(f"拷贝数量是否一致: {'是' if total_a == total_b else '否'}")
    print(f"权重数值是否完全一致: {'是' if identical else '否'}")

    if identical:
        print("\n✓ 两种迁移逻辑等价，训练差异可能来自其他因素")
    else:
        print("\n✗ 两种迁移逻辑存在差异，可能是训练结果不同的原因")


if __name__ == "__main__":
    main()
