import json
from pathlib import Path

root = Path("d:/Code/DeepLearning/YOLO-Fusion-UA-Lite/src/diagnostic/outputs/reports")
lin = json.loads((root / "m5_linear_probe.json").read_text(encoding="utf-8"))
fis = json.loads((root / "m5_fisher_ratio.json").read_text(encoding="utf-8"))
se = json.loads((root / "m5_se_analysis.json").read_text(encoding="utf-8"))
rgb = Path("d:/Code/DeepLearning/YOLO-Fusion-UA-Lite/result/dataset_probe/rgb_l_channel/rgb_l_summary.txt").read_text(encoding="utf-8")
ir = Path("d:/Code/DeepLearning/YOLO-Fusion-UA-Lite/result/dataset_probe/ir_van_car_regions/ir_van_car_ttest_summary.txt").read_text(encoding="utf-8")


def rows_for(rows, pair):
    return [r for r in rows if r.get("class_pair") == pair]


lines = ["# M5 特征诊断摘要（双路注意力版）", "", "## 数据层"]
lines += [f"- RGB统计: {x}" for x in rgb.strip().splitlines()]
lines += [f"- IR统计: {x}" for x in ir.strip().splitlines()]
lines += ["", "## 线性探针（van:car）"]
for r in sorted(rows_for(lin["rows"], "van:car"), key=lambda z: z["feature_key"]):
    lines.append(f"- {r['feature_key']}: {r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
lines += ["", "## Fisher top-10（van:car）"]
for r in sorted(rows_for(fis["rows"], "van:car"), key=lambda z: z["feature_key"]):
    lines.append(f"- {r['feature_key']}: {r['top10_mean']:.6f}")
lines += ["", "## SE 相关（van:car，双路）"]
for r in sorted(rows_for(se["rows"], "van:car"), key=lambda z: (z["feature_key"], z.get("attn_source", ""))):
    lines.append(
        f"- {r['feature_key']} | {r.get('attn_source', 'na')}: r={r['pearson_r']:.6f}, p={r['p_value']:.3g}"
    )

out = root / "m5_diagnostic_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
