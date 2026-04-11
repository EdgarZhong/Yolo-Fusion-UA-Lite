from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = ["m5", "exp0", "expa", "expb", "expc"]
MODEL_META = {
    "m5": {
        "display": "M5",
        "compactness": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "m5_compactness.json",
    },
    "exp0": {
        "display": "Exp-0",
        "compactness": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_compactness.json",
    },
    "expa": {
        "display": "Exp-A",
        "compactness": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expa_compactness.json",
    },
    "expb": {
        "display": "Exp-B",
        "compactness": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_compactness.json",
    },
    "expc": {
        "display": "Exp-C",
        "compactness": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_compactness.json",
    },
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_rows(payload: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for row in payload.get("rows", []):
        out[(str(row["feature_key"]), str(row["class_pair"]))] = row
    return out


def _fmt(v: float | int | None, digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{digits}f}"


def _pct_change(old: float | None, new: float | None) -> str:
    if old is None or new is None or abs(float(old)) < 1e-12:
        return "NA"
    delta = (float(new) - float(old)) / abs(float(old)) * 100.0
    return f"{delta:+.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _summary_lines(lookup: dict[tuple[str, str, str], dict]) -> list[str]:
    better_var_models = []
    better_snr_models = []
    negative_control_models = []

    expa_vc = lookup.get(("expa", "feat_fused_p3", "van:car"))
    for model_key in ["exp0", "expb", "expc", "m5"]:
        row_vc = lookup.get((model_key, "feat_fused_p3", "van:car"))
        row_cb = lookup.get((model_key, "feat_fused_p3", "car:bus"))
        if expa_vc and row_vc:
            if row_vc.get("intra_var_a") is not None and expa_vc.get("intra_var_a") is not None:
                if float(row_vc["intra_var_a"]) < float(expa_vc["intra_var_a"]):
                    better_var_models.append(MODEL_META[model_key]["display"])
            if row_vc.get("discriminative_snr") is not None and expa_vc.get("discriminative_snr") is not None:
                if float(row_vc["discriminative_snr"]) >= float(expa_vc["discriminative_snr"]):
                    better_snr_models.append(MODEL_META[model_key]["display"])
        if row_vc and row_cb:
            snr_vc = row_vc.get("discriminative_snr")
            snr_cb = row_cb.get("discriminative_snr")
            if snr_vc is not None and snr_cb is not None and float(snr_cb) > float(snr_vc):
                negative_control_models.append(MODEL_META[model_key]["display"])

    lines = []
    lines.append(
        f"- 在 `feat_fused_p3 · van:car` 上，相对 Exp-A，类内方差更低的模型："
        f"{'、'.join(better_var_models) if better_var_models else '无'}。"
    )
    lines.append(
        f"- 在同一主对比上，判别 SNR 不低于 Exp-A 的模型："
        f"{'、'.join(better_snr_models) if better_snr_models else '无'}。"
    )
    lines.append(
        f"- 负对照 `car:bus` 的 SNR 高于 `van:car` 的模型："
        f"{'、'.join(negative_control_models) if negative_control_models else '无'}。"
    )
    return lines


def build_report(output_md: Path) -> Path:
    compactness_data = {k: _read_json(v["compactness"]) for k, v in MODEL_META.items()}
    lookup: dict[tuple[str, str, str], dict] = {}
    seeds = set()
    pair_counts = set()

    for model_key, payload in compactness_data.items():
        seeds.add(int(payload.get("seed", 42)))
        pair_counts.add(int(payload.get("n_cos_pairs", 10000)))
        for key, row in _index_rows(payload).items():
            lookup[(model_key, key[0], key[1])] = row

    lines: list[str] = []
    lines.append("# 特征紧凑度与悖论机制分析报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat()}")
    lines.append(f"- 种子：{sorted(seeds)[0] if seeds else 42}")
    lines.append(f"- 类内余弦采样对数：{sorted(pair_counts)[0] if pair_counts else 10000}")
    lines.append("")

    lines.append("## 1. 核心发现摘要")
    lines.append("")
    lines.extend(_summary_lines(lookup))
    lines.append("")

    lines.append("## 2. 主表：feat_fused_p3 · van:car")
    lines.append("")
    rows = []
    for model_key in MODEL_ORDER:
        row = lookup[(model_key, "feat_fused_p3", "van:car")]
        rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt(row.get("intra_var_a")),
                _fmt(row.get("intra_var_b")),
                _fmt(row.get("cos_intra_a_mean")),
                _fmt(row.get("cos_intra_b_mean")),
                _fmt(row.get("inter_dist_l2")),
                _fmt(row.get("discriminative_snr")),
                _fmt(row.get("norm_mean_a")),
                _fmt(row.get("norm_mean_b")),
            ]
        )
    lines.extend(
        _table(
            ["模型", "intra_var(van)", "intra_var(car)", "cos_intra(van)", "cos_intra(car)", "inter_dist_L2", "SNR", "norm_mean(van)", "norm_mean(car)"],
            rows,
        )
    )
    lines.append("")

    lines.append("## 3. 主表：feat_fused_p3 · freight_car:truck")
    lines.append("")
    rows = []
    for model_key in MODEL_ORDER:
        row = lookup[(model_key, "feat_fused_p3", "freight_car:truck")]
        rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt(row.get("intra_var_a")),
                _fmt(row.get("intra_var_b")),
                _fmt(row.get("cos_intra_a_mean")),
                _fmt(row.get("cos_intra_b_mean")),
                _fmt(row.get("inter_dist_l2")),
                _fmt(row.get("discriminative_snr")),
                _fmt(row.get("norm_mean_a")),
                _fmt(row.get("norm_mean_b")),
            ]
        )
    lines.extend(
        _table(
            ["模型", "intra_var(fc)", "intra_var(truck)", "cos_intra(fc)", "cos_intra(truck)", "inter_dist_L2", "SNR", "norm_mean(fc)", "norm_mean(truck)"],
            rows,
        )
    )
    lines.append("")

    lines.append("## 4. 增量表（相对 Exp-A，feat_fused_p3 · van:car）")
    lines.append("")
    expa = lookup[("expa", "feat_fused_p3", "van:car")]
    delta_rows = []
    for key_name, field in [
        ("intra_var(van)", "intra_var_a"),
        ("cos_intra(van)", "cos_intra_a_mean"),
        ("inter_dist_L2", "inter_dist_l2"),
        ("SNR", "discriminative_snr"),
    ]:
        row = [key_name]
        base = expa.get(field)
        for model_key in ["exp0", "expb", "expc"]:
            cur = lookup[(model_key, "feat_fused_p3", "van:car")].get(field)
            if base is None or cur is None:
                row.append("NA")
            else:
                row.append(f"{float(cur) - float(base):+.4f}")
        delta_rows.append(row)
    lines.extend(_table(["指标", "Exp-0 − Exp-A", "Exp-B − Exp-A", "Exp-C − Exp-A"], delta_rows))
    lines.append("")

    lines.append("## 5. 关键对比：注意力前 vs 注意力后")
    lines.append("")
    rows = []
    for model_key in ["exp0", "expb", "expc"]:
        row_pre = lookup.get((model_key, "feat_pre_attn_ir", "van:car"))
        row_post = lookup.get((model_key, "feat_fused_p3", "van:car"))
        if row_pre is None or row_post is None:
            continue
        rows.append(
            [
                MODEL_META[model_key]["display"],
                "pre_attn_ir",
                _fmt(row_pre.get("intra_var_a")),
                _fmt(row_pre.get("intra_var_b")),
                _fmt(row_pre.get("discriminative_snr")),
            ]
        )
        rows.append(
            [
                MODEL_META[model_key]["display"],
                "fused_p3",
                _fmt(row_post.get("intra_var_a")),
                _fmt(row_post.get("intra_var_b")),
                _fmt(row_post.get("discriminative_snr")),
            ]
        )
        rows.append(
            [
                MODEL_META[model_key]["display"],
                "变化率",
                _pct_change(row_pre.get("intra_var_a"), row_post.get("intra_var_a")),
                _pct_change(row_pre.get("intra_var_b"), row_post.get("intra_var_b")),
                _pct_change(row_pre.get("discriminative_snr"), row_post.get("discriminative_snr")),
            ]
        )
    lines.extend(_table(["模型", "层", "intra_var(van)", "intra_var(car)", "SNR(van:car)"], rows))
    lines.append("")

    lines.append("## 6. 附录：全量数据")
    lines.append("")
    appendix_rows = []
    for model_key in MODEL_ORDER:
        payload = compactness_data[model_key]
        for row in payload.get("rows", []):
            appendix_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row.get("feature_key")),
                    str(row.get("class_pair")),
                    _fmt(row.get("intra_var_a")),
                    _fmt(row.get("intra_var_b")),
                    _fmt(row.get("cos_intra_a_mean")),
                    _fmt(row.get("cos_intra_b_mean")),
                    _fmt(row.get("inter_dist_l2")),
                    _fmt(row.get("inter_dist_cos")),
                    _fmt(row.get("discriminative_snr")),
                    _fmt(row.get("norm_mean_a")),
                    _fmt(row.get("norm_mean_b")),
                ]
            )
    lines.extend(
        _table(
            [
                "model",
                "feature_key",
                "class_pair",
                "intra_var_a",
                "intra_var_b",
                "cos_intra_a_mean",
                "cos_intra_b_mean",
                "inter_dist_l2",
                "inter_dist_cos",
                "snr",
                "norm_mean_a",
                "norm_mean_b",
            ],
            appendix_rows,
        )
    )
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "compactness_analysis_report.md"),
    )
    args = parser.parse_args()

    out = build_report(Path(args.output).resolve())
    print(f"[Diagnostic][CompactnessReport] done: {out}")


if __name__ == "__main__":
    main()
