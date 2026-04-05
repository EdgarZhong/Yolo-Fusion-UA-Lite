from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAIRS = ["van:car", "freight_car:truck", "car:bus"]
MODEL_META = {
    "m5": {
        "display": "M5 (三点参考)",
        "module": "FeatureAttentionConcat ×3",
        "attn": "SE",
        "eval_csv": ROOT / "result" / "FA-Concat_FPN-PAN_tuned" / "FA-Concat_FPN-PAN_tuned.csv",
        "linear": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "m5_linear_probe.json",
        "fisher": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "m5_fisher_ratio.json",
        "silhouette": None,
        "se": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "m5_se_analysis.json",
        "approx": None,
    },
    "exp0": {
        "display": "Exp-0",
        "module": "FeatureAttentionConcat",
        "attn": "SE",
        "eval_csv": ROOT / "result" / "exp0_eval" / "exp0_eval.csv",
        "linear": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_linear_probe.json",
        "fisher": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_fisher_ratio.json",
        "silhouette": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_silhouette.json",
        "se": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_se_analysis.json",
        "approx": None,
    },
    "expa": {
        "display": "Exp-A",
        "module": "InceptionConcat",
        "attn": "无",
        "eval_csv": ROOT / "result" / "expa_eval" / "expa_eval.csv",
        "linear": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expa_linear_probe.json",
        "fisher": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expa_fisher_ratio.json",
        "silhouette": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expa_silhouette.json",
        "se": None,
        "approx": None,
    },
    "expb": {
        "display": "Exp-B",
        "module": "InceptionCoordAttnConcat",
        "attn": "CoordAttn",
        "eval_csv": ROOT / "result" / "expb_eval" / "expb_eval.csv",
        "linear": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_linear_probe.json",
        "fisher": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_fisher_ratio.json",
        "silhouette": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_silhouette.json",
        "se": None,
        "approx": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_attention_analysis.json",
    },
    "expc": {
        "display": "Exp-C",
        "module": "InceptionSimAMConcat",
        "attn": "SimAM",
        "eval_csv": ROOT / "result" / "expc_eval" / "expc_eval.csv",
        "linear": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_linear_probe.json",
        "fisher": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_fisher_ratio.json",
        "silhouette": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_silhouette.json",
        "se": None,
        "approx": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_attention_analysis.json",
    },
}


def _read_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_csv(path: Path) -> dict:
    metrics = {}
    classes = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    mode = "metrics"
    for row in rows:
        if not row:
            continue
        if row[0] == "metric":
            continue
        if row[0] == "class":
            mode = "classes"
            continue
        if mode == "metrics":
            metrics[row[0]] = float(row[1])
        else:
            classes[row[0]] = {
                "precision": float(row[1]),
                "recall": float(row[2]),
                "ap50": float(row[3]),
                "ap": float(row[4]),
            }
    return {"metrics": metrics, "classes": classes}


def _index_rows(payload: dict | None, key_fields: list[str]) -> dict[tuple[str, ...], dict]:
    out = {}
    if payload is None:
        return out
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    for row in rows:
        out[tuple(str(row[k]) for k in key_fields)] = row
    return out


def _fmt_float(v: float, digits: int = 4) -> str:
    return f"{float(v):.{digits}f}"


def _fmt_acc(row: dict | None) -> str:
    if row is None:
        return "NA"
    return f"{float(row['acc_mean']):.4f}±{float(row['acc_std']):.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _build_delta_rows(model_order: list[str], lookup: dict[tuple[str, str, str], dict], value_key: str, digits: int) -> list[list[str]]:
    expa_vals = {
        pair: float(lookup[("expa", "feat_fused_p3", pair)][value_key])
        for pair in PAIRS
        if ("expa", "feat_fused_p3", pair) in lookup
    }
    rows = []
    for pair in PAIRS:
        base = expa_vals.get(pair)
        if base is None:
            continue
        row = [pair]
        for model_key in model_order:
            cur = lookup.get((model_key, "feat_fused_p3", pair))
            row.append("NA" if cur is None else f"{float(cur[value_key]) - base:+.{digits}f}")
        rows.append(row)
    return rows


def _load_layer_validation(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("rows", []))


def _load_tsne_summary(path: Path | None) -> tuple[dict[tuple[str, str, int], dict], str]:
    if path is None or not path.exists():
        return {}, "未生成 t-SNE 稳定性摘要。"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {(str(r["model"]), str(r["pair"]), int(r["perplexity"])): r for r in data.get("rows", [])}
    messages = []
    for model_key in ["Exp-0", "Exp-A", "Exp-B", "Exp-C"]:
        for pair in ["van:car", "freight_car:truck"]:
            r15 = rows.get((model_key, pair, 15))
            r30 = rows.get((model_key, pair, 30))
            r50 = rows.get((model_key, pair, 50))
            if not (r15 and r30 and r50):
                continue
            stable = True
            for other in [r15, r50]:
                sil_ratio = abs(float(other["silhouette_2d"])) / max(1e-9, abs(float(r30["silhouette_2d"])))
                sep_ratio = float(other["sep_ratio"]) / max(1e-9, float(r30["sep_ratio"]))
                if sil_ratio < 0.5 or sil_ratio > 2.0 or sep_ratio < 0.5 or sep_ratio > 2.0:
                    stable = False
            messages.append(f"{model_key} {pair}: {'通过' if stable else '存在差异'}")
    summary_text = "；".join(messages) if messages else "未生成可判定的稳定性摘要。"
    return rows, summary_text


def build_report(output_md: Path, layer_validation: Path | None, tsne_summary: Path | None) -> Path:
    eval_data = {k: _read_eval_csv(v["eval_csv"]) for k, v in MODEL_META.items()}
    linear_data = {k: _read_json(v["linear"]) for k, v in MODEL_META.items()}
    fisher_data = {k: _read_json(v["fisher"]) for k, v in MODEL_META.items()}
    silhouette_data = {k: _read_json(v["silhouette"]) for k, v in MODEL_META.items()}
    se_data = {k: _read_json(v["se"]) for k, v in MODEL_META.items()}
    approx_data = {k: _read_json(v["approx"]) for k, v in MODEL_META.items()}

    linear_lookup = {}
    fisher_lookup = {}
    silhouette_lookup = {}
    for model_key, payload in linear_data.items():
        for key, row in _index_rows(payload, ["feature_key", "class_pair"]).items():
            linear_lookup[(model_key, key[0], key[1])] = row
    for model_key, payload in fisher_data.items():
        for key, row in _index_rows(payload, ["feature_key", "class_pair"]).items():
            fisher_lookup[(model_key, key[0], key[1])] = row
    for model_key, payload in silhouette_data.items():
        for key, row in _index_rows(payload, ["feature_key", "class_pair"]).items():
            silhouette_lookup[(model_key, key[0], key[1])] = row

    lines: list[str] = []
    lines.append("# 四模型特征诊断对比报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat()}")
    lines.append("- 全局种子：42")
    lines.append("- 数据集：DroneVehicle test set（8876 图，176842 实例）")
    lines.append("")
    lines.append("## 1. 实验概览")
    lines.append("")

    overview_rows = []
    for model_key in ["m5", "exp0", "expa", "expb", "expc"]:
        meta = MODEL_META[model_key]
        ev = eval_data[model_key]
        metrics = ev["metrics"]
        classes = ev["classes"]
        overview_rows.append(
            [
                meta["display"],
                meta["module"],
                meta["attn"],
                _fmt_float(metrics["metrics/mAP50(B)"]),
                _fmt_float(metrics["metrics/mAP50-95(B)"]),
                _fmt_float(classes["van"]["ap50"]),
                _fmt_float(classes["van"]["recall"]),
                _fmt_float(classes["freight_car"]["ap50"]),
                _fmt_float(classes["truck"]["ap50"]),
            ]
        )
    lines.extend(
        _table(
            ["模型", "P3 融合模块", "注意力类型", "mAP50", "mAP50-95", "van mAP50", "van Recall", "fc mAP50", "truck mAP50"],
            overview_rows,
        )
    )
    lines.append("")

    lines.append("## 2. 线性探针（主表：feat_fused_p3）")
    lines.append("")
    linear_main_rows = []
    for model_key in ["m5", "exp0", "expa", "expb", "expc"]:
        linear_main_rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt_acc(linear_lookup.get((model_key, "feat_fused_p3", "van:car"))),
                _fmt_acc(linear_lookup.get((model_key, "feat_fused_p3", "freight_car:truck"))),
                _fmt_acc(linear_lookup.get((model_key, "feat_fused_p3", "car:bus"))),
            ]
        )
    lines.extend(_table(["模型", "van:car acc±std", "fc:truck acc±std", "car:bus acc±std"], linear_main_rows))
    lines.append("")
    lines.append("### 2.1 增量表（相对 Exp-A）")
    lines.append("")
    lines.extend(
        _table(
            ["指标", "Exp-0 − Exp-A", "Exp-B − Exp-A", "Exp-C − Exp-A"],
            _build_delta_rows(["exp0", "expb", "expc"], linear_lookup, "acc_mean", 4),
        )
    )
    lines.append("")

    lines.append("## 3. Fisher Ratio（主表：feat_fused_p3, top-10 mean）")
    lines.append("")
    fisher_main_rows = []
    for model_key in ["m5", "exp0", "expa", "expb", "expc"]:
        fisher_main_rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt_float(fisher_lookup[(model_key, "feat_fused_p3", "van:car")]["top10_mean"]),
                _fmt_float(fisher_lookup[(model_key, "feat_fused_p3", "freight_car:truck")]["top10_mean"]),
                _fmt_float(fisher_lookup[(model_key, "feat_fused_p3", "car:bus")]["top10_mean"]),
            ]
        )
    lines.extend(_table(["模型", "van:car", "fc:truck", "car:bus"], fisher_main_rows))
    lines.append("")
    lines.append("### 3.1 增量表（相对 Exp-A）")
    lines.append("")
    lines.extend(
        _table(
            ["指标", "Exp-0 − Exp-A", "Exp-B − Exp-A", "Exp-C − Exp-A"],
            _build_delta_rows(["exp0", "expb", "expc"], fisher_lookup, "top10_mean", 4),
        )
    )
    lines.append("")

    lines.append("## 4. Silhouette Score（feat_fused_p3）")
    lines.append("")
    silhouette_main_rows = []
    for model_key in ["exp0", "expa", "expb", "expc"]:
        silhouette_main_rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt_float(silhouette_lookup[(model_key, "feat_fused_p3", "van:car")]["silhouette"]),
                _fmt_float(silhouette_lookup[(model_key, "feat_fused_p3", "freight_car:truck")]["silhouette"]),
                _fmt_float(silhouette_lookup[(model_key, "feat_fused_p3", "car:bus")]["silhouette"]),
            ]
        )
    lines.extend(_table(["模型", "van:car", "fc:truck", "car:bus"], silhouette_main_rows))
    lines.append("")
    lines.append("### 4.1 增量表（相对 Exp-A）")
    lines.append("")
    lines.extend(
        _table(
            ["指标", "Exp-0 − Exp-A", "Exp-B − Exp-A", "Exp-C − Exp-A"],
            _build_delta_rows(["exp0", "expb", "expc"], silhouette_lookup, "silhouette", 4),
        )
    )
    lines.append("")

    lines.append("## 5. SE 权重相关性")
    lines.append("")
    lines.append("### 5.1 主表（feat_fused_p3）")
    lines.append("")
    se_rows = []
    for model_key in ["m5", "exp0"]:
        payload = se_data[model_key]
        if payload is None:
            continue
        for row in payload["rows"]:
            if row["feature_key"] != "feat_fused_p3":
                continue
            se_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["class_pair"]),
                    str(row["attn_source"]),
                    _fmt_float(row["pearson_r"]),
                    f"{float(row['p_value']):.4g}",
                ]
            )
    lines.extend(_table(["模型", "class_pair", "attn_source", "pearson_r", "p_value"], se_rows))
    lines.append("")

    lines.append("### 5.2 Exp-B/C 近似注意力分析")
    lines.append("")
    approx_rows = []
    for model_key in ["expb", "expc"]:
        payload = approx_data[model_key]
        if payload is None:
            continue
        for row in payload["rows"]:
            approx_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["class_pair"]),
                    str(row["attn_source"]),
                    _fmt_float(row["pearson_r"]),
                    f"{float(row['p_value']):.4g}",
                    f"{float(row['finite_ratio']):.4f}",
                    f"{float(row['out_of_unit_ratio']):.4f}",
                    "†近似值：output/(input+ε)",
                ]
            )
    if approx_rows:
        lines.extend(
            _table(
                ["模型", "class_pair", "attn_source", "pearson_r", "p_value", "finite_ratio", "out_of_unit_ratio", "说明"],
                approx_rows,
            )
        )
    else:
        lines.append("未生成可用的 Exp-B/C 近似注意力结果。")
    lines.append("")

    lines.append("## 6. t-SNE 可视化")
    lines.append("")
    lines.append("![van:car 四模型对比](figures/four_model_tsne_van_car.png)")
    lines.append("")
    lines.append("![freight_car:truck 四模型对比](figures/four_model_tsne_fc_truck.png)")
    lines.append("")
    _, tsne_text = _load_tsne_summary(tsne_summary)
    lines.append(f"稳定性检验：{tsne_text}")
    lines.append("")

    lines.append("## 7. 附录：全层分析数据")
    lines.append("")
    lines.append("### 7.1 线性探针全层数据")
    lines.append("")
    linear_all_rows = []
    for model_key in ["m5", "exp0", "expa", "expb", "expc"]:
        payload = linear_data[model_key]
        for row in payload["rows"]:
            linear_all_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["feature_key"]),
                    str(row["class_pair"]),
                    _fmt_float(row["acc_mean"]),
                    _fmt_float(row["acc_std"]),
                    str(row["n_samples"]),
                ]
            )
    lines.extend(_table(["model", "feature_key", "class_pair", "acc_mean", "acc_std", "n_samples"], linear_all_rows))
    lines.append("")

    lines.append("### 7.2 Fisher Ratio 全层数据")
    lines.append("")
    fisher_all_rows = []
    for model_key in ["m5", "exp0", "expa", "expb", "expc"]:
        payload = fisher_data[model_key]
        for row in payload["rows"]:
            fisher_all_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["feature_key"]),
                    str(row["class_pair"]),
                    _fmt_float(row["top10_mean"]),
                    str(row["n_samples"]),
                ]
            )
    lines.extend(_table(["model", "feature_key", "class_pair", "top10_mean", "n_samples"], fisher_all_rows))
    lines.append("")

    lines.append("### 7.3 SE / 注意力权重全行数据")
    lines.append("")
    se_all_rows = []
    for model_key in ["m5", "exp0"]:
        payload = se_data[model_key]
        if payload is None:
            continue
        for row in payload["rows"]:
            se_all_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["feature_key"]),
                    str(row["class_pair"]),
                    str(row["attn_source"]),
                    _fmt_float(row["pearson_r"]),
                    f"{float(row['p_value']):.6g}",
                    str(row["n_objects"]),
                    str(row["n_images_attn"]),
                ]
            )
    for model_key in ["expb", "expc"]:
        payload = approx_data[model_key]
        if payload is None:
            continue
        for row in payload["rows"]:
            se_all_rows.append(
                [
                    f"{MODEL_META[model_key]['display']}†",
                    str(row["feature_key"]),
                    str(row["class_pair"]),
                    str(row["attn_source"]),
                    _fmt_float(row["pearson_r"]),
                    f"{float(row['p_value']):.6g}",
                    str(row["n_objects"]),
                    str(row["n_images_attn"]),
                ]
            )
    lines.extend(
        _table(
            ["model", "feature_key", "class_pair", "attn_source", "pearson_r", "p_value", "n_objects", "n_images_attn"],
            se_all_rows,
        )
    )
    lines.append("")

    lines.append("### 7.4 Silhouette 全层数据")
    lines.append("")
    silhouette_all_rows = []
    for model_key in ["exp0", "expa", "expb", "expc"]:
        payload = silhouette_data[model_key]
        if payload is None:
            continue
        for row in payload["rows"]:
            silhouette_all_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row["feature_key"]),
                    str(row["class_pair"]),
                    _fmt_float(row["silhouette"]),
                    str(row["n_samples"]),
                ]
            )
    lines.extend(_table(["model", "feature_key", "class_pair", "silhouette", "n_samples"], silhouette_all_rows))
    lines.append("")

    lines.append("## 8. 层索引验证记录")
    lines.append("")
    layer_rows = _load_layer_validation(layer_validation)
    if layer_rows:
        lines.extend(_table(["模型", "Layer 7", "Layer 17", "Layer 23", "Layer 24", "Layer 23 children"], [[
            str(row["model"]),
            str(row["layer_7"]),
            str(row["layer_17"]),
            str(row["layer_23"]),
            str(row["layer_24"]),
            ", ".join(row["children"]),
        ] for row in layer_rows]))
    else:
        lines.append("未提供层索引验证记录。")
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "four_model_diagnostic_report.md"),
    )
    parser.add_argument("--layer-validation", type=str, default="")
    parser.add_argument("--tsne-summary", type=str, default="")
    args = parser.parse_args()

    out = build_report(
        output_md=Path(args.output).resolve(),
        layer_validation=Path(args.layer_validation).resolve() if args.layer_validation else None,
        tsne_summary=Path(args.tsne_summary).resolve() if args.tsne_summary else None,
    )
    print(f"[Diagnostic][Report] done: {out}")


if __name__ == "__main__":
    main()
