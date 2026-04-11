from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = ["m5", "exp0", "expa", "expb", "expc"]
MODEL_META = {
    "m5": {
        "display": "M5",
        "stratified": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "m5_stratified.json",
        "eval_csv": ROOT / "result" / "FA-Concat_FPN-PAN_tuned" / "FA-Concat_FPN-PAN_tuned.csv",
    },
    "exp0": {
        "display": "Exp-0",
        "stratified": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "exp0_stratified.json",
        "eval_csv": ROOT / "result" / "exp0_eval" / "exp0_eval.csv",
    },
    "expa": {
        "display": "Exp-A",
        "stratified": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expa_stratified.json",
        "eval_csv": ROOT / "result" / "expa_eval" / "expa_eval.csv",
    },
    "expb": {
        "display": "Exp-B",
        "stratified": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expb_stratified.json",
        "eval_csv": ROOT / "result" / "expb_eval" / "expb_eval.csv",
    },
    "expc": {
        "display": "Exp-C",
        "stratified": ROOT / "src" / "diagnostic" / "outputs" / "reports" / "expc_stratified.json",
        "eval_csv": ROOT / "result" / "expc_eval" / "expc_eval.csv",
    },
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_csv(path: Path) -> dict:
    metrics = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    for row in rows:
        if not row or row[0] in {"metric", "class"}:
            continue
        if len(row) == 2:
            metrics[row[0]] = float(row[1])
    return metrics


def _fmt(v: float | int | None, digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{digits}f}"


def _fmt_acc(row: dict | None) -> str:
    if row is None or row.get("linear_probe_acc") is None or row.get("linear_probe_std") is None:
        return "NA"
    return f"{float(row['linear_probe_acc']):.4f}±{float(row['linear_probe_std']):.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(output_md: Path) -> Path:
    stratified_data = {k: _read_json(v["stratified"]) for k, v in MODEL_META.items()}
    eval_metrics = {k: _read_eval_csv(v["eval_csv"]) for k, v in MODEL_META.items()}
    lookup: dict[tuple[str, str, str], dict] = {}

    for model_key, payload in stratified_data.items():
        for row in payload.get("rows", []):
            lookup[(model_key, str(row["condition"]), str(row["class_pair"]))] = row

    lines: list[str] = []
    lines.append("# 光照条件分层诊断报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat()}")
    threshold = next(iter(stratified_data.values())).get("brightness_threshold", 40.0)
    lines.append(f"- 亮暗阈值：{float(threshold):.1f}")
    lines.append("- 悖论判定口径：以全局 test-set mAP 为“检测更高”依据；某条件下只要 Exp-0 或 Exp-C 任一满足“probe 更低且 mAP 更高”，即记为“是”。")
    lines.append("")

    lines.append("## 1. 数据分布")
    lines.append("")
    dist_rows = []
    for model_key in MODEL_ORDER:
        any_row = stratified_data[model_key]["rows"][0]
        total = int(any_row["n_total_samples"])
        bright = int(any_row["n_bright_samples"])
        dark = int(any_row["n_dark_samples"])
        ratio = 0.0 if total == 0 else dark / total
        dist_rows.append(
            [
                MODEL_META[model_key]["display"],
                str(total),
                str(bright),
                str(dark),
                f"{ratio:.4f}",
            ]
        )
    lines.extend(_table(["模型", "总样本", "亮光样本(≥40)", "暗光样本(<40)", "暗光比例"], dist_rows))
    lines.append("")

    lines.append("## 2. 分层线性探针（feat_fused_p3 · van:car）")
    lines.append("")
    probe_rows = []
    for model_key in MODEL_ORDER:
        bright_row = lookup.get((model_key, "bright", "van:car"))
        dark_row = lookup.get((model_key, "dark", "van:car"))
        all_row = lookup.get((model_key, "all", "van:car"))
        delta = "NA"
        if bright_row and dark_row and bright_row.get("linear_probe_acc") is not None and dark_row.get("linear_probe_acc") is not None:
            delta = f"{float(dark_row['linear_probe_acc']) - float(bright_row['linear_probe_acc']):+.4f}"
        probe_rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt_acc(bright_row),
                _fmt_acc(dark_row),
                _fmt_acc(all_row),
                delta,
            ]
        )
    lines.extend(_table(["模型", "亮光 acc±std", "暗光 acc±std", "全量 acc±std", "亮暗差值"], probe_rows))
    lines.append("")

    lines.append("## 3. 分层 Fisher Ratio（feat_fused_p3 · van:car）")
    lines.append("")
    fisher_rows = []
    for model_key in MODEL_ORDER:
        bright_row = lookup.get((model_key, "bright", "van:car"))
        dark_row = lookup.get((model_key, "dark", "van:car"))
        all_row = lookup.get((model_key, "all", "van:car"))
        delta = "NA"
        if bright_row and dark_row and bright_row.get("fisher_top10") is not None and dark_row.get("fisher_top10") is not None:
            delta = f"{float(dark_row['fisher_top10']) - float(bright_row['fisher_top10']):+.4f}"
        fisher_rows.append(
            [
                MODEL_META[model_key]["display"],
                _fmt(bright_row.get("fisher_top10") if bright_row else None),
                _fmt(dark_row.get("fisher_top10") if dark_row else None),
                _fmt(all_row.get("fisher_top10") if all_row else None),
                delta,
            ]
        )
    lines.extend(_table(["模型", "亮光 Fisher", "暗光 Fisher", "全量 Fisher", "亮暗差值"], fisher_rows))
    lines.append("")

    lines.append("## 4. 悖论分层验证")
    lines.append("")
    paradox_rows = []
    map_delta_exp0 = float(eval_metrics["exp0"]["metrics/mAP50(B)"]) - float(eval_metrics["expa"]["metrics/mAP50(B)"])
    map_delta_expc = float(eval_metrics["expc"]["metrics/mAP50(B)"]) - float(eval_metrics["expa"]["metrics/mAP50(B)"])
    for condition in ["bright", "dark"]:
        row_exp0 = lookup.get(("exp0", condition, "van:car"))
        row_expa = lookup.get(("expa", condition, "van:car"))
        row_expc = lookup.get(("expc", condition, "van:car"))
        delta_exp0 = "NA"
        delta_expc = "NA"
        paradox = False
        if row_exp0 and row_expa and row_exp0.get("linear_probe_acc") is not None and row_expa.get("linear_probe_acc") is not None:
            probe_delta_exp0 = float(row_exp0["linear_probe_acc"]) - float(row_expa["linear_probe_acc"])
            delta_exp0 = f"{probe_delta_exp0:+.4f}"
            paradox = paradox or (probe_delta_exp0 < 0.0 and map_delta_exp0 > 0.0)
        if row_expc and row_expa and row_expc.get("linear_probe_acc") is not None and row_expa.get("linear_probe_acc") is not None:
            probe_delta_expc = float(row_expc["linear_probe_acc"]) - float(row_expa["linear_probe_acc"])
            delta_expc = f"{probe_delta_expc:+.4f}"
            paradox = paradox or (probe_delta_expc < 0.0 and map_delta_expc > 0.0)
        paradox_rows.append([condition, delta_exp0, delta_expc, "是" if paradox else "否"])
    lines.extend(_table(["条件", "Exp-0 − Exp-A (probe)", "Exp-C − Exp-A (probe)", "悖论是否存在"], paradox_rows))
    lines.append("")
    lines.append(
        f"- 全局 mAP50 增量：Exp-0 − Exp-A = {map_delta_exp0:+.4f}，Exp-C − Exp-A = {map_delta_expc:+.4f}。"
    )
    lines.append("")

    lines.append("## 5. 附录：全量分层数据")
    lines.append("")
    appendix_rows = []
    for model_key in MODEL_ORDER:
        for row in stratified_data[model_key].get("rows", []):
            appendix_rows.append(
                [
                    MODEL_META[model_key]["display"],
                    str(row.get("condition")),
                    str(row.get("class_pair")),
                    _fmt(row.get("linear_probe_acc")),
                    _fmt(row.get("linear_probe_std")),
                    _fmt(row.get("fisher_top10")),
                    str(row.get("n_condition_samples")),
                    str(row.get("n_pair_samples_raw")),
                    str(row.get("n_probe_samples_balanced")),
                    str(row.get("n_fisher_samples_balanced")),
                ]
            )
    lines.extend(
        _table(
            [
                "model",
                "condition",
                "class_pair",
                "probe_acc",
                "probe_std",
                "fisher_top10",
                "n_condition_samples",
                "n_pair_samples_raw",
                "n_probe_samples_balanced",
                "n_fisher_samples_balanced",
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
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "stratified_analysis_report.md"),
    )
    args = parser.parse_args()

    out = build_report(Path(args.output).resolve())
    print(f"[Diagnostic][StratifiedReport] done: {out}")


if __name__ == "__main__":
    main()
