from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_jsons(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        fp = Path(p).resolve()
        data = json.loads(fp.read_text(encoding="utf-8"))
        data["_file"] = str(fp)
        out.append(data)
    return out


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def build_report(
    linear_probe_jsons: list[dict],
    fisher_jsons: list[dict],
    se_jsons: list[dict],
    output_md: Path,
) -> Path:
    lines = []
    lines.append("# Cross-Model Diagnostic Comparison")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().isoformat()}")
    lines.append("")

    if linear_probe_jsons:
        lines.append("## Linear Probe 对比")
        rows = []
        for js in linear_probe_jsons:
            model = Path(str(js.get("feature_file", ""))).stem
            for r in js.get("rows", []):
                rows.append(
                    [
                        model,
                        str(r.get("feature_key", "")),
                        str(r.get("class_pair", "")),
                        f"{float(r.get('acc_mean', 0.0)):.4f}",
                        f"{float(r.get('acc_std', 0.0)):.4f}",
                        str(r.get("n_samples", "")),
                    ]
                )
        lines.extend(_render_table(["model", "feature", "pair", "acc_mean", "acc_std", "n"], rows))
        lines.append("")

    if fisher_jsons:
        lines.append("## Fisher Ratio 对比")
        rows = []
        for js in fisher_jsons:
            model = Path(str(js.get("feature_file", ""))).stem
            for r in js.get("rows", []):
                rows.append(
                    [
                        model,
                        str(r.get("feature_key", "")),
                        str(r.get("class_pair", "")),
                        f"{float(r.get('top10_mean', 0.0)):.6f}",
                        str(r.get("n_samples", "")),
                    ]
                )
        lines.extend(_render_table(["model", "feature", "pair", "fisher_top10_mean", "n"], rows))
        lines.append("")

    if se_jsons:
        lines.append("## SE 权重相关性对比")
        rows = []
        for js in se_jsons:
            model = Path(str(js.get("feature_file", ""))).stem
            for r in js.get("rows", []):
                rows.append(
                    [
                        model,
                        str(r.get("feature_key", "")),
                        str(r.get("class_pair", "")),
                        f"{float(r.get('pearson_r', 0.0)):.6f}",
                        f"{float(r.get('p_value', 1.0)):.6g}",
                        str(r.get("n_objects", "")),
                    ]
                )
        lines.extend(_render_table(["model", "feature", "pair", "pearson_r", "p_value", "n_obj"], rows))
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-probe-jsons", nargs="*", default=[])
    parser.add_argument("--fisher-jsons", nargs="*", default=[])
    parser.add_argument("--se-jsons", nargs="*", default=[])
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "cross_model_comparison.md"),
    )
    args = parser.parse_args()

    out = build_report(
        linear_probe_jsons=_load_jsons(list(args.linear_probe_jsons)),
        fisher_jsons=_load_jsons(list(args.fisher_jsons)),
        se_jsons=_load_jsons(list(args.se_jsons)),
        output_md=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][CrossCompare] done: {out}")


if __name__ == "__main__":
    main()
