from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def run_data_level_stats(device: str) -> None:
    rgb_script = ROOT / "src" / "tools" / "probe_rgb_l_channel_stats.py"
    ir_script = ROOT / "src" / "tools" / "probe_ir_van_car_regions.py"
    if not rgb_script.exists():
        raise FileNotFoundError(f"未找到脚本: {rgb_script}")
    if not ir_script.exists():
        raise FileNotFoundError(f"未找到脚本: {ir_script}")

    commands = [
        [sys.executable, str(rgb_script)],
        [sys.executable, str(ir_script)],
    ]
    for cmd in commands:
        print(f"[Diagnostic][DataLevel] run: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            raise RuntimeError(f"命令失败: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    _ = parser.parse_args()
    run_data_level_stats(device="0")
    print("[Diagnostic][DataLevel] done")


if __name__ == "__main__":
    main()
