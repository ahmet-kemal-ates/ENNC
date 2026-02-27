import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper to export ENNC-lite LUT JSON for framework integration."
    )
    parser.add_argument(
        "--train_glob",
        default=r".\ENNC\Ststem Identification\Models\trainRW*.mat",
        help="Glob pattern for RW train MAT files.",
    )
    parser.add_argument(
        "--out_json",
        default=r".\exports\ennc_lite_model.json",
        help="Output JSON path.",
    )
    parser.add_argument("--soc_points", type=int, default=41)
    parser.add_argument("--i_points", type=int, default=61)
    parser.add_argument("--temp_values", type=str, default="25")
    parser.add_argument("--v_clip_min", type=float, default=2.5)
    parser.add_argument("--v_clip_max", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    builder = root / "ENNC" / "Ststem Identification" / "build_ennc_lite_from_data.py"
    if not builder.exists():
        raise FileNotFoundError(f"Builder script not found: {builder}")

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(builder),
        "--train_glob",
        args.train_glob,
        "--out_json",
        str(out_path),
        "--soc_points",
        str(args.soc_points),
        "--i_points",
        str(args.i_points),
        "--temp_values",
        args.temp_values,
        "--v_clip_min",
        str(args.v_clip_min),
        "--v_clip_max",
        str(args.v_clip_max),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(root))
    print(str(out_path))


if __name__ == "__main__":
    main()
