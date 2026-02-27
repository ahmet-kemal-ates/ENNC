import argparse
import glob
import json
import os

import numpy as np
import scipy.io


def _col(x):
    x = np.asarray(x).reshape(-1)
    return x.astype(np.float32, copy=False)


def _load_train_mat(path):
    d = scipy.io.loadmat(path)
    i = _col(d["tr_Iin"])
    soc = _col(d["tr_SoC"])
    v = _col(d["tr_Vout"])
    t = _col(d["Temp"]) if "Temp" in d else np.full_like(i, 25.0, dtype=np.float32)
    return i, soc, v, t


def _fill_missing_2d(arr):
    out = arr.copy()
    n_s, n_i = out.shape
    for s in range(n_s):
        row = out[s]
        good = np.isfinite(row)
        if np.any(good):
            x = np.arange(n_i)
            row[~good] = np.interp(x[~good], x[good], row[good]).astype(np.float32)
            out[s] = row
    col_mean = np.nanmean(out, axis=0)
    if np.any(np.isnan(col_mean)):
        valid = np.where(np.isfinite(col_mean))[0]
        if len(valid) > 0:
            miss = np.where(~np.isfinite(col_mean))[0]
            col_mean[miss] = np.interp(miss, valid, col_mean[valid]).astype(np.float32)
        else:
            col_mean[:] = 3.7
    for s in range(n_s):
        if np.any(~np.isfinite(out[s])):
            out[s, ~np.isfinite(out[s])] = col_mean[~np.isfinite(out[s])]
    return out


def main():
    parser = argparse.ArgumentParser(description="Build ENNC-lite LUT directly from train RW MAT data.")
    parser.add_argument("--train_glob", default="Models/trainRW*.mat", help="Glob for training MAT files.")
    parser.add_argument("--out_json", required=True, help="Output JSON file.")
    parser.add_argument("--soc_points", type=int, default=31, help="SoE grid size.")
    parser.add_argument("--i_points", type=int, default=51, help="Current grid size.")
    parser.add_argument("--temp_values", type=str, default="25", help="Comma-separated fixed temps in C.")
    parser.add_argument("--v_clip_min", type=float, default=2.5, help="Minimum voltage clip.")
    parser.add_argument("--v_clip_max", type=float, default=5.0, help="Maximum voltage clip.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.train_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.train_glob}")

    i_all, soc_all, v_all, t_all = [], [], [], []
    for p in files:
        i, soc, v, t = _load_train_mat(p)
        mask = np.isfinite(i) & np.isfinite(soc) & np.isfinite(v) & np.isfinite(t)
        i_all.append(i[mask])
        soc_all.append(soc[mask])
        v_all.append(v[mask])
        t_all.append(t[mask])

    i_all = np.concatenate(i_all).astype(np.float32)
    soc_all = np.clip(np.concatenate(soc_all).astype(np.float32), 0.0, 1.0)
    v_all = np.concatenate(v_all).astype(np.float32)
    t_all = np.concatenate(t_all).astype(np.float32)

    i_abs = float(np.max(np.abs(i_all)))
    soc_grid = np.linspace(0.0, 1.0, args.soc_points, dtype=np.float32)
    i_grid = np.linspace(-i_abs, i_abs, args.i_points, dtype=np.float32)
    temp_grid = np.asarray([float(x.strip()) for x in args.temp_values.split(",") if x.strip()], dtype=np.float32)

    voltage_table = np.zeros((len(temp_grid), len(soc_grid), len(i_grid)), dtype=np.float32)
    si = np.clip(np.searchsorted(soc_grid, soc_all), 0, len(soc_grid) - 1)
    ii = np.clip(np.searchsorted(i_grid, i_all), 0, len(i_grid) - 1)

    for ti, t0 in enumerate(temp_grid):
        tw = np.exp(-np.abs(t_all - t0) / 7.0).astype(np.float32)
        sums = np.zeros((len(soc_grid), len(i_grid)), dtype=np.float64)
        wsum = np.zeros((len(soc_grid), len(i_grid)), dtype=np.float64)
        for k in range(len(v_all)):
            s = si[k]
            c = ii[k]
            w = float(tw[k])
            sums[s, c] += w * float(v_all[k])
            wsum[s, c] += w
        grid = np.full((len(soc_grid), len(i_grid)), np.nan, dtype=np.float32)
        mask = wsum > 1e-12
        grid[mask] = (sums[mask] / wsum[mask]).astype(np.float32)
        grid = _fill_missing_2d(grid)
        grid = np.clip(grid, args.v_clip_min, args.v_clip_max)
        voltage_table[ti] = grid

    payload = {
        "paper_lineage": "ENNC-inspired white-box data-driven mapping compressed as LUT",
        "source_files": [os.path.abspath(f) for f in files],
        "soc_grid": soc_grid.tolist(),
        "i_grid": i_grid.tolist(),
        "temp_grid": temp_grid.tolist(),
        "voltage_table": voltage_table.tolist(),
        "clip": {"v_min": float(args.v_clip_min), "v_max": float(args.v_clip_max)},
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(out_json)


if __name__ == "__main__":
    main()
