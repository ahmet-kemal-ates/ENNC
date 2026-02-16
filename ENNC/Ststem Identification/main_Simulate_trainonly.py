# ========= Determinism & numerics safety =========
import os, random, numpy as np
os.environ["PYTHONHASHSEED"] = "7"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

random.seed(7)
np.random.seed(7)

import tensorflow as tf
tf.random.set_seed(7)
# ===============================================

import json
import sys
from datetime import datetime
import scipy.io
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tkinter.filedialog import askopenfilename, askopenfilenames

from ENNC import ENNC

# ------------------ Helpers ------------------
def _col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.astype(np.float32, copy=False)

def _normalize_voltage(v, vmin, vmax):
    return (v - vmin) / (vmax - vmin)

def _denorm_voltage(vn, vmin, vmax):
    return vn * (vmax - vmin) + vmin

# ------------------ Load model directory ------------------
if "-m" not in sys.argv:
    pick = askopenfilename(
        title="Select model.json inside trained model folder",
        filetypes=(("JSON", "*.json"),),
    )
    model_dir = os.path.dirname(pick)
else:
    model_dir = sys.argv[sys.argv.index("-m") + 1]

# ------------------ Load training metadata ------------------
with open(os.path.join(model_dir, "architecture_fixes.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

norm = cfg["preprocessing"]

minInputVTr = float(norm["minInputVTr"])
maxInputVTr = float(norm["maxInputVTr"])
maxInputITr = float(norm["maxInputITr"])
maxInputTTr = float(norm["maxInputTTr"])
Temp_in     = bool(norm["temp_in"])

# Load Ts & Cn from results.mat (authoritative)
res = scipy.io.loadmat(os.path.join(model_dir, "results.mat"))
Ts = float(np.asarray(res["Ts"]).squeeze())
Cn = np.asarray(res["Cn"]).astype(np.float32)

# ------------------ Load test files ------------------
if "-ts" not in sys.argv:
    testFiles = list(askopenfilenames(
        title="Select test RW file(s)",
        filetypes=(("MAT files", "*.mat"),),
    ))
else:
    ts_arg = sys.argv[sys.argv.index("-ts") + 1]
    testFiles = [p.strip() for p in ts_arg.split(";") if p.strip()]

if not testFiles:
    raise RuntimeError("No test files selected")

# ------------------ Rebuild ENNC model ------------------
ennc_model = ENNC(
    Cn=Cn,
    Ts=Ts,
    cRate_in=True,
    SoC_in=True,
    Temp_in=Temp_in,
)
ennc_model.net.load_weights(os.path.join(model_dir, "model_weights.weights.h5"))

# ------------------ Run simulation ------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join(model_dir, f"Simulation_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

results = []

for fpath in testFiles:
    d = scipy.io.loadmat(fpath)

    ts_Iin  = _col(d["ts_Iin"])
    ts_SoC  = _col(d["ts_SoC"])
    ts_Vout = _col(d["ts_Vout"])
    ts_Temp = _col(d["Temp"]) if ("Temp" in d and Temp_in) else np.zeros_like(ts_Iin)

    # --------- Normalize using TRAIN-ONLY stats ---------
    true_norm = _normalize_voltage(ts_Vout, minInputVTr, maxInputVTr).reshape(-1)
    ts_Iin_norm  = (ts_Iin / maxInputITr).reshape(-1)
    ts_Temp_norm = (ts_Temp / maxInputTTr).reshape(-1)

    inputs = [
        ts_Iin_norm.reshape(-1, 1, 1),
        ts_SoC.reshape(-1, 1, 1),
    ]
    if Temp_in:
        inputs.append(ts_Temp_norm.reshape(-1, 1, 1))

    pred_norm = ennc_model.net.predict(inputs, verbose=0).reshape(-1)

    # Denormalize prediction to volts (physical domain)
    pred_v = _denorm_voltage(pred_norm, minInputVTr, maxInputVTr)
    true_v = ts_Vout.reshape(-1)

    # --------- Metrics (normalized) ---------
    mse_n  = mean_squared_error(true_norm, pred_norm)
    mae_n  = mean_absolute_error(true_norm, pred_norm)
    rmse_n = float(np.sqrt(mse_n))
    r2_n   = r2_score(true_norm, pred_norm)

    # --------- Metrics (volts) ---------
    mse_v  = mean_squared_error(true_v, pred_v)
    mae_v  = mean_absolute_error(true_v, pred_v)
    rmse_v = float(np.sqrt(mse_v))
    r2_v   = r2_score(true_v, pred_v)

    name = os.path.basename(fpath)
    results.append({
        "file": name,
        "mse_v": float(mse_v), "mae_v": float(mae_v), "rmse_v": float(rmse_v), "r2": float(r2_v),
        "mse_n": float(mse_n), "mae_n": float(mae_n), "rmse_n": float(rmse_n), "r2_n": float(r2_n),
    })

    # Plot in volts (more interpretable)
    plt.figure()
    plt.plot(true_v, label="Actual (V)")
    plt.plot(pred_v, "--", label="Predicted (V)")
    plt.title(name)
    plt.legend()
    plt.grid(True)
    safe = os.path.splitext(name)[0]
    plt.savefig(os.path.join(out_dir, f"sim_{safe}.pdf"), bbox_inches="tight")
    plt.close()

# ------------------ Print summary ------------------
print("\nSIMULATION RESULTS (train-only normalization)")
for r in results:
    print(
        f"{r['file']} | "
        f"RMSE_V={r['rmse_v']:.6f}V  MAE_V={r['mae_v']:.6f}V  MSE_V={r['mse_v']:.6f}  R²={r['r2']:.6f}   ||   "
        f"RMSE_N={r['rmse_n']:.6f}  MAE_N={r['mae_n']:.6f}  MSE_N={r['mse_n']:.6f}"
    )

# Save MAT summary
sio.savemat(os.path.join(out_dir, "simulation_results.mat"), {
    "files": np.array([r["file"] for r in results], dtype=object),
    "mse_v": np.array([r["mse_v"] for r in results], dtype=np.float64),
    "mae_v": np.array([r["mae_v"] for r in results], dtype=np.float64),
    "rmse_v": np.array([r["rmse_v"] for r in results], dtype=np.float64),
    "r2": np.array([r["r2"] for r in results], dtype=np.float64),
    "mse_n": np.array([r["mse_n"] for r in results], dtype=np.float64),
    "mae_n": np.array([r["mae_n"] for r in results], dtype=np.float64),
    "rmse_n": np.array([r["rmse_n"] for r in results], dtype=np.float64),
    "r2_n": np.array([r["r2_n"] for r in results], dtype=np.float64),
})

print(f"\nResults saved in: {out_dir}")
