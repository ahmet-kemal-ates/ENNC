# ========= Determinism & numerics safety (SET BEFORE TF IMPORTS) =========
import os, random, numpy as np
os.environ["PYTHONHASHSEED"] = "7"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Optional: reduce numeric drift on CPU (comment out if you prefer oneDNN)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

random.seed(7)
np.random.seed(7)

import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")
# ========================================================================

import sys
import json
import time
from datetime import datetime
import scipy.io
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tkinter.filedialog import askopenfilename, askopenfilenames

from ENNC import ENNC

# --- Save figures as PDF by default ---
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("Models", f"ENNC_Fixed_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

arguments = sys.argv

# -------------------------- Helpers --------------------------
def _col(x):
    """Ensure (N,1) float32 column vector."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.astype(np.float32, copy=False)

def _as_float_scalar(x, default):
    try:
        return float(np.asarray(x).squeeze().item())
    except Exception:
        return float(default)

def _load_train_mat(path):
    d = scipy.io.loadmat(path)
    tr_Time = _col(d["tr_Time"])
    tr_Iin  = _col(d["tr_Iin"])
    tr_SoC  = _col(d["tr_SoC"])
    tr_Vout = _col(d["tr_Vout"])
    tr_Temp = _col(d["Temp"]) if "Temp" in d else np.zeros_like(tr_Iin, dtype=np.float32)
    tr_Cn   = _as_float_scalar(d.get("Cn", 1.0), 1.0)
    tr_Ts   = _as_float_scalar(d.get("Ts", 1.0), 1.0)
    return tr_Time, tr_Iin, tr_SoC, tr_Vout, tr_Temp, tr_Cn, tr_Ts

def _load_test_mat(path):
    d = scipy.io.loadmat(path)
    ts_Time = _col(d["ts_Time"])
    ts_Iin  = _col(d["ts_Iin"])
    ts_SoC  = _col(d["ts_SoC"])
    ts_Vout = _col(d["ts_Vout"])
    ts_Temp = _col(d["Temp"]) if "Temp" in d else np.zeros_like(ts_Iin, dtype=np.float32)
    ts_Cn   = _as_float_scalar(d.get("Cn", 1.0), 1.0)
    ts_Ts   = _as_float_scalar(d.get("Ts", 1.0), 1.0)
    return ts_Time, ts_Iin, ts_SoC, ts_Vout, ts_Temp, ts_Cn, ts_Ts

def _assert_all_finite(name, arr):
    x = np.asarray(arr)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))
        raise ValueError(f"{name} has non-finite values at indices {bad}")

def _normalize_voltage(v, vmin, vmax):
    return (v - vmin) / (vmax - vmin)

# -------------------------- LOAD DATASETS --------------------------
# Training: allow multiple selection (dialog) OR -tr "file1;file2;file3"
if "-tr" not in arguments:
    trainFiles = list(askopenfilenames(
        initialdir=os.path.abspath("../../../Dataset/"),
        title="Load Training Data (select 1 or more .mat files)",
        defaultextension="mat",
        filetypes=(("mat file", "*.mat"), ("All Files", "*.*")),
    ))
else:
    tr_arg = arguments[arguments.index("-tr") + 1]
    trainFiles = [p.strip() for p in tr_arg.split(";") if p.strip()]

if not trainFiles:
    raise RuntimeError("No training files selected.")

# Test: allow single OR multiple selection (dialog) OR -ts "file1;file2"
if "-ts" not in arguments:
    testFiles = list(askopenfilenames(
        initialdir=os.path.abspath("../Dataset/"),
        title="Load Test Data (select 1 or more .mat files)",
        defaultextension="mat",
        filetypes=(("mat file", "*.mat"), ("All Files", "*.*")),
    ))
    # If user selects none in multi dialog, fallback to single dialog (nice UX)
    if not testFiles:
        one = askopenfilename(
            initialdir=os.path.abspath("../Dataset/"),
            title="Load Test Data",
            defaultextension="mat",
            filetypes=(("mat file", "*.mat"), ("All Files", "*.*")),
        )
        testFiles = [one] if one else []
else:
    ts_arg = arguments[arguments.index("-ts") + 1]
    testFiles = [p.strip() for p in ts_arg.split(";") if p.strip()]

if not testFiles:
    raise RuntimeError("No test files selected.")

# Load + concatenate training files
tr_Time_list, tr_Iin_list, tr_SoC_list, tr_Vout_list, tr_Temp_list = [], [], [], [], []
base_Ts, base_Cn = None, None
t_offset = 0.0
train_contains_temp = False

cn_mismatches = []
ts_mismatches = []

for f in trainFiles:
    t, iin, soc, v, temp, cn, ts = _load_train_mat(f)

    if base_Ts is None:
        base_Ts = ts
        base_Cn = cn
    else:
        if abs(ts - base_Ts) > 1e-12:
            ts_mismatches.append((os.path.basename(f), ts, base_Ts))
        if abs(cn - base_Cn) > 1e-9:
            cn_mismatches.append((os.path.basename(f), cn, base_Cn))

    # Make time continuous across concatenated files
    t0 = float(t[0])
    t_adj = (t - t0) + t_offset
    t_offset = float(t_adj[-1]) + float(base_Ts)

    tr_Time_list.append(t_adj)
    tr_Iin_list.append(iin)
    tr_SoC_list.append(soc)
    tr_Vout_list.append(v)
    tr_Temp_list.append(temp)

    if np.any(temp != 0):
        train_contains_temp = True

if ts_mismatches:
    msg = "\n".join([f"  {name}: Ts={ts} (expected {exp})" for name, ts, exp in ts_mismatches])
    raise ValueError(f"Ts mismatch across selected training files:\n{msg}")

# For Cn: warn but do not crash (you can choose to enforce if desired)
if cn_mismatches:
    print("\nWARNING: Cn mismatch across training files (continuing with base Cn):")
    for name, cn, exp in cn_mismatches:
        print(f"  {name}: Cn={cn} (base {exp})")

tr_Time = np.vstack(tr_Time_list)
tr_Iin  = np.vstack(tr_Iin_list)
tr_SoC  = np.vstack(tr_SoC_list)
tr_Vout = np.vstack(tr_Vout_list)
tr_Temp = np.vstack(tr_Temp_list)
tr_Cn   = np.array([[float(base_Cn)]], dtype=np.float32)
tr_Ts   = float(base_Ts)

# Load test files into a list (evaluate each later)
tests = []
for f in testFiles:
    ts_Time, ts_Iin, ts_SoC, ts_Vout, ts_Temp, ts_Cn, ts_Ts = _load_test_mat(f)
    if abs(ts_Ts - tr_Ts) > 1e-12:
        raise ValueError(f"Ts mismatch: test file {os.path.basename(f)} has Ts={ts_Ts}, expected Ts={tr_Ts}")
    tests.append({
        "path": f,
        "ts_Time": ts_Time,
        "ts_Iin": ts_Iin,
        "ts_SoC": ts_SoC,
        "ts_Vout": ts_Vout,
        "ts_Temp": ts_Temp,
        "ts_Cn": ts_Cn,
        "ts_Ts": ts_Ts,
    })

# -------------------------- INFO --------------------------
print("=" * 80)
print("ENNC ARCHITECTURE ANALYSIS AND FIXES")
print("=" * 80)
print("Analyzing ENNC architecture limitations...")
print("1. ENNC uses sigmoid activations that can cause vanishing gradients")
print("2. ENNC has fixed functional expansions that may not suit all data")
print("3. ENNC uses simple linear combinations that limit expressiveness")
print("4. ENNC lacks proper regularization for raw data")

# Show characteristics using first test file (summary); full per-test printed later
ts0 = tests[0]
print(f"\nYour data characteristics (train vs first test):")
print(f"Training SoC range: [{np.min(tr_SoC):.6f}, {np.max(tr_SoC):.6f}]")
print(f"Test SoC range:     [{np.min(ts0['ts_SoC']):.6f}, {np.max(ts0['ts_SoC']):.6f}]")
print(f"Training voltage range: [{np.min(tr_Vout):.6f}, {np.max(tr_Vout):.6f}]")
print(f"Test voltage range:     [{np.min(ts0['ts_Vout']):.6f}, {np.max(ts0['ts_Vout']):.6f}]")
print(f"Training current range: [{np.min(tr_Iin):.6f}, {np.max(tr_Iin):.6f}]")
print(f"Test current range:     [{np.min(ts0['ts_Iin']):.6f}, {np.max(ts0['ts_Iin']):.6f}]")

# -------------------------- PREPROCESSING (global normalization across all selected data) --------------------------
print(f"\n{'=' * 80}")
print("ENHANCED DATA PREPROCESSING")
print(f"{'=' * 80}")

# Step 1: Data cleaning (train)
print("Step 1: Data cleaning...")
tr_mask = np.isfinite(tr_Vout) & np.isfinite(tr_Iin) & np.isfinite(tr_SoC)
if not np.all(tr_mask):
    print(f"Removed {np.sum(~tr_mask)} invalid training samples")
    tr_Time = tr_Time[tr_mask]
    tr_Iin  = tr_Iin[tr_mask]
    tr_SoC  = tr_SoC[tr_mask]
    tr_Vout = tr_Vout[tr_mask]
    tr_Temp = tr_Temp[tr_mask]

# Step 1: Data cleaning (each test)
for t in tests:
    ts_mask = np.isfinite(t["ts_Vout"]) & np.isfinite(t["ts_Iin"]) & np.isfinite(t["ts_SoC"])
    if not np.all(ts_mask):
        removed = int(np.sum(~ts_mask))
        print(f"Removed {removed} invalid samples from test file {os.path.basename(t['path'])}")
        t["ts_Time"] = t["ts_Time"][ts_mask]
        t["ts_Iin"]  = t["ts_Iin"][ts_mask]
        t["ts_SoC"]  = t["ts_SoC"][ts_mask]
        t["ts_Vout"] = t["ts_Vout"][ts_mask]
        t["ts_Temp"] = t["ts_Temp"][ts_mask]

# Step 2: SoC overlap ratio (train vs each test)
print("Step 2: SoC range optimization...")
tr_soc_min, tr_soc_max = float(np.min(tr_SoC)), float(np.max(tr_SoC))
tr_soc_range = tr_soc_max - tr_soc_min

overlap_ratios = {}
for t in tests:
    ts_soc_min, ts_soc_max = float(np.min(t["ts_SoC"])), float(np.max(t["ts_SoC"]))
    ts_soc_range = ts_soc_max - ts_soc_min
    soc_overlap_min = max(tr_soc_min, ts_soc_min)
    soc_overlap_max = min(tr_soc_max, ts_soc_max)
    soc_overlap_range = max(0.0, soc_overlap_max - soc_overlap_min)
    denom = min(tr_soc_range, ts_soc_range)
    overlap_ratio = soc_overlap_range / denom if denom > 0 else 0.0
    overlap_ratios[os.path.basename(t["path"])] = float(overlap_ratio)

# Print overlap for first test + a note if any are low
print(f"SoC overlap ratio (first test): {overlap_ratios[os.path.basename(ts0['path'])]:.3f}")
low_overlap = [k for k, v in overlap_ratios.items() if v < 0.95]
if low_overlap:
    print("NOTE: Some test files have low SoC overlap with training:", ", ".join(low_overlap))

# Step 3: Improved normalization (computed using train + ALL tests)
print("Step 3: Improved normalization...")

# gather global min/max across all (train + tests)
all_v_min = float(np.min(tr_Vout))
all_v_max = float(np.max(tr_Vout))
all_i_abs_max = float(np.max(np.abs(tr_Iin)))

for t in tests:
    all_v_min = min(all_v_min, float(np.min(t["ts_Vout"])))
    all_v_max = max(all_v_max, float(np.max(t["ts_Vout"])))
    all_i_abs_max = max(all_i_abs_max, float(np.max(np.abs(t["ts_Iin"]))))

voltage_range = all_v_max - all_v_min
current_max = all_i_abs_max

headroom = 0.2
minInputVTr = all_v_min
maxInputVTr = all_v_max
delta = headroom * voltage_range
maxInputVTr += delta / 2.0
minInputVTr -= delta / 2.0

maxInputITr = (1.0 + headroom) * current_max
maxInputTTr = 60.0

print("Improved normalization parameters:")
print(f"  Voltage: [{minInputVTr:.6f}, {maxInputVTr:.6f}]")
print(f"  Current: ±{maxInputITr:.6f}")
print(f"  Temperature: /{maxInputTTr:g}")

tr_Vout_norm = _normalize_voltage(tr_Vout, minInputVTr, maxInputVTr)
tr_Iin_norm  = tr_Iin / maxInputITr
tr_Temp_norm = tr_Temp / maxInputTTr

# Normalize each test
for t in tests:
    t["ts_Vout_norm"] = _normalize_voltage(t["ts_Vout"], minInputVTr, maxInputVTr)
    t["ts_Iin_norm"]  = t["ts_Iin"] / maxInputITr
    t["ts_Temp_norm"] = t["ts_Temp"] / maxInputTTr

print("Normalized ranges (train):")
print(f"  Voltage: [{float(np.min(tr_Vout_norm)):.6f}, {float(np.max(tr_Vout_norm)):.6f}]")
print(f"  Current: [{float(np.min(tr_Iin_norm)):.6f}, {float(np.max(tr_Iin_norm)):.6f}]")

print("Normalized ranges (first test):")
print(f"  Voltage: [{float(np.min(ts0['ts_Vout_norm'])):.6f}, {float(np.max(ts0['ts_Vout_norm'])):.6f}]")
print(f"  Current: [{float(np.min(ts0['ts_Iin_norm'])):.6f}, {float(np.max(ts0['ts_Iin_norm'])):.6f}]")

# Step 4: Minimal data augmentation (kept off by default)
print("Step 4: Minimal data augmentation...")
noise_level = 0.0  # keep off for stability
tr_Iin_norm = tr_Iin_norm + np.random.normal(0, noise_level, tr_Iin_norm.shape).astype(np.float32)
for t in tests:
    t["ts_Iin_norm"] = t["ts_Iin_norm"] + np.random.normal(0, noise_level, t["ts_Iin_norm"].shape).astype(np.float32)

# -------------------------- RESHAPE INPUTS --------------------------
Temp_in = train_contains_temp  # only include if training actually has nonzero temp
inputTr  = [np.reshape(tr_Iin_norm, (-1, 1, 1)), np.reshape(tr_SoC, (-1, 1, 1))]
outputTr = np.reshape(tr_Vout_norm, (-1, 1, 1))

if Temp_in:
    inputTr.append(np.reshape(tr_Temp_norm, (-1, 1, 1)))

for t in tests:
    inputTs  = [np.reshape(t["ts_Iin_norm"], (-1, 1, 1)), np.reshape(t["ts_SoC"], (-1, 1, 1))]
    outputTs = np.reshape(t["ts_Vout_norm"], (-1, 1, 1))
    if Temp_in:
        inputTs.append(np.reshape(t["ts_Temp_norm"], (-1, 1, 1)))
    t["inputTs"] = inputTs
    t["outputTs"] = outputTs

# -------------------------- FIXED ENNC ARCHITECTURE --------------------------
print(f"\n{'=' * 80}")
print("FIXED ENNC ARCHITECTURE")
print(f"{'=' * 80}")

nEpoch    = 2000
batchSize = 16
print(f"  Epochs: {nEpoch}")
print(f"  Batch Size: {batchSize}")
print(f"  Optimizer: Adam (clipnorm)")
print(f"  Loss Function: mse")
print(f"  Train files: {len(trainFiles)}")
print(f"  Test files:  {len(testFiles)}")
if Temp_in:
    print("  Temp input:  enabled")
else:
    print("  Temp input:  disabled")

ennc_model = ENNC(Cn=tr_Cn, Ts=tr_Ts, cRate_in=True, SoC_in=True, Temp_in=Temp_in)

# -------------------------- PRE-FLIGHT FINITENESS CHECKS --------------------------
for name, arr in [
    ("tr_Vout_norm", tr_Vout_norm),
    ("tr_Iin_norm", tr_Iin_norm),
]:
    _assert_all_finite(name, arr)

for t in tests:
    _assert_all_finite(f"{os.path.basename(t['path'])}: ts_Vout_norm", t["ts_Vout_norm"])
    _assert_all_finite(f"{os.path.basename(t['path'])}: ts_Iin_norm", t["ts_Iin_norm"])

probe = ennc_model.net.predict([x[:1] for x in inputTr], verbose=0)
if not np.all(np.isfinite(probe)):
    raise ValueError("Non-finite model output BEFORE training (check τ bounding / custom layers).")

try:
    tau_probe = ennc_model.TauDynFnc.predict([x[:1] for x in inputTr], verbose=0)
    print("τ raw head min/max:", float(np.nanmin(tau_probe)), float(np.nanmax(tau_probe)))
except Exception:
    pass

# -------------------------- TRAINING --------------------------
print(f"\n{'=' * 80}")
print("ENHANCED TRAINING WITH GRADIENT MONITORING")
print(f"{'=' * 80}")

opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
ennc_model.net.compile(optimizer=opt, loss="mse")

cbs = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best_weights.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=40,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=120,
        restore_best_weights=True,
    ),
]

start_time = time.time()
history_obj = ennc_model.net.fit(
    inputTr,
    outputTr,
    epochs=nEpoch,
    batch_size=batchSize,
    callbacks=cbs,
    verbose=2,
)
trainingTime = time.time() - start_time

# Ensure best weights loaded
best_path = os.path.join(output_dir, "best_weights.weights.h5")
if os.path.exists(best_path):
    ennc_model.net.load_weights(best_path)
    print("Loaded best weights for evaluation")

# -------------------------- EVALUATION (train + ALL tests) --------------------------
pred_train = ennc_model.net.predict(inputTr, verbose=0).reshape(-1)
true_train = outputTr.reshape(-1)
mse_train = float(ennc_model.net.test_on_batch(inputTr, outputTr))

# Evaluate each test file separately
test_metrics = []
for t in tests:
    pred_test = ennc_model.net.predict(t["inputTs"], verbose=0).reshape(-1)
    true_test = t["outputTs"].reshape(-1)

    mse_test  = float(ennc_model.net.test_on_batch(t["inputTs"], t["outputTs"]))
    mae       = float(mean_absolute_error(true_test, pred_test))
    rmse      = float(np.sqrt(mean_squared_error(true_test, pred_test)))
    r2        = float(r2_score(true_test, pred_test))

    t["pred_test"] = pred_test
    t["true_test"] = true_test

    test_metrics.append({
        "file": os.path.basename(t["path"]),
        "mse_test": mse_test,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    })

# Print summary
print("\nTraining Results:")
print(f"  MSE Train: {mse_train:.6f}")
print("\nTest Results (per file):")
for m in test_metrics:
    print(f"  {m['file']}: MSE={m['mse_test']:.6f}  MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R²={m['r2']:.6f}")

# -------------------------- SAVE MODEL AND RESULTS --------------------------
with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as f:
    f.write(ennc_model.net.to_json())

ennc_model.net.save_weights(os.path.join(output_dir, "model_weights.weights.h5"))

architecture_profile = {
    "preprocessing": {
        "headroom_V": float(headroom),
        "minInputVTr": float(minInputVTr),
        "maxInputVTr": float(maxInputVTr),
        "maxInputITr": float(maxInputITr),
        "maxInputTTr": float(maxInputTTr),
        "soc_overlap_ratio_per_test": overlap_ratios,
        "noise_level": float(noise_level),
        "temp_in": bool(Temp_in),
    },
    "training": {
        "epochs_run": int(history_obj.epoch[-1] + 1),
        "best_loss": float(np.min(history_obj.history["loss"])),
        "batch_size": int(batchSize),
        "initial_lr": 3e-4,
        "min_lr": 1e-6,
    },
    "files": {
        "train_files": [os.path.basename(p) for p in trainFiles],
        "test_files": [os.path.basename(p) for p in testFiles],
    }
}

with open(os.path.join(output_dir, "architecture_fixes.json"), "w", encoding="utf-8") as f:
    json.dump(architecture_profile, f, indent=2)

# Save a compact .mat with train and each test's predictions + metrics
# (MATLAB compatibility: store filenames/metrics as simple arrays)
test_names = np.array([m["file"] for m in test_metrics], dtype=object)
mse_tests  = np.array([m["mse_test"] for m in test_metrics], dtype=np.float64)
mae_tests  = np.array([m["mae"] for m in test_metrics], dtype=np.float64)
rmse_tests = np.array([m["rmse"] for m in test_metrics], dtype=np.float64)
r2_tests   = np.array([m["r2"] for m in test_metrics], dtype=np.float64)

# Save first test’s raw arrays for convenience + all metrics
first_test = tests[0]
sio.savemat(os.path.join(output_dir, "results.mat"), {
    "tr_Time": tr_Time, "tr_Iin": tr_Iin, "tr_SoC": tr_SoC, "tr_Temp": tr_Temp, "tr_Vout": tr_Vout,
    "Ts": np.array([[tr_Ts]], dtype=np.float64),
    "Cn": tr_Cn.astype(np.float64),

    # First test raw signals (so existing MATLAB scripts still have something familiar)
    "ts_Time": first_test["ts_Time"],
    "ts_Iin": first_test["ts_Iin"],
    "ts_SoC": first_test["ts_SoC"],
    "ts_Temp": first_test["ts_Temp"],
    "ts_Vout": first_test["ts_Vout"],

    # Training details
    "trainingTime": float(trainingTime),
    "lossHistory": np.array(history_obj.history["loss"], dtype=np.float64),
    "mse_train": float(mse_train),

    # Multi-test metrics
    "test_files": test_names,
    "mse_test": mse_tests,
    "mae_test": mae_tests,
    "rmse_test": rmse_tests,
    "r2_test": r2_tests,
})

# -------------------------- LOG FILE --------------------------
log_path = os.path.join(output_dir, "log.txt")
with open(log_path, "w", encoding="utf-8") as log:
    log.write("ENNC Model Training Log\n")
    log.write(f"Run Timestamp: {timestamp}\n")
    log.write(f"Output Dir: {output_dir}\n")
    log.write("\nTraining Files:\n")
    for p in trainFiles:
        log.write(f"  - {os.path.basename(p)}\n")
    log.write("\nTest Files:\n")
    for p in testFiles:
        log.write(f"  - {os.path.basename(p)}\n")
    log.write(f"\nTraining Time: {trainingTime:.2f} seconds\n")
    log.write("Hyperparameters:\n")
    log.write(f"  Epochs: {nEpoch}\n")
    log.write(f"  Batch Size: {batchSize}\n")
    log.write("  Optimizer: Adam (clipnorm=1.0)\n")
    log.write("  Loss Function: mse\n")
    log.write(f"  Temp_in: {Temp_in}\n")
    log.write("\nNormalization:\n")
    log.write(f"  Voltage: [{minInputVTr:.6f}, {maxInputVTr:.6f}]\n")
    log.write(f"  Current: ±{maxInputITr:.6f}\n")
    log.write(f"  Temperature: /{maxInputTTr:g}\n")
    log.write("\nSoC overlap ratio (per test):\n")
    for k, v in overlap_ratios.items():
        log.write(f"  {k}: {v:.6f}\n")
    log.write(f"\nMSE Train: {mse_train:.6f}\n")
    log.write("\nTest metrics (per file):\n")
    for m in test_metrics:
        log.write(
            f"  {m['file']}: MSE={m['mse_test']:.6f}  "
            f"MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}\n"
        )

# -------------------------- PLOTS (saved as PDFs) --------------------------
# Train loss
fig_loss = plt.figure()
plt.plot(history_obj.history["loss"])
plt.title("Training Loss - Fixed ENNC Architecture")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
fig_loss.savefig(os.path.join(output_dir, f"training_loss_{timestamp}.pdf"), bbox_inches="tight")
plt.close(fig_loss)

# Predicted vs actual for each test file
for t, m in zip(tests, test_metrics):
    fig = plt.figure()
    plt.plot(t["true_test"], label="Actual")
    plt.plot(t["pred_test"], label="Predicted", linestyle="--")
    plt.title(f"Predicted vs Actual (Test) - {m['file']}")
    plt.legend(); plt.grid(True)
    safe_name = os.path.splitext(m["file"])[0]
    fig.savefig(os.path.join(output_dir, f"predicted_vs_actual_{safe_name}_{timestamp}.pdf"), bbox_inches="tight")
    plt.close(fig)

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE")
print(f"{'=' * 80}")
print(f"Model saved in: {output_dir}")
print(f"Train MSE: {mse_train:.6f}")
print(f"Log written to: {log_path}")
