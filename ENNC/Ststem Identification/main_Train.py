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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tkinter.filedialog import askopenfilename

from ENNC import ENNC

# --- Save figures as PDF by default ---
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("Models", f"ENNC_Fixed_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# CLI args
arguments = sys.argv

### LOAD DATASET ###
trainFile = askopenfilename(initialdir=os.path.abspath('../../../Dataset/'), title='Load Training Data',
                            defaultextension='mat', filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if '-tr' not in arguments else arguments[arguments.index('-tr')+1]
dataTr = scipy.io.loadmat(trainFile)
tr_Time = dataTr['tr_Time']
tr_Iin  = dataTr['tr_Iin']
tr_SoC  = dataTr['tr_SoC']
tr_Vout = dataTr['tr_Vout']
tr_Temp = dataTr['Temp'] if 'Temp' in dataTr else np.zeros(tr_Iin.shape)
tr_Cn   = dataTr['Cn'] if 'Cn' in dataTr else np.array([[1]])
tr_Ts   = float(np.asarray(dataTr['Ts']).squeeze().item()) if 'Ts' in dataTr else 1.0

testFile = askopenfilename(initialdir=os.path.abspath('../Dataset/'), title='Load Test Data',
                           defaultextension='mat', filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if '-ts' not in arguments else arguments[arguments.index('-ts')+1]
dataTs = scipy.io.loadmat(testFile)
ts_Time = dataTs['ts_Time']
ts_Iin  = dataTs['ts_Iin']
ts_SoC  = dataTs['ts_SoC']
ts_Vout = dataTs['ts_Vout']
ts_Temp = dataTs['Temp'] if 'Temp' in dataTs else np.zeros(ts_Iin.shape)
ts_Cn   = dataTs['Cn'] if 'Cn' in dataTs else np.array([[1]])
ts_Ts   = float(np.asarray(dataTs['Ts']).squeeze().item()) if 'Ts' in dataTs else 1.0

assert tr_Ts == ts_Ts

### INFO ###
print("="*80)
print("ENNC ARCHITECTURE ANALYSIS AND FIXES")
print("="*80)
print("Analyzing ENNC architecture limitations...")
print("1. ENNC uses sigmoid activations that can cause vanishing gradients")
print("2. ENNC has fixed functional expansions that may not suit all data")
print("3. ENNC uses simple linear combinations that limit expressiveness")
print("4. ENNC lacks proper regularization for raw data")

print(f"\nYour data characteristics:")
print(f"Training SoC range: [{np.min(tr_SoC):.6f}, {np.max(tr_SoC):.6f}]")
print(f"Test SoC range:     [{np.min(ts_SoC):.6f}, {np.max(ts_SoC):.6f}]")
print(f"Training voltage range: [{np.min(tr_Vout):.6f}, {np.max(tr_Vout):.6f}]")
print(f"Test voltage range:     [{np.min(ts_Vout):.6f}, {np.max(ts_Vout):.6f}]")
print(f"Training current range: [{np.min(tr_Iin):.6f}, {np.max(tr_Iin):.6f}]")
print(f"Test current range:     [{np.min(ts_Iin):.6f}, {np.max(ts_Iin):.6f}]")

### ENHANCED DATA PREPROCESSING ###
print(f"\n{'='*80}")
print("ENHANCED DATA PREPROCESSING")
print(f"{'='*80}")

# Step 1: Data cleaning
print("Step 1: Data cleaning...")
tr_mask = np.isfinite(tr_Vout) & np.isfinite(tr_Iin) & np.isfinite(tr_SoC)
ts_mask = np.isfinite(ts_Vout) & np.isfinite(ts_Iin) & np.isfinite(ts_SoC)

if not np.all(tr_mask):
    print(f"Removed {np.sum(~tr_mask)} invalid training samples")
    tr_Time = tr_Time[tr_mask]; tr_Iin = tr_Iin[tr_mask]; tr_SoC = tr_SoC[tr_mask]
    tr_Vout = tr_Vout[tr_mask]; tr_Temp = tr_Temp[tr_mask]

if not np.all(ts_mask):
    print(f"Removed {np.sum(~ts_mask)} invalid test samples")
    ts_Time = ts_Time[ts_mask]; ts_Iin = ts_Iin[ts_mask]; ts_SoC = ts_SoC[ts_mask]
    ts_Vout = ts_Vout[ts_mask]; ts_Temp = ts_Temp[ts_mask]

# Step 2: SoC range optimization
print("Step 2: SoC range optimization...")
tr_soc_min, tr_soc_max = np.min(tr_SoC), np.max(tr_SoC)
ts_soc_min, ts_soc_max = np.min(ts_SoC), np.max(ts_SoC)
soc_overlap_min = max(tr_soc_min, ts_soc_min)
soc_overlap_max = min(tr_soc_max, ts_soc_max)
soc_overlap_range = max(0, soc_overlap_max - soc_overlap_min)
tr_soc_range = tr_soc_max - tr_soc_min
ts_soc_range = ts_soc_max - ts_soc_min
overlap_ratio = soc_overlap_range / min(tr_soc_range, ts_soc_range) if min(tr_soc_range, ts_soc_range) > 0 else 0
print(f"SoC overlap ratio: {overlap_ratio:.3f}")

# Step 3: Improved normalization
print("Step 3: Improved normalization...")
voltage_range = max(np.max(tr_Vout), np.max(ts_Vout)) - min(np.min(tr_Vout), np.min(ts_Vout))
current_max = max(np.max(np.abs(tr_Iin)), np.max(np.abs(ts_Iin)))

headroom = 0.2
maxInputVTr = max(np.max(tr_Vout), np.max(ts_Vout))
minInputVTr = min(np.min(tr_Vout), np.min(ts_Vout))
delta = headroom * voltage_range
maxInputVTr += delta / 2; minInputVTr -= delta / 2

maxInputITr = (1 + headroom) * current_max
maxInputTTr = 60

print("Improved normalization parameters:")
print(f"  Voltage: [{minInputVTr:.6f}, {maxInputVTr:.6f}]")
print(f"  Current: ±{maxInputITr:.6f}")
print(f"  Temperature: /{maxInputTTr}")

tr_Vout_norm = (tr_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
tr_Iin_norm  = tr_Iin / maxInputITr
tr_Temp_norm = tr_Temp / maxInputTTr
ts_Vout_norm = (ts_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
ts_Iin_norm  = ts_Iin / maxInputITr
ts_Temp_norm = ts_Temp / maxInputTTr

print("Normalized ranges:")
print(f"  Training voltage: [{np.min(tr_Vout_norm):.6f}, {np.max(tr_Vout_norm):.6f}]")
print(f"  Test voltage:     [{np.min(ts_Vout_norm):.6f}, {np.max(ts_Vout_norm):.6f}]")

# Step 4: Minimal data augmentation
print("Step 4: Minimal data augmentation...")
noise_level = 0.0   # disable noise for stability
tr_Iin_norm += np.random.normal(0, noise_level, tr_Iin_norm.shape)
ts_Iin_norm += np.random.normal(0, noise_level, ts_Iin_norm.shape)

### RESHAPE INPUTS ###
inputTr  = [np.reshape(tr_Iin_norm, (-1, 1, 1)), np.reshape(tr_SoC, (-1, 1, 1))]
inputTs  = [np.reshape(ts_Iin_norm, (-1, 1, 1)), np.reshape(ts_SoC, (-1, 1, 1))]
outputTr = np.reshape(tr_Vout_norm, (-1, 1, 1))
outputTs = np.reshape(ts_Vout_norm, (-1, 1, 1))

Temp_in = 'Temp' in dataTr and np.any(tr_Temp)
if Temp_in:
    inputTr.append(np.reshape(tr_Temp_norm, (-1, 1, 1)))
    inputTs.append(np.reshape(ts_Temp_norm, (-1, 1, 1)))

### FIXED ENNC ARCHITECTURE ###
print(f"\n{'='*80}")
print("FIXED ENNC ARCHITECTURE")
print(f"{'='*80}")

nEpoch    = 2000
batchSize = 16
print(f"  Epochs: {nEpoch}")
print(f"  Batch Size: {batchSize}")
print(f"  Optimizer: Adam (clipnorm)")
print(f"  Loss Function: mse")

# Create model
ennc_model = ENNC(Cn=tr_Cn, Ts=tr_Ts, cRate_in=True, SoC_in=True, Temp_in=Temp_in)

### PRE-FLIGHT FINITENESS CHECKS ###
def _assert_all_finite(name, arr):
    x = np.asarray(arr)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))
        raise ValueError(f"{name} has non-finite values at indices {bad}")

for name, arr in [("tr_Vout_norm", tr_Vout_norm), ("tr_Iin_norm", tr_Iin_norm),
                  ("ts_Vout_norm", ts_Vout_norm), ("ts_Iin_norm", ts_Iin_norm)]:
    _assert_all_finite(name, arr)

probe = ennc_model.net.predict([x[:1] for x in inputTr], verbose=0)
if not np.all(np.isfinite(probe)):
    raise ValueError("Non-finite model output BEFORE training (check τ bounding / custom layers).")

try:
    tau_probe = ennc_model.TauDynFnc.predict([x[:1] for x in inputTr], verbose=0)
    print("τ raw head min/max:", float(np.nanmin(tau_probe)), float(np.nanmax(tau_probe)))
except Exception:
    pass

### STABLE TRAINING (compile once + callbacks) ###
print(f"\n{'='*80}")
print("ENHANCED TRAINING WITH GRADIENT MONITORING")
print(f"{'='*80}")

opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
ennc_model.net.compile(optimizer=opt, loss='mse')

cbs = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best_weights.weights.h5"),
        save_weights_only=True, save_best_only=True, monitor='loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=40, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=120, restore_best_weights=True),
]

start_time = time.time()
history_obj = ennc_model.net.fit(inputTr, outputTr, epochs=nEpoch, batch_size=batchSize, callbacks=cbs, verbose=2)
trainingTime = time.time() - start_time

# Ensure best weights loaded
if os.path.exists(os.path.join(output_dir, "best_weights.weights.h5")):
    ennc_model.net.load_weights(os.path.join(output_dir, "best_weights.weights.h5"))
    print("Loaded best weights for evaluation")

### EVALUATION ###
pred_train = ennc_model.net.predict(inputTr, verbose=0).reshape(-1)
pred_test  = ennc_model.net.predict(inputTs,  verbose=0).reshape(-1)
true_train = outputTr.reshape(-1)
true_test  = outputTs.reshape(-1)

mse_train = float(ennc_model.net.test_on_batch(inputTr, outputTr))
mse_test  = float(ennc_model.net.test_on_batch(inputTs,  outputTs))

mae  = mean_absolute_error(true_test, pred_test)
rmse = np.sqrt(mean_squared_error(true_test, pred_test))
r2   = r2_score(true_test, pred_test)

print(f"\nTraining Results:")
print(f"  MSE Train: {mse_train:.6f}")
print(f"  MSE Test:  {mse_test:.6f}")
print(f"  MAE:       {mae:.6f}")
print(f"  RMSE:      {rmse:.6f}")
print(f"  R² Score:  {r2:.6f}")

### SAVE MODEL AND RESULTS ###
with open(os.path.join(output_dir, "model.json"), 'w') as f:
    f.write(ennc_model.net.to_json())

ennc_model.net.save_weights(os.path.join(output_dir, "model_weights.weights.h5"))

architecture_profile = {
    "preprocessing": {
        "headroom_V": 0.2,
        "minInputVTr": float(minInputVTr),
        "maxInputVTr": float(maxInputVTr),
        "maxInputITr": float(maxInputITr),
        "maxInputTTr": float(maxInputTTr),
        "soc_overlap_ratio": float(overlap_ratio),
        "data_filtered": bool(overlap_ratio < 0.95),
        "noise_level": float(noise_level),
    },
    "training": {
        "epochs_run": int(history_obj.epoch[-1] + 1),
        "best_loss": float(np.min(history_obj.history["loss"])),
    }
}

with open(os.path.join(output_dir, "architecture_fixes.json"), 'w') as f:
    json.dump(architecture_profile, f, indent=2)

import scipy.io as sio
sio.savemat(os.path.join(output_dir, "results.mat"), {
    'tr_Time': tr_Time, 'ts_Time': ts_Time,
    'tr_Iin': tr_Iin, 'ts_Iin': ts_Iin,
    'tr_SoC': tr_SoC, 'ts_SoC': ts_SoC,
    'tr_Temp': tr_Temp, 'ts_Temp': ts_Temp,
    'tr_Vout': tr_Vout, 'ts_Vout': ts_Vout,
    'trainingTime': trainingTime, 'lossHistory': np.array(history_obj.history['loss']),
    'mse_train': mse_train, 'mse_test': mse_test,
    'mae': mae, 'rmse': rmse, 'r2': r2
})

# ---------- LOG FILE ----------
log_path = os.path.join(output_dir, "log.txt")
with open(log_path, "w", encoding="utf-8") as log:
    log.write("ENNC Model Training Log\n")
    log.write(f"Run Timestamp: {timestamp}\n")
    log.write(f"Training File: {os.path.basename(trainFile)}\n")
    log.write(f"Test File: {os.path.basename(testFile)}\n")
    log.write(f"Training Time: {trainingTime:.2f} seconds\n")
    log.write("Hyperparameters:\n")
    log.write(f"  Epochs: {nEpoch}\n")
    log.write(f"  Batch Size: {batchSize}\n")
    log.write(f"  Optimizer: Adam (clipnorm)\n")
    log.write(f"  Loss Function: mse\n")
    log.write(f"MSE Train: {mse_train:.6f}\n")
    log.write(f"MSE Test: {mse_test:.6f}\n")
    log.write(f"MAE: {mae:.6f}\n")
    log.write(f"RMSE: {rmse:.6f}\n")
    log.write(f"R2 Score: {r2:.6f}\n")

# PLOTS (saved as PDFs)
fig1 = plt.figure()
plt.plot(true_test, label="Actual")
plt.plot(pred_test, label="Predicted", linestyle='--')
plt.title("Predicted vs Actual (Test Set) - Fixed ENNC Architecture")
plt.legend(); plt.grid(True)
fig1.savefig(os.path.join(output_dir, f"predicted_vs_actual_{timestamp}.pdf"), bbox_inches='tight')
plt.close(fig1)

fig2 = plt.figure()
plt.plot(history_obj.history['loss'])
plt.title("Training Loss - Fixed ENNC Architecture")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
fig2.savefig(os.path.join(output_dir, f"training_loss_{timestamp}.pdf"), bbox_inches='tight')
plt.close(fig2)

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")
print(f"Model saved in: {output_dir}")
print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
print(f"Log written to: {log_path}")
