# ENNC - Integration Version

This branch packages ENNC for direct integration workflows with REC-EMS.

Main goal:
- Keep ENNC training/identification assets available.
- Provide a simple export path to generate `ennc_lite_model.json` for framework-side ESS integration.

## Repository Layout

- `ENNC/Ststem Identification/Models/`
  - RW train/test MAT datasets (`trainRW*.mat`, `testRW*.mat`)
  - trained model artifacts and reports
- `ENNC/Ststem Identification/build_ennc_lite_from_data.py`
  - core LUT builder from train RW MAT files
- `build_ennc_lite_for_framework.py`
  - root-level wrapper for export to REC-EMS compatible JSON
- `dataset_generator_RWx.py`
  - utility for generating RW train/test dataset pairs

## Quick Start

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Export ENNC-lite for REC-EMS

### Option A - One command from repo root (recommended)

```powershell
python .\build_ennc_lite_for_framework.py `
  --train_glob ".\ENNC\Ststem Identification\Models\trainRW*.mat" `
  --out_json ".\exports\ennc_lite_model.json" `
  --soc_points 41 `
  --i_points 61 `
  --temp_values 25
```

Output:
- `exports/ennc_lite_model.json`

This JSON can be copied into REC-EMS `data/` and used as:
- `"model": "ennc_lite"`
- `"ennc_lite_model": "data/ennc_lite_model.json"`

### Option B - Use the core builder directly

```powershell
python ".\ENNC\Ststem Identification\build_ennc_lite_from_data.py" `
  --train_glob ".\ENNC\Ststem Identification\Models\trainRW*.mat" `
  --out_json ".\exports\ennc_lite_model.json"
```

## Train ENNC (original workflow)

```powershell
Set-Location ".\ENNC\Ststem Identification"
python .\main_Train.py
```

Training logs and artifacts are written under:
- `ENNC/Ststem Identification/Models/`

## Notes

- The `ennc_lite` export is an ENNC-inspired LUT surrogate intended for fast framework optimization.
- For integration tests, use fixed seeds and matched GA settings when comparing with linear ESS.
