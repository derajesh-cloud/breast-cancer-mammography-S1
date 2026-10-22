# Supplementary File S1 — code, data checksums and raw outputs

Reproduces every number, table and figure in the manuscript
"Integrating Structured Mammography Report Descriptors with Image-Derived Features for Breast Cancer Diagnosis".

## Data (all public, de-identified)
| File | Dataset | Source | SHA-256 |
|---|---|---|---|
| `data/mammographic.dat` | Mammographic Mass (Elter et al. 2007), 830 complete cases | UCI id 161; complete-case copy in the KEEL repository (PyPI package `keel-ds` 0.2.5, `keel_ds/data/balanced/raw/mammographic.dat`) | 7fc6e6c0c63cb59eff6faf8e0a9c74993a1572d37dcddd70be78ac3eca0f2477 |
| `data/woods_mammography.csv` | Mammography microcalcification features (Woods et al. 1993), 11,183 regions | imbalanced-learn benchmark collection; mirror github.com/jbrownlee/Datasets (`mammography.csv`) | 7d3dea3f075f30bbdbb8980e7725684059b4fe3d0f850fadd0f635f6993d8730 |
| (bundled) | Wisconsin Diagnostic Breast Cancer (Street et al. 1993), 569 cases | `sklearn.datasets.load_breast_cancer()` (UCI id 17) | — |

Column order of `mammographic.dat`: BI-RADS, age, shape, margin, density, severity (0 benign, 1 malignant).

## Run
```
pip install scikit-learn==1.9.1 xgboost==3.4.1 imbalanced-learn==0.14.2 shap==0.54.1 pandas matplotlib scipy
python code/run_experiments.py      # ~25 min on 2 CPU cores; writes results/ and figs/
python code/make_fig1.py            # workflow diagram
```
`QUICK=1 python code/run_experiments.py` runs a fast smoke test (fewer folds), not used for any reported number.

## Outputs
- `results/results.json` — all metrics, DeLong tests, calibration, decision-curve values, SHAP rankings, latency, software versions
- `results/C1_*.csv`, `results/C2_imbalance.csv`, `results/C3_wdbc.csv` — per-model tables
- `figs/*.png` — figures at 300 dpi
