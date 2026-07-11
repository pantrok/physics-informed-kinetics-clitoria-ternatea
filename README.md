# Physics-Informed Kinetic Modeling of *Clitoria ternatea* Extraction (QbD/PAT reinforcement)

Computational framework for the extraction-kinetics chain of the *Clitoria ternatea* study, reoriented
to the ENBIS special issue *"Statistical, AI and Grey Approaches for QbD and PAT"* (Journal of
Chemometrics, Wiley). This repository is the reinforced, physics-informed successor to the original
`Code/` repository: it drops the circular absorbance-based instantaneous-inference stage and deepens
the phenomenological/ML kinetic chain with uncertainty, design-space, and methodological-robustness
analyses. No new laboratory measurements were made — everything here is re-analysis of the existing
`data/dataset.csv`.

## Why the reorientation

The original manuscript's Stage 1 (instantaneous concentration inference from absorbance) derived its
reference concentrations from the same absorbances via Beer-Lambert law, making that inference task
circular (R²=1.000 / RMSE=0.000 by construction) — not a defensible chemometric result. `conc_promedio`
values are real reference measurements at each timepoint, so the kinetic chain (Stage 2-3 below) is fed
by genuine experimental data and loses nothing by dropping Stage 1.

**What is kept and reinforced:** the phenomenological Ext-Deg fit → physical R²≥0.50 filter → Monte
Carlo physics-informed augmentation (±60% variance filter) → Random Forest mapping of process variables
`[T, solvent, ratio] → k_ext` (no absorbance involved). This is the central contribution, now extended
with confidence intervals, design-space response surfaces, an RSM-vs-RF robustness comparison, and an
augmentation validation study.

## Repository structure

```
/src        analysis scripts (.py)
/data       dataset.csv (measured data, factorial extraction design + UV-Vis responses)
/results    output tables (.csv)
/figures    output figures (.png) + draft captions
```

## Scripts (`/src`)

Active pipeline (current manuscript scope):

- `script_1_eda_pairplot.py` — exploratory data analysis, bivariate pairplot (Fig. 2). Independent of
  the Stage-1 circularity issue; kept as-is.
- `kinetic_pipeline.py` — shared physics-informed pipeline (Ext-Deg fit, physical filter, Monte Carlo
  augmentation, RF training) used by every script below. Not a standalone deliverable; refactored out
  of `script_5` so Tasks 1-4 reuse identical, reproducible logic instead of re-deriving it.
- `script_5_augmented_kinetic_prediction.py` — phenomenological fit (Table 2), augmented RF prediction
  of k_ext (Table 3), **k_ext 95% confidence intervals** (Table 4) and parity plots with error bars
  (Fig. 6).
- `script_6_response_surfaces.py` — k_ext response surfaces over the measured design space (Fig. 7),
  model prediction only, never extrapolated beyond T∈[60,75]°C.
- `script_7_augmentation_validation.py` — augmentation ablation curve (Fig. 8) and variance-filter
  (±60%) acceptance/rejection audit (Table 6).
- `script_8_rsm_vs_rf.py` — classical 2nd-order RSM (Box-Behnken ANOVA) vs. RF comparison on the same
  train/test split of the real valid-curve data (Table 5).

Legacy / deprecated (kept for historical reference only — **not** part of the current manuscript
scope, **not** run as part of reproduction):

- `script_2_instantaneous_inference.py`, `script_3_feature_importance.py`,
  `script_4_ablation_study_global.py` — the eliminated Stage-1 absorbance-based instantaneous
  inference, feature importance, and ablation study (circular w.r.t. Beer-Lambert; see "Why the
  reorientation" above).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Reproducing the figures and tables

All scripts assume they are run from inside `/src` (they reference `../data` and `../results`/`../figures`
with relative paths) and use `random_state=42` throughout for reproducibility:

```bash
cd src
python script_1_eda_pairplot.py              # -> figures/fig_2_pairplot_eda.png
python script_5_augmented_kinetic_prediction.py  # -> results/table_2,3,4 + figures/fig_6
python script_6_response_surfaces.py         # -> figures/fig_7_response_surfaces.png
python script_7_augmentation_validation.py   # -> results/table_ablation, table_6 + figures/fig_8
python script_8_rsm_vs_rf.py                 # -> results/table_5, table_5b
```

## Key outputs

| File | Description |
|---|---|
| `results/table_2_phenomenological_fit.csv` | Ext-Deg fit quality, valid curve counts per compound |
| `results/table_3_augmented_ml_prediction.csv` | Final RF k_ext prediction metrics (augmented dataset) |
| `results/table_4_kext_confidence_intervals.csv` | k_ext 95% CI per operating condition |
| `results/table_5_rsm_vs_rf.csv`, `table_5b_rsm_anova.csv` | Classical RSM vs. RF comparison + ANOVA |
| `results/table_6_variance_filter_audit.csv`, `table_6b_*_summary.csv` | ±60% variance filter accept/reject audit |
| `results/table_ablation_r2_vs_augmentation_size.csv` | Augmentation ablation data |
| `figures/fig_6_ml_augmented_parity.png` | k_ext parity plots with 95% RF prediction-interval error bars |
| `figures/fig_7_response_surfaces.png` | k_ext response surfaces over the design space (model prediction) |
| `figures/fig_8_augmentation_ablation.png` | R² vs. augmented dataset size |
| `figures/CAPTIONS_borrador.md` | Draft figure captions for Fig. 7 / Fig. 8 |
| `RESULTADOS_resumen.md` | Consolidated, publication-precision numbers for the manuscript (single source of truth) |
| `ANOMALIAS.md` | Honest log of unexpected findings (DPPH data limitation, RSM non-estimability, etc.) |

## Methodological guardrails

- Design-space mesh predictions are interpolation strictly within the measured range
  (T∈[60,75]°C, the 3 solvents and 3 ratios of the design) — never extrapolated, and always labeled
  as model predictions, never as experimental data.
- `random_state=42` and fixed seeds throughout for reproducibility.
- No additional ML algorithms beyond the already-justified Random Forest (no XGBoost/SVR/neural nets).
- No t-tests or correlation matrices added; the only inferential analysis is the RSM ANOVA (Task 3),
  the standard analysis for a Box-Behnken design.

## Data availability

`data/dataset.csv` contains the factorial extraction design variables and UV-Vis spectrophotometric
responses for *Clitoria ternatea* extracts (Box-Behnken design: temperature, solvent composition,
solid:liquid ratio; 4 target compounds: anthocyanins, phenolics, ABTS and DPPH antioxidant capacity).
