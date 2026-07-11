"""
Script 11: Absorbance Replicate CV by Compound (Table 12)
Addendum Task 6C (BRIEF_addendum_tarea6_sigma_y_fisica.md).

Quantifies the instrumental measurement uncertainty of the absorbance triplicate
(A_replicate_1/2/3 -- three repeated absorbance readings of the SAME sample, not
distinct spectral bands; see the Task-6 §1 rename and ANOMALIAS.md §0). This is
direct Methods-section material and the empirical anchor for the sigma sensitivity
analysis in script_10 (Task 6A): sigma=0.08 is compared against the CV measured here.
"""

import os

import numpy as np
import pandas as pd

from kinetic_pipeline import TARGETS_MAP, load_dataset

A_MEAN_FLOOR = 0.005  # avoid dividing by near-zero absorbance


def compute_replicate_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_cv = []
    for comp_es, comp_en in TARGETS_MAP.items():
        sub = df[df['compuesto'] == comp_es]
        reps = sub[['A_replicate_1', 'A_replicate_2', 'A_replicate_3']]
        a_mean = reps.mean(axis=1)
        a_std = reps.std(axis=1, ddof=1)
        mask = a_mean > A_MEAN_FLOOR
        cv_pct = (a_std[mask] / a_mean[mask]) * 100
        all_cv.append(cv_pct)

        rows.append({
            'Compound': comp_en, 'n_obs': int(mask.sum()),
            'A_mean_min': a_mean[mask].min() if mask.any() else np.nan,
            'A_mean_max': a_mean[mask].max() if mask.any() else np.nan,
            'CV_median_pct': cv_pct.median(), 'CV_mean_pct': cv_pct.mean(),
            'CV_P25_pct': cv_pct.quantile(0.25), 'CV_P75_pct': cv_pct.quantile(0.75),
            'CV_of_mean_of_3_median_pct': cv_pct.median() / np.sqrt(3),
            'CV_of_mean_of_3_mean_pct': cv_pct.mean() / np.sqrt(3),
        })

    global_cv = pd.concat(all_cv, ignore_index=True)
    rows.append({
        'Compound': 'ALL (global)', 'n_obs': int(global_cv.shape[0]),
        'A_mean_min': np.nan, 'A_mean_max': np.nan,
        'CV_median_pct': global_cv.median(), 'CV_mean_pct': global_cv.mean(),
        'CV_P25_pct': global_cv.quantile(0.25), 'CV_P75_pct': global_cv.quantile(0.75),
        'CV_of_mean_of_3_median_pct': global_cv.median() / np.sqrt(3),
        'CV_of_mean_of_3_mean_pct': global_cv.mean() / np.sqrt(3),
    })
    return pd.DataFrame(rows)


def generate_replicate_cv(data_path: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    df = load_dataset(data_path)
    table_12 = compute_replicate_cv(df)

    t12_path = os.path.join(results_dir, 'table_12_replicate_cv_by_compound.csv')
    table_12.round(3).to_csv(t12_path, index=False)

    print("=" * 80)
    print("TABLE 12: Absorbance replicate (triplicate) CV by compound")
    print("=" * 80)
    print(table_12.round(3).to_string(index=False))
    print(f"[Saved to {t12_path}]\n")

    anthocyanins_row = table_12[table_12['Compound'] == 'Anthocyanins'].iloc[0]
    print(f"NOTE -- Anthocyanins: CV_median={anthocyanins_row['CV_median_pct']:.1f}%, "
          f"absorbance range [{anthocyanins_row['A_mean_min']:.4f}, {anthocyanins_row['A_mean_max']:.4f}] "
          f"-- near the detection limit, elevating relative noise. Honest limitation, not an error "
          f"(see ANOMALIAS.md).")


if __name__ == "__main__":
    generate_replicate_cv(os.path.join('..', 'data', 'dataset.csv'), os.path.join('..', 'results'))
