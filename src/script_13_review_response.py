"""
Script 13: Pre-Submission Review Response
Addendum Task 9 (BRIEF_addendum_tarea9_respuesta_revision.md) -- answers a simulated
pre-submission review (Major Revision, no critical findings) with re-analysis of the
existing dataset and already-trained pipelines. No new measurements.

Sections:
  9A. Per-fold distribution of the LOGO (LeaveOneGroupOut) out-of-condition validation,
      for the RF -- reporting variability across folds, not just the pooled aggregate.
  9B. A structured (grey-box) Arrhenius-form comparator for k_ext, evaluated under the
      same LOGO scheme as RF/GP -- the most demanding test of the central conclusion,
      since a 4-parameter model is identifiable with far fewer conditions than an RF.
  9C. Reviewer-requested reporting items: RF/GP hyperparameters (+ a small RF
      hyperparameter stability sweep under LOGO), physical-filter threshold sensitivity
      (0.40/0.50/0.60), and k_ext/k_deg parameter correlation from the fit covariance.

Reuses (read-only imports; does not modify) kinetic_pipeline.py, script_8_rsm_vs_rf.py,
script_9_grouped_validation.py and script_12_gp_and_consolidation.py.
"""

import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

from kinetic_pipeline import (TARGETS_MAP, RANDOM_STATE, load_dataset, fit_valid_curves,
                               augment_curves, build_rf_pipeline, run_compound_pipeline)
from script_8_rsm_vs_rf import parse_ethanol_pct, parse_ratio_val
from script_9_grouped_validation import loco_splits, _fit_best_rf, RF_PARAM_GRID, run_grouped_rf
from script_12_gp_and_consolidation import fit_gp_for_compound, fit_all_curves_no_filter

R_GAS = 8.314  # J / (mol K)


# =====================================================================================
# Section 9A -- per-fold LOGO distribution
# =====================================================================================

def run_grouped_rf_per_fold(df_aug, curvas_validas, random_state=RANDOM_STATE):
    """Same LeaveOneGroupOut scheme as script_9.run_grouped_rf (exhaustive: every
    operating condition is held out exactly once, no synthetic replicate of the held-out
    condition ever appears in that fold's training set), but keeps the R2/RMSE of EACH
    fold individually instead of only pooling predictions into one aggregate number."""
    id_to_condition = {c['id_cinetica']: f"{c['T_C']:.0f}C, {c['disolvente']}, {c['relacion']}"
                        for c in curvas_validas}
    X = df_aug[['T_C', 'disolvente', 'relacion']].reset_index(drop=True)
    y = df_aug['k_ext'].reset_index(drop=True).values
    groups = df_aug['id_cinetica'].reset_index(drop=True).values
    splits = loco_splits(groups)
    if splits is None:
        return None

    rows = []
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out = groups[test_idx][0]
        best_rf = _fit_best_rf(X_train, y_train, groups[train_idx], random_state)
        y_pred = best_rf.predict(X_test)
        r2_fold = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan
        rmse_fold = mean_squared_error(y_test, y_pred) ** 0.5
        rows.append({
            'held_out_condition_id': held_out,
            'held_out_condition': id_to_condition.get(held_out, str(held_out)),
            'n_test_rows': len(y_test), 'R2_fold': r2_fold, 'RMSE_fold': rmse_fold,
        })
    return pd.DataFrame(rows)


def build_fold_tables(df):
    detail_frames, summary_rows = [], []
    for comp_es, comp_en in TARGETS_MAP.items():
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)
        if 'df_aug' not in result or result.get('n_valid', 0) < 2:
            summary_rows.append({'Compound': comp_en, 'n_folds': result.get('n_valid', 0),
                                 'R2_mean': np.nan, 'R2_median': np.nan, 'R2_min': np.nan,
                                 'R2_max': np.nan, 'R2_IQR': np.nan, 'R2_std': np.nan,
                                 'RMSE_mean': np.nan, 'RMSE_std': np.nan,
                                 'note': 'Insufficient valid curves for LOGO (n<2)'})
            continue

        fold_df = run_grouped_rf_per_fold(result['df_aug'], result['curvas_validas'], random_state=RANDOM_STATE)
        if fold_df is None or fold_df.empty:
            summary_rows.append({'Compound': comp_en, 'n_folds': 0, 'R2_mean': np.nan,
                                 'R2_median': np.nan, 'R2_min': np.nan, 'R2_max': np.nan,
                                 'R2_IQR': np.nan, 'R2_std': np.nan, 'RMSE_mean': np.nan,
                                 'RMSE_std': np.nan, 'note': 'LOGO produced no folds'})
            continue

        fold_df.insert(0, 'Compound', comp_en)
        detail_frames.append(fold_df)

        r2 = fold_df['R2_fold'].dropna()
        q1, q3 = (r2.quantile(0.25), r2.quantile(0.75)) if len(r2) else (np.nan, np.nan)
        summary_rows.append({
            'Compound': comp_en, 'n_folds': len(fold_df),
            'R2_mean': r2.mean() if len(r2) else np.nan,
            'R2_median': r2.median() if len(r2) else np.nan,
            'R2_min': r2.min() if len(r2) else np.nan,
            'R2_max': r2.max() if len(r2) else np.nan,
            'R2_IQR': (q3 - q1) if len(r2) else np.nan,
            'R2_std': r2.std(ddof=1) if len(r2) > 1 else np.nan,
            'RMSE_mean': fold_df['RMSE_fold'].mean(), 'RMSE_std': fold_df['RMSE_fold'].std(ddof=1),
            'note': '',
        })

    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return summary, detail


# =====================================================================================
# Section 9B -- structured Arrhenius-form comparator
# =====================================================================================

def arrhenius_model(X, A, Ea, b1, b2):
    """k_ext(T, ethanol_pct, ratio_val) = A * exp(-Ea / (R*T_K)) * (1 + b1*ethanol_frac + b2*ratio_val)."""
    T_C, ethanol_pct, ratio_val = X
    T_K = T_C + 273.15
    ethanol_frac = ethanol_pct / 100.0
    f = 1 + b1 * ethanol_frac + b2 * ratio_val
    return A * np.exp(-Ea / (R_GAS * T_K)) * f


N_ARRHENIUS_PARAMS = 4


def _fit_arrhenius(X, y):
    """X shape (3, n). Returns popt or None if not estimable."""
    if X.shape[1] <= N_ARRHENIUS_PARAMS:
        return None
    p0 = [max(float(np.mean(y)) * 5, 1e-6), 50000.0, 0.0, 0.0]
    try:
        popt, _ = curve_fit(arrhenius_model, X, y, p0=p0,
                             bounds=([1e-10, 0, -50, -50], [np.inf, 300000, 50, 50]), maxfev=20000)
        return popt
    except (RuntimeError, ValueError):
        return None


def run_arrhenius_logo(curvas_validas):
    """Leave-one-condition-out on the real (unaugmented) valid curves -- the structured
    model does not need Monte Carlo augmentation, unlike the RF, since it has far fewer
    parameters than data points once n_valid > 4."""
    n = len(curvas_validas)
    if n <= N_ARRHENIUS_PARAMS + 1:
        return None, f'Not estimable: n_valid={n} <= {N_ARRHENIUS_PARAMS + 1} (need >1 spare condition for LOGO)'

    X_all = np.array([[float(c['T_C']), parse_ethanol_pct(c['disolvente']), parse_ratio_val(c['relacion'])]
                       for c in curvas_validas]).T
    y_all = np.array([c['k_ext'] for c in curvas_validas])

    y_true_all, y_pred_all = [], []
    n_fit_failures = 0
    for i in range(n):
        mask = np.arange(n) != i
        popt = _fit_arrhenius(X_all[:, mask], y_all[mask])
        if popt is None:
            n_fit_failures += 1
            continue
        y_pred = arrhenius_model(X_all[:, [i]], *popt)[0]
        y_true_all.append(y_all[i])
        y_pred_all.append(y_pred)

    if len(y_true_all) < 2:
        return None, f'Not estimable: LOGO refit failed in {n_fit_failures}/{n} folds'

    note = '' if n_fit_failures == 0 else f'{n_fit_failures}/{n} LOGO folds failed to converge'
    return {
        'r2': r2_score(y_true_all, y_pred_all),
        'rmse': mean_squared_error(y_true_all, y_pred_all) ** 0.5,
        'n_folds_fit': len(y_true_all),
    }, note


def build_structured_comparator_table(df, table_7_path):
    table_7 = pd.read_csv(table_7_path)
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
        curvas_validas, _ = fit_valid_curves(sub_df) if not sub_df.empty else ([], [])
        n_valid = len(curvas_validas)

        r2_rf_row = table_7.loc[table_7['Compound'] == comp_en, 'R2_grouped_LOCO']
        r2_rf = r2_rf_row.values[0] if len(r2_rf_row) else np.nan

        gp_result = fit_gp_for_compound(df, comp_es=comp_es) if comp_es == 'Fenoles' else None
        r2_gp = gp_result['r2_loco'] if gp_result is not None else np.nan
        gp_note = '' if comp_es == 'Fenoles' else 'GP not evaluated for this compound (Phenolics-only scope, Task 7)'

        arrhenius_result, arrhenius_note = run_arrhenius_logo(curvas_validas) if n_valid >= 2 else (None, 'No valid curves')
        rows.append({
            'Compound': comp_en, 'n_conditions': n_valid,
            'R2_arrhenius_LOGO': arrhenius_result['r2'] if arrhenius_result else np.nan,
            'R2_RF_LOGO': r2_rf,
            'R2_GP_LOGO': r2_gp,
            'RMSE_arrhenius': arrhenius_result['rmse'] if arrhenius_result else np.nan,
            'notes': '; '.join(n for n in [arrhenius_note, gp_note] if n),
        })
    return pd.DataFrame(rows)


# =====================================================================================
# Section 9C.1 -- RF/GP hyperparameters + small RF stability sweep under LOGO
# =====================================================================================

RF_SWEEP_N_ESTIMATORS = [50, 100, 200, 300]
RF_SWEEP_MAX_DEPTH = [None, 5, 10, 15]


def get_full_fit_best_params(df):
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)
        if result.get('insufficient') or 'grid' not in result:
            rows.append({'Compound': comp_en, 'best_params': 'N/A (insufficient data)'})
            continue
        rows.append({'Compound': comp_en, 'best_params': str(result['grid'].best_params_)})
    return rows


def run_rf_hyperparam_sweep(df):
    """Fixed (non-tuned) RF at each candidate hyperparameter value, evaluated under the
    same LOGO scheme, to check the negative out-of-condition result is not an artifact of
    the specific GridSearchCV grid used elsewhere (analogous to the sigma sweep, Task 6A)."""
    from sklearn.base import clone
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)
        if 'df_aug' not in result or result.get('n_valid', 0) < 2:
            continue
        df_aug = result['df_aug']
        X = df_aug[['T_C', 'disolvente', 'relacion']].reset_index(drop=True)
        y = df_aug['k_ext'].reset_index(drop=True).values
        groups = df_aug['id_cinetica'].reset_index(drop=True).values
        splits = loco_splits(groups)
        if splits is None:
            continue

        for param_name, values in [('n_estimators', RF_SWEEP_N_ESTIMATORS), ('max_depth', RF_SWEEP_MAX_DEPTH)]:
            for val in values:
                pipe = build_rf_pipeline(RANDOM_STATE)
                pipe.set_params(**{f'model__{param_name}': val})
                y_true_all, y_pred_all = [], []
                for train_idx, test_idx in splits:
                    fold_pipe = clone(pipe)
                    fold_pipe.fit(X.iloc[train_idx], y[train_idx])
                    y_pred_all.extend(fold_pipe.predict(X.iloc[test_idx]))
                    y_true_all.extend(y[test_idx])
                rows.append({
                    'Compound': comp_en, 'swept_param': param_name, 'value': str(val),
                    'R2_grouped_LOGO': r2_score(y_true_all, y_pred_all),
                    'RMSE_grouped_LOGO': mean_squared_error(y_true_all, y_pred_all) ** 0.5,
                })
    return pd.DataFrame(rows)


def write_hyperparameters_report(df, results_dir):
    best_params = get_full_fit_best_params(df)
    sweep = run_rf_hyperparam_sweep(df)
    sweep_path = os.path.join(results_dir, 'rf_hyperparameter_sweep_logo.csv')
    sweep.round(4).to_csv(sweep_path, index=False)

    lines = ["# model_hyperparameters.md — RF and GP hyperparameters (addendum, review response)\n"]
    lines.append("## Random Forest (k_ext regressor)\n")
    lines.append("Search grid (`GridSearchCV`, `cv=3`, `scoring='r2'`), tuned on the training fold each time "
                  "(random split for Table 3; grouped inner CV for every LOGO fold elsewhere):\n")
    lines.append(f"- `n_estimators`: {RF_PARAM_GRID['model__n_estimators']}")
    lines.append(f"- `max_depth`: {RF_PARAM_GRID['model__max_depth']}")
    lines.append(f"- `min_samples_split`: {RF_PARAM_GRID['model__min_samples_split']}\n")
    lines.append("Selected values (best_params_, full-data fit on the augmented dataset, per compound):\n")
    for row in best_params:
        lines.append(f"- **{row['Compound']}**: `{row['best_params']}`")

    lines.append("\n## Gaussian Process (k_ext regressor, Phenolics only)\n")
    lines.append("Kernel: `ConstantKernel() * RBF(length_scale=[1,1,1]) + WhiteKernel()`, anisotropic "
                  "over `[T, ethanol_pct, ratio_val]` (standardized). Hyperparameters (kernel amplitude, "
                  "RBF length-scales, white-noise level) are optimized by maximizing the log marginal "
                  "likelihood (`GaussianProcessRegressor` default, L-BFGS-B, `n_restarts_optimizer=5`) — "
                  "not manually chosen.\n")

    lines.append("\n## RF hyperparameter stability sweep under out-of-condition (LOGO) validation\n")
    lines.append("Fixed (untuned) RF at each candidate value, evaluated under the same LeaveOneGroupOut "
                  "scheme as the main result, to check the negative out-of-condition R2 is not an artifact "
                  "of the specific GridSearchCV grid. Full data in `results/rf_hyperparameter_sweep_logo.csv`.\n")
    if not sweep.empty:
        for comp_en, grp in sweep.groupby('Compound', sort=False):
            r2_range = f"[{grp['R2_grouped_LOGO'].min():.3f}, {grp['R2_grouped_LOGO'].max():.3f}]"
            lines.append(f"- **{comp_en}**: R2 range across the sweep = {r2_range} "
                          f"({'stays negative throughout' if grp['R2_grouped_LOGO'].max() < 0 else 'crosses into positive territory somewhere in the sweep'})")

    report_path = os.path.join(results_dir, 'model_hyperparameters.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Hyperparameters report written to: {report_path}")
    return sweep


# =====================================================================================
# Section 9C.2 -- physical-validity threshold sensitivity
# =====================================================================================

THRESHOLDS = [0.40, 0.50, 0.60]


def build_threshold_sensitivity_table(df):
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
        curvas_all = fit_all_curves_no_filter(sub_df) if not sub_df.empty else []
        for threshold in THRESHOLDS:
            curvas_thr = [c for c in curvas_all if c['r2_fit'] >= threshold]
            if len(curvas_thr) < 2:
                rows.append({'Compound': comp_en, 'threshold': threshold, 'n_valid': len(curvas_thr),
                             'R2_grouped_LOGO': np.nan, 'RMSE_grouped_LOGO': np.nan,
                             'note': 'n<2, LOGO not applicable'})
                continue
            df_aug, _ = augment_curves(curvas_thr, n_synth=30, band=0.60, seed=RANDOM_STATE)
            grouped = run_grouped_rf(df_aug, random_state=RANDOM_STATE)
            rows.append({
                'Compound': comp_en, 'threshold': threshold, 'n_valid': len(curvas_thr),
                'R2_grouped_LOGO': grouped['r2'] if grouped else np.nan,
                'RMSE_grouped_LOGO': grouped['rmse'] if grouped else np.nan, 'note': '',
            })
    return pd.DataFrame(rows)


# =====================================================================================
# Section 9C.3 -- k_ext / k_deg parameter correlation (from fit covariance)
# =====================================================================================

def ext_deg_model(t, C0, k_ext, k_deg):
    denom = k_deg - k_ext
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return C0 * (k_ext / denom) * (np.exp(-k_ext * t) - np.exp(-k_deg * t))


def build_kinetic_parameter_correlation_table(df):
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
        if sub_df.empty:
            continue
        for cid in sub_df['id_cinetica'].unique():
            datos = sub_df[sub_df['id_cinetica'] == cid].sort_values('tiempo_min')
            if len(datos) < 4:
                continue
            t, C = datos['tiempo_min'].values, datos['conc_promedio'].values
            try:
                popt, pcov = curve_fit(ext_deg_model, t, C, p0=[max(C) * 1.5, 0.05, 0.005],
                                        bounds=([0, 0, 0], [np.inf, 1.0, 1.0]), maxfev=10000)
            except RuntimeError:
                continue
            r2 = r2_score(C, ext_deg_model(t, *popt))
            if r2 < 0.50:
                continue
            var_kext, var_kdeg, cov_kext_kdeg = pcov[1, 1], pcov[2, 2], pcov[1, 2]
            corr = (cov_kext_kdeg / np.sqrt(var_kext * var_kdeg)
                    if var_kext > 0 and var_kdeg > 0 else np.nan)
            kdeg_std = np.sqrt(var_kdeg) if var_kdeg >= 0 else np.nan
            rows.append({
                'Compound': comp_en, 'id_cinetica': cid,
                'T_C': datos['T_C'].iloc[0], 'disolvente': datos['disolvente'].iloc[0],
                'relacion': datos['relacion'].iloc[0], 'r2_fit': r2,
                'k_ext': popt[1], 'k_deg': popt[2],
                'corr_k_ext_k_deg': corr,
                'k_deg_std': kdeg_std,
                'k_deg_CI95_lower': popt[2] - 1.96 * kdeg_std if pd.notna(kdeg_std) else np.nan,
                'k_deg_CI95_upper': popt[2] + 1.96 * kdeg_std if pd.notna(kdeg_std) else np.nan,
            })
    return pd.DataFrame(rows)


# =====================================================================================
# Orchestration
# =====================================================================================

def main(data_path='../data/dataset.csv', results_dir='../results'):
    os.makedirs(results_dir, exist_ok=True)
    df = load_dataset(data_path)

    print("=" * 80); print("9A: per-fold LOGO distribution"); print("=" * 80)
    summary, detail = build_fold_tables(df)
    summary_path = os.path.join(results_dir, 'paper_table_validation_folds.csv')
    detail_path = os.path.join(results_dir, 'validation_fold_detail.csv')
    summary.round(4).to_csv(summary_path, index=False)
    detail.round(4).to_csv(detail_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"[Saved to {summary_path}, {detail_path}]\n")

    print("=" * 80); print("9B: structured Arrhenius comparator under LOGO"); print("=" * 80)
    table_7_path = os.path.join(results_dir, 'table_7_grouped_validation.csv')
    comparator = build_structured_comparator_table(df, table_7_path)
    comparator_path = os.path.join(results_dir, 'paper_table_structured_comparator.csv')
    comparator.round(4).to_csv(comparator_path, index=False)
    print(comparator.round(4).to_string(index=False))
    print(f"[Saved to {comparator_path}]\n")

    print("=" * 80); print("9C.1: hyperparameters + RF stability sweep"); print("=" * 80)
    write_hyperparameters_report(df, results_dir)

    print("=" * 80); print("9C.2: physical-validity threshold sensitivity"); print("=" * 80)
    threshold_table = build_threshold_sensitivity_table(df)
    threshold_path = os.path.join(results_dir, 'threshold_sensitivity.csv')
    threshold_table.round(4).to_csv(threshold_path, index=False)
    print(threshold_table.round(4).to_string(index=False))
    print(f"[Saved to {threshold_path}]\n")

    print("=" * 80); print("9C.3: k_ext / k_deg parameter correlation"); print("=" * 80)
    corr_table = build_kinetic_parameter_correlation_table(df)
    corr_path = os.path.join(results_dir, 'kinetic_parameter_correlation.csv')
    corr_table.round(4).to_csv(corr_path, index=False)
    print(corr_table.round(4).to_string(index=False))
    print(f"[Saved to {corr_path}]\n")

    print("Done.")


if __name__ == "__main__":
    main()
