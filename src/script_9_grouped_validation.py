"""
Script 9: Grouped (Leave-One-Condition-Out) Validation of the k_ext Random Forest
Addendum Task 5 (BRIEF_addendum_tarea5_validacion_agrupada.md) -- leakage diagnostic.

script_5's Table 3 evaluates the k_ext Random Forest with a ROW-WISE
train_test_split(random_state=42) on df_aug, which holds ~30 Monte Carlo replicates per
operating condition (all sharing the same id_cinetica). That split can place replicates
of the SAME condition on both sides, so the RF may be recalling an already-seen
condition's mean rather than predicting an unseen one (condition-identity leakage).

This script re-evaluates the IDENTICAL pipeline (kinetic_pipeline.run_compound_pipeline)
under LeaveOneGroupOut cross-validation grouped by id_cinetica: every replicate of a
held-out condition is excluded from training for that fold. It also builds two
interpretable baselines (global-mean, nearest-condition) and re-runs the augmentation
ablation curve under the same grouped scheme, for direct contrast with the old
row-wise-split ablation (Task 4).

Does NOT modify script_5 / kinetic_pipeline.py or any Task 1-4 output; writes new files only.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV

from kinetic_pipeline import TARGETS_MAP, RANDOM_STATE, load_dataset, run_compound_pipeline, build_rf_pipeline
from script_8_rsm_vs_rf import parse_ethanol_pct, parse_ratio_val

RF_PARAM_GRID = {'model__n_estimators': [50, 100, 200],
                  'model__max_depth': [None, 5, 10],
                  'model__min_samples_split': [2, 5]}
ABLATION_N_SYNTH = [0, 10, 30, 60, 90]


def loco_splits(groups):
    """List of (train_idx, test_idx) for LeaveOneGroupOut, or None if <2 groups."""
    if len(np.unique(groups)) < 2:
        return None
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def _fit_best_rf(X_train, y_train, train_groups, random_state):
    """Fit the RF pipeline, tuning hyperparameters with a GROUPED inner CV (GroupKFold)
    whenever possible, so the identity-leakage bug being diagnosed is not silently
    reintroduced one level down, inside the hyperparameter search itself."""
    pipe = build_rf_pipeline(random_state)
    n = len(X_train)
    if n < 2:
        pipe.fit(X_train, y_train)
        return pipe
    n_inner_groups = len(np.unique(train_groups))
    if n_inner_groups >= 2:
        cv = GroupKFold(n_splits=min(3, n_inner_groups))
        grid = GridSearchCV(pipe, RF_PARAM_GRID, cv=cv, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train, groups=train_groups)
        return grid.best_estimator_
    cv = min(3, n)
    if cv < 2:
        pipe.fit(X_train, y_train)
        return pipe
    grid = GridSearchCV(pipe, RF_PARAM_GRID, cv=cv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def run_grouped_rf(df_aug, random_state=RANDOM_STATE):
    """LOCO evaluation: for each held-out condition, tune+fit on all other conditions,
    predict the held-out one, and pool predictions across folds into a single
    out-of-group R2/RMSE (the number that matters for the leakage diagnosis)."""
    X = df_aug[['T_C', 'disolvente', 'relacion']].reset_index(drop=True)
    y = df_aug['k_ext'].reset_index(drop=True).values
    groups = df_aug['id_cinetica'].reset_index(drop=True).values
    splits = loco_splits(groups)
    if splits is None:
        return None

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        best_rf = _fit_best_rf(X_train, y_train, groups[train_idx], random_state)
        y_pred = best_rf.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return {
        'r2': r2_score(y_true_all, y_pred_all),
        'rmse': mean_squared_error(y_true_all, y_pred_all) ** 0.5,
        'n_groups': len(np.unique(groups)),
    }


def run_grouped_baselines(df_aug, curvas_validas):
    """Two interpretable out-of-group baselines: (1) global-mean of k_ext in train,
    (2) mean k_ext of the nearest training CONDITION in standardized (T, ethanol_pct,
    ratio_val) space. Neither uses the RF; both are computed under the same LOCO splits
    so they are directly comparable to run_grouped_rf's R2."""
    X_groups = df_aug['id_cinetica'].reset_index(drop=True).values
    y = df_aug['k_ext'].reset_index(drop=True).values
    splits = loco_splits(X_groups)
    if splits is None:
        return None

    factors = {c['id_cinetica']: np.array([float(c['T_C']), parse_ethanol_pct(c['disolvente']),
                                            parse_ratio_val(c['relacion'])]) for c in curvas_validas}
    scale = np.array(list(factors.values())).std(axis=0)
    scale[scale == 0] = 1.0

    y_true_all, y_pred_global, y_pred_nearest = [], [], []
    for train_idx, test_idx in splits:
        y_train = y[train_idx]
        train_groups = np.unique(X_groups[train_idx])
        test_group = np.unique(X_groups[test_idx])[0]

        global_mean = y_train.mean()
        group_means = {g: y[X_groups == g].mean() for g in train_groups}
        test_vec = factors[test_group] / scale
        nearest_g = min(train_groups, key=lambda g: np.linalg.norm(factors[g] / scale - test_vec))

        n_test = int((X_groups[test_idx] == test_group).sum())
        y_true_all.extend(y[test_idx])
        y_pred_global.extend([global_mean] * n_test)
        y_pred_nearest.extend([group_means[nearest_g]] * n_test)

    return {
        'r2_global_mean': r2_score(y_true_all, y_pred_global),
        'r2_nearest_condition': r2_score(y_true_all, y_pred_nearest),
    }


def evaluate_compound(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE):
    result = run_compound_pipeline(df, comp_es, comp_en, n_synth=n_synth, band=band, seed=seed)
    if 'df_aug' not in result or result.get('n_valid', 0) < 2:
        return None
    grouped = run_grouped_rf(result['df_aug'], random_state=seed)
    baselines = run_grouped_baselines(result['df_aug'], result['curvas_validas'])
    return {'result': result, 'grouped': grouped, 'baselines': baselines}


def build_tables_7_and_8(df):
    t7_rows, t8_rows = [], []
    for comp_es, comp_en in TARGETS_MAP.items():
        ev = evaluate_compound(df, comp_es, comp_en)
        if ev is None or ev['grouped'] is None:
            t7_rows.append({'Compound': comp_en, 'n_grupos': np.nan, 'R2_random_Table3': np.nan,
                             'R2_grouped_LOCO': np.nan, 'RMSE_grouped': np.nan, 'delta_R2': np.nan,
                             'note': 'Insufficient valid curves for LOCO (n<2)'})
            t8_rows.append({'Compound': comp_en, 'R2_RF_grouped': np.nan,
                             'R2_global_mean': np.nan, 'R2_nearest_condition': np.nan})
            continue

        n_groups = ev['grouped']['n_groups']
        r2_random = ev['result'].get('r2', np.nan)
        r2_grouped = ev['grouped']['r2']
        delta = r2_grouped - r2_random if pd.notna(r2_random) else np.nan
        note = 'No concluyente (n_grupos<4, ejecutado por completitud)' if n_groups < 4 else ''
        t7_rows.append({'Compound': comp_en, 'n_grupos': n_groups, 'R2_random_Table3': r2_random,
                         'R2_grouped_LOCO': r2_grouped, 'RMSE_grouped': ev['grouped']['rmse'],
                         'delta_R2': delta, 'note': note})

        bl = ev['baselines'] or {'r2_global_mean': np.nan, 'r2_nearest_condition': np.nan}
        t8_rows.append({'Compound': comp_en, 'R2_RF_grouped': r2_grouped,
                         'R2_global_mean': bl['r2_global_mean'],
                         'R2_nearest_condition': bl['r2_nearest_condition']})

    return pd.DataFrame(t7_rows), pd.DataFrame(t8_rows)


def build_grouped_ablation(df):
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        for n_synth in ABLATION_N_SYNTH:
            result = run_compound_pipeline(df, comp_es, comp_en, n_synth=n_synth, band=0.60, seed=RANDOM_STATE)
            n_total = len(result['df_aug']) if 'df_aug' in result else 0
            grouped = run_grouped_rf(result['df_aug'], random_state=RANDOM_STATE) if 'df_aug' in result else None
            rows.append({
                'Compound': comp_en, 'n_synth_per_curve': n_synth, 'augmented_dataset_size': n_total,
                'R2_grouped_LOCO': grouped['r2'] if grouped else np.nan,
                'RMSE_grouped': grouped['rmse'] if grouped else np.nan,
            })
    return pd.DataFrame(rows)


def plot_grouped_vs_random_ablation(grouped_df, random_df, figures_dir):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

    for ax, color, comp_en in zip(axes, colors, TARGETS_MAP.values()):
        g = grouped_df[grouped_df['Compound'] == comp_en].sort_values('n_synth_per_curve')
        r = random_df[random_df['Compound'] == comp_en].sort_values('n_synth_per_curve')
        ax.plot(g['augmented_dataset_size'], g['R2_grouped_LOCO'], marker='o', lw=2,
                color=color, label='Grouped (LOCO)')
        ax.plot(r['augmented_dataset_size'], r['test_R2'], marker='s', lw=1.5, ls='--',
                color=color, alpha=0.6, label='Random split (Task 4, old)')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_title(comp_en, fontsize=12, fontweight='bold')
        ax.set_xlabel('Augmented dataset size', fontsize=10)
        ax.set_ylabel('Test $R^2$', fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle('Fig. 9 — Augmentation Ablation: Grouped (LOCO) vs. Random Row-Wise Split',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'fig_9_ablation_grouped.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grouped-vs-random ablation figure generated at: {fig_path}")


def generate_grouped_validation(data_path: str, results_dir: str, figures_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    df = load_dataset(data_path)

    table_7, table_8 = build_tables_7_and_8(df)
    t7_path = os.path.join(results_dir, 'table_7_grouped_validation.csv')
    t8_path = os.path.join(results_dir, 'table_8_grouped_baselines.csv')
    table_7.round(4).to_csv(t7_path, index=False)
    table_8.round(4).to_csv(t8_path, index=False)

    print("=" * 80)
    print("TABLE 7: Random row-wise split (Table 3) vs. Grouped LOCO -- leakage diagnostic")
    print("=" * 80)
    print(table_7.round(4).to_string(index=False))
    print(f"[Saved to {t7_path}]\n")

    print("=" * 80)
    print("TABLE 8: Grouped RF vs. global-mean / nearest-condition baselines (all out-of-group)")
    print("=" * 80)
    print(table_8.round(4).to_string(index=False))
    print(f"[Saved to {t8_path}]\n")

    grouped_ablation = build_grouped_ablation(df)
    t9_path = os.path.join(results_dir, 'table_9_ablation_grouped.csv')
    grouped_ablation.round(4).to_csv(t9_path, index=False)
    print("=" * 80)
    print("TABLE 9: Augmentation ablation under grouped LOCO validation")
    print("=" * 80)
    print(grouped_ablation.round(4).to_string(index=False))
    print(f"[Saved to {t9_path}]\n")

    random_ablation_path = os.path.join(results_dir, 'table_ablation_r2_vs_augmentation_size.csv')
    if os.path.exists(random_ablation_path):
        random_ablation = pd.read_csv(random_ablation_path)
        plot_grouped_vs_random_ablation(grouped_ablation, random_ablation, figures_dir)
    else:
        print(f"[WARN] {random_ablation_path} not found -- skipping grouped-vs-random overlay figure.")


if __name__ == "__main__":
    generate_grouped_validation(os.path.join('..', 'data', 'dataset.csv'),
                                 os.path.join('..', 'results'), os.path.join('..', 'figures'))
