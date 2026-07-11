"""
Script 10: Augmentation Noise Sensitivity (sigma) and Physics-Informed Ablation
Addendum Tasks 6A and 6B (BRIEF_addendum_tarea6_sigma_y_fisica.md).

Works on a COPY of the augmentation pipeline (does not modify kinetic_pipeline.py /
script_5, per the brief's guardrails) so sigma (the Monte Carlo noise fraction, currently
hardcoded at 0.08 in kinetic_pipeline.augment_curves) can be swept and so a physics-free
"naive" augmentation control can be built for comparison. All performance numbers use the
grouped LeaveOneGroupOut (LOCO) validation established in script_9 (Task 5) -- never a
random row-wise split -- because Task 5 showed that is the only trustworthy way to
measure out-of-condition generalization here.

Task 6A: sweep sigma over {0.02, 0.04, 0.057, 0.08, 0.10, 0.15, 0.20} (0.057 = the
empirical CV/sqrt(3) of the mean-of-three concentration measurement; 0.08 = the value
script_5 already uses) and track, per compound: grouped R2/RMSE, the +/-60% variance
filter's rejection rate, and the mean dispersion (std) of accepted synthetic k_ext per
condition.

Task 6B: at the sigma chosen from 6A, compare the physics-informed augmentation
(perturb C(t) -> refit Ext-Deg -> extract k_ext -> +/-60% filter) against a naive,
physics-free control (perturb the curve's own real k_ext directly with the same
relative Gaussian noise, no refit, no physical filter), same n_synth, same sigma,
evaluated under the identical grouped LOCO scheme.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kinetic_pipeline import TARGETS_MAP, RANDOM_STATE, load_dataset, fit_valid_curves, augment_curves
from script_9_grouped_validation import run_grouped_rf

SIGMA_GRID = [0.02, 0.04, 0.057, 0.08, 0.10, 0.15, 0.20]
SIGMA_CURRENT = 0.08
SIGMA_EMPIRICAL_MEAN3 = 0.057   # CV/sqrt(3) of the mean-of-three concentration (Task 6C anchor)
SIGMA_EMPIRICAL_SINGLE = 0.10   # approx. single-reading CV (rounded), reference band
N_SYNTH_DEFAULT = 30
BAND_DEFAULT = 0.60


def _valid_curves_for_compound(df, comp_es):
    sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
    if sub_df.empty:
        return []
    curvas_validas, _ = fit_valid_curves(sub_df)
    return curvas_validas


def augment_naive(curvas_validas, n_synth=N_SYNTH_DEFAULT, noise_frac=SIGMA_CURRENT, seed=RANDOM_STATE):
    """Physics-free control: perturb each valid curve's own real k_ext directly with
    relative Gaussian noise (same sigma, same n_synth as the physics-informed method) --
    no Ext-Deg refit, no +/-60% physical variance filter."""
    np.random.seed(seed)
    rows = []
    for curva in curvas_validas:
        rows.append({'T_C': curva['T_C'], 'disolvente': curva['disolvente'], 'relacion': curva['relacion'],
                     'k_ext': curva['k_ext'], 'id_cinetica': curva['id_cinetica'], 'synthetic': False})
        for _ in range(n_synth):
            k_synth = curva['k_ext'] + np.random.normal(0, noise_frac * curva['k_ext'])
            k_synth = max(k_synth, 0.0)
            rows.append({'T_C': curva['T_C'], 'disolvente': curva['disolvente'], 'relacion': curva['relacion'],
                         'k_ext': k_synth, 'id_cinetica': curva['id_cinetica'], 'synthetic': True})
    return pd.DataFrame(rows)


# --- Task 6A: sigma sensitivity -------------------------------------------------------

def sigma_sensitivity_for_compound(df, comp_es, comp_en, n_synth=N_SYNTH_DEFAULT,
                                    band=BAND_DEFAULT, seed=RANDOM_STATE):
    curvas_validas = _valid_curves_for_compound(df, comp_es)
    rows = []
    for sigma in SIGMA_GRID:
        if not curvas_validas:
            rows.append({'Compound': comp_en, 'sigma': sigma, 'R2_grouped_LOCO': np.nan,
                         'RMSE_grouped': np.nan, 'rejection_rate': np.nan,
                         'mean_kext_std_by_condition': np.nan})
            continue

        df_aug, audit_df = augment_curves(curvas_validas, n_synth=n_synth, band=band,
                                           noise_frac=sigma, seed=seed)
        grouped = run_grouped_rf(df_aug, random_state=seed)

        synth = df_aug[df_aug['synthetic']]
        dispersion = (synth.groupby(['T_C', 'disolvente', 'relacion'])['k_ext'].std().mean()
                      if not synth.empty else np.nan)
        total_attempted = audit_df['n_attempted'].sum()
        total_rejected = audit_df['n_rejected'].sum()
        rejection_rate = total_rejected / total_attempted if total_attempted else np.nan

        rows.append({
            'Compound': comp_en, 'sigma': sigma,
            'R2_grouped_LOCO': grouped['r2'] if grouped else np.nan,
            'RMSE_grouped': grouped['rmse'] if grouped else np.nan,
            'rejection_rate': rejection_rate,
            'mean_kext_std_by_condition': dispersion,
        })
    return pd.DataFrame(rows)


def build_table_10(df):
    return pd.concat([sigma_sensitivity_for_compound(df, comp_es, comp_en)
                       for comp_es, comp_en in TARGETS_MAP.items()], ignore_index=True)


def plot_sigma_sensitivity(table_10, figures_dir):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    metrics = [('R2_grouped_LOCO', 'Test $R^2$ (grouped LOCO)'),
               ('rejection_rate', 'Variance-filter (±60%) rejection rate'),
               ('mean_kext_std_by_condition', 'Mean synthetic $k_{ext}$ std per condition')]

    for ax, (col, ylabel) in zip(axes, metrics):
        for color, (comp_en, grp) in zip(colors, table_10.groupby('Compound', sort=False)):
            grp = grp.sort_values('sigma')
            ax.plot(grp['sigma'], grp[col], marker='o', color=color, label=comp_en)
        ax.axvline(SIGMA_CURRENT, color='black', ls='--', lw=1.2, label=f'σ={SIGMA_CURRENT} (current)')
        ax.axvline(SIGMA_EMPIRICAL_MEAN3, color='gray', ls=':', lw=1.2,
                   label=f'σ={SIGMA_EMPIRICAL_MEAN3} (CV/√3 empirical)')
        ax.axvline(SIGMA_EMPIRICAL_SINGLE, color='gray', ls='-.', lw=1.0, alpha=0.7,
                   label=f'σ≈{SIGMA_EMPIRICAL_SINGLE} (single reading CV)')
        ax.set_xlabel('σ (augmentation noise fraction)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)

    axes[0].legend(fontsize=8, loc='best')
    plt.suptitle('Fig. 10 — Augmentation Noise (σ) Sensitivity Under Grouped LOCO Validation',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'fig_10_sigma_sensitivity.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sigma sensitivity figure generated at: {fig_path}")


# --- Task 6B: physics-informed vs. naive ablation -------------------------------------

def physics_vs_naive_for_compound(df, comp_es, comp_en, sigma, n_synth=N_SYNTH_DEFAULT,
                                   band=BAND_DEFAULT, seed=RANDOM_STATE):
    curvas_validas = _valid_curves_for_compound(df, comp_es)
    if not curvas_validas:
        return {'Compound': comp_en, 'R2_physics_informed_LOCO': np.nan, 'R2_naive_LOCO': np.nan,
                'RMSE_physics': np.nan, 'RMSE_naive': np.nan, 'delta_R2': np.nan}

    df_aug_pi, _ = augment_curves(curvas_validas, n_synth=n_synth, band=band, noise_frac=sigma, seed=seed)
    df_aug_naive = augment_naive(curvas_validas, n_synth=n_synth, noise_frac=sigma, seed=seed)

    grouped_pi = run_grouped_rf(df_aug_pi, random_state=seed)
    grouped_naive = run_grouped_rf(df_aug_naive, random_state=seed)
    r2_pi = grouped_pi['r2'] if grouped_pi else np.nan
    r2_naive = grouped_naive['r2'] if grouped_naive else np.nan

    return {
        'Compound': comp_en,
        'R2_physics_informed_LOCO': r2_pi, 'R2_naive_LOCO': r2_naive,
        'RMSE_physics': grouped_pi['rmse'] if grouped_pi else np.nan,
        'RMSE_naive': grouped_naive['rmse'] if grouped_naive else np.nan,
        'delta_R2': (r2_pi - r2_naive) if (pd.notna(r2_pi) and pd.notna(r2_naive)) else np.nan,
    }


def build_table_11(df, sigma):
    return pd.DataFrame([physics_vs_naive_for_compound(df, comp_es, comp_en, sigma)
                          for comp_es, comp_en in TARGETS_MAP.items()])


def generate_sigma_and_physics(data_path: str, results_dir: str, figures_dir: str,
                                sigma_for_6b: float = SIGMA_CURRENT) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    df = load_dataset(data_path)

    table_10 = build_table_10(df)
    t10_path = os.path.join(results_dir, 'table_10_sigma_sensitivity.csv')
    table_10.round(4).to_csv(t10_path, index=False)
    print("=" * 80)
    print("TABLE 10: Augmentation noise (sigma) sensitivity under grouped LOCO validation")
    print("=" * 80)
    print(table_10.round(4).to_string(index=False))
    print(f"[Saved to {t10_path}]\n")
    plot_sigma_sensitivity(table_10, figures_dir)

    print(f"\n[Using sigma={sigma_for_6b} for Task 6B physics-informed vs. naive ablation]\n")
    table_11 = build_table_11(df, sigma_for_6b)
    t11_path = os.path.join(results_dir, 'table_11_physics_ablation.csv')
    table_11.round(4).to_csv(t11_path, index=False)
    print("=" * 80)
    print(f"TABLE 11: Physics-informed vs. naive augmentation (sigma={sigma_for_6b}), grouped LOCO")
    print("=" * 80)
    print(table_11.round(4).to_string(index=False))
    print(f"[Saved to {t11_path}]\n")


if __name__ == "__main__":
    generate_sigma_and_physics(os.path.join('..', 'data', 'dataset.csv'),
                                os.path.join('..', 'results'), os.path.join('..', 'figures'),
                                sigma_for_6b=SIGMA_CURRENT)
