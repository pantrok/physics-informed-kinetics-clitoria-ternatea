"""
Script 7: Physics-Informed Augmentation Validation (Fig. 8 & Table 6)
Task 4 of the ENBIS analytical-reinforcement brief.

(a) Augmentation ablation curve: retrains the k_ext Random Forest varying the
    number of Monte Carlo synthetics accepted per valid curve (0, 10, 30, 60, 90)
    and plots test R^2 vs. augmented dataset size, per compound.
(b) Variance-filter audit: counts how many synthetic k_ext draws are accepted vs.
    rejected by the +/-60% phenomenological variance filter, per compound and per
    operating condition (Table 6).

n_synth=30 is the value actually used for Table 3 / Fig. 6 / Fig. 7; the ablation
here re-runs the SAME seeded pipeline (kinetic_pipeline.run_compound_pipeline) at
other n_synth values purely to characterize the augmentation choice -- it does not
change the reported Table 3 metrics.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kinetic_pipeline import TARGETS_MAP, RANDOM_STATE, load_dataset, run_compound_pipeline

ABLATION_N_SYNTH = [0, 10, 30, 60, 90]


def run_ablation(df):
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        for n_synth in ABLATION_N_SYNTH:
            result = run_compound_pipeline(df, comp_es, comp_en, n_synth=n_synth, band=0.60, seed=RANDOM_STATE)
            n_total = len(result['df_aug']) if 'df_aug' in result else 0
            if result.get('insufficient') or 'r2' not in result:
                rows.append({'Compound': comp_en, 'n_synth_per_curve': n_synth,
                             'augmented_dataset_size': n_total, 'test_R2': np.nan, 'test_RMSE': np.nan})
            else:
                rows.append({'Compound': comp_en, 'n_synth_per_curve': n_synth,
                             'augmented_dataset_size': n_total, 'test_R2': result['r2'], 'test_RMSE': result['rmse']})
    return pd.DataFrame(rows)


def run_variance_filter_audit(df):
    frames = []
    for comp_es, comp_en in TARGETS_MAP.items():
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)
        if 'audit' not in result or result['audit'].empty:
            continue
        audit = result['audit'].copy()
        audit.insert(0, 'Compound', comp_en)
        frames.append(audit)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    audit_all = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for comp_en, grp in audit_all.groupby('Compound'):
        total_attempted = grp['n_attempted'].sum()
        total_accepted = grp['n_accepted'].sum()
        total_rejected = grp['n_rejected'].sum()
        summary_rows.append({
            'Compound': comp_en, 'n_conditions': len(grp),
            'total_attempted': total_attempted, 'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'acceptance_rate': total_accepted / total_attempted if total_attempted else np.nan,
            'rejection_rate': total_rejected / total_attempted if total_attempted else np.nan,
        })
    return audit_all, pd.DataFrame(summary_rows)


def plot_ablation(ablation_df, figures_dir):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.3)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    for color, (comp_en, grp) in zip(colors, ablation_df.groupby('Compound', sort=False)):
        grp = grp.sort_values('n_synth_per_curve')
        ax.plot(grp['augmented_dataset_size'], grp['test_R2'], marker='o', label=comp_en, color=color, lw=2)
    ax.set_xlabel('Augmented dataset size (real + accepted synthetic k_ext)', fontsize=13)
    ax.set_ylabel('Test $R^2$ (k_ext Random Forest)', fontsize=13)
    ax.set_title('Fig. 8 — Augmentation Ablation Curve\n(model prediction quality vs. Monte Carlo augmentation size)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'fig_8_augmentation_ablation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Augmentation ablation curve generated at: {fig_path}")


def generate_augmentation_validation(data_path: str, results_dir: str, figures_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    df = load_dataset(data_path)

    ablation_df = run_ablation(df)
    ablation_path = os.path.join(results_dir, 'table_ablation_r2_vs_augmentation_size.csv')
    ablation_df.round(4).to_csv(ablation_path, index=False)
    print("=" * 80)
    print("ABLATION: test R2 vs. augmented dataset size (n_synth per curve = 0,10,30,60,90)")
    print("=" * 80)
    print(ablation_df.round(4).to_string(index=False))
    print(f"[Saved to {ablation_path}]\n")

    plot_ablation(ablation_df, figures_dir)

    audit_all, summary_df = run_variance_filter_audit(df)
    audit_path = os.path.join(results_dir, 'table_6_variance_filter_audit.csv')
    summary_path = os.path.join(results_dir, 'table_6b_variance_filter_audit_summary.csv')
    audit_all.round(4).to_csv(audit_path, index=False)
    summary_df.round(4).to_csv(summary_path, index=False)

    print("=" * 80)
    print("TABLE 6: Variance-filter (+/-60%) audit -- per operating condition")
    print("=" * 80)
    print(audit_all.round(4).to_string(index=False) if not audit_all.empty else "[No accepted synthetic data]")
    print(f"[Saved to {audit_path}]\n")

    print("=" * 80)
    print("TABLE 6 (summary): acceptance / rejection rates per compound")
    print("=" * 80)
    print(summary_df.round(4).to_string(index=False) if not summary_df.empty else "[No accepted synthetic data]")
    print(f"[Saved to {summary_path}]\n")


if __name__ == "__main__":
    generate_augmentation_validation(os.path.join('..', 'data', 'dataset.csv'),
                                      os.path.join('..', 'results'), os.path.join('..', 'figures'))
