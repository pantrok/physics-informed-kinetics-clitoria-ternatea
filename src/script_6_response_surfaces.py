"""
Script 6: Physics-Informed Response Surfaces / Design-Space Visualization (Fig. 7)
Task 2 of the ENBIS analytical-reinforcement brief.

Builds a dense interpolation mesh strictly within the measured design space
(T_C in [60, 75] deg C, the 3 disolvente levels, the 3 relacion levels of the
Box-Behnken design) and maps it through each compound's ALREADY-TRAINED k_ext
Random Forest (see kinetic_pipeline.run_compound_pipeline, same pipeline used
by script_5). Real experimental (T, k_ext) points from the valid Ext-Deg fits
are overlaid for visual contrast.

GUARDRAIL: everything plotted here is a MODEL PREDICTION over an interpolation
mesh -- never extrapolated beyond the measured T range -- and is always
labeled as such, never as new experimental data.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kinetic_pipeline import TARGETS_MAP, RANDOM_STATE, T_MIN, T_MAX, load_dataset, run_compound_pipeline

T_STEP = 0.5


def build_mesh(disolventes, relaciones):
    t_grid = np.round(np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP), 1)
    rows = [{'T_C': t, 'disolvente': d, 'relacion': r} for d in disolventes for r in relaciones for t in t_grid]
    return pd.DataFrame(rows), t_grid


def generate_response_surfaces(data_path: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    df = load_dataset(data_path)
    disolventes = sorted(df['disolvente'].unique())
    relaciones = sorted(df['relacion'].unique())
    combos = [f"{d} | {r}" for d in disolventes for r in relaciones]

    mesh_df, t_grid = build_mesh(disolventes, relaciones)

    sns.set_theme(style='white', context='paper', font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    axes = axes.ravel()

    for i, (comp_es, comp_en) in enumerate(TARGETS_MAP.items()):
        ax = axes[i]
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)

        if result.get('insufficient') or 'grid' not in result:
            ax.set_title(comp_en, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, "Insufficient valid physical data\nfor ML mapping", ha='center', va='center',
                    fontsize=13, color='red', weight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            continue

        mesh_pred = result['grid'].best_estimator_.predict(mesh_df[['T_C', 'disolvente', 'relacion']])
        surface = np.zeros((len(combos), len(t_grid)))
        for row_idx, (d, r) in enumerate([(d, r) for d in disolventes for r in relaciones]):
            mask = (mesh_df['disolvente'] == d) & (mesh_df['relacion'] == r)
            surface[row_idx, :] = mesh_pred[mask.values]

        im = ax.imshow(surface, aspect='auto', origin='lower', cmap='viridis',
                        extent=[t_grid.min(), t_grid.max(), -0.5, len(combos) - 0.5])
        ax.set_yticks(range(len(combos)))
        ax.set_yticklabels(combos, fontsize=9)
        ax.set_xlabel('Temperature, T (°C)', fontsize=12)
        ax.set_title(f"{comp_en}\n(model prediction — interpolation within measured design space)",
                     fontsize=13, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('$k_{ext}$ (min$^{-1}$) — model prediction', fontsize=10)

        for curva in result['curvas_validas']:
            combo_label = f"{curva['disolvente']} | {curva['relacion']}"
            if combo_label not in combos:
                continue
            row_idx = combos.index(combo_label)
            ax.scatter([curva['T_C']], [row_idx], marker='o', s=90, facecolor='none',
                       edgecolor='white', linewidth=1.8, zorder=5)
            ax.scatter([curva['T_C']], [row_idx], marker='x', s=50, color='black', linewidth=1.6, zorder=6)

    plt.suptitle('Fig. 7 — Physics-Informed $k_{ext}$ Response Surfaces (Design Space)\n'
                 'Model prediction, interpolation within the measured design space [T: 60-75 °C]. '
                 'White-ringed markers = real experimental points.', y=1.02, fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'fig_7_response_surfaces.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Response-surface design-space map generated at: {fig_path}")


if __name__ == "__main__":
    generate_response_surfaces(os.path.join('..', 'data', 'dataset.csv'), os.path.join('..', 'figures'))
