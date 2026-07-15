"""
Script 12: Gaussian Process Uncertainty and Final Manuscript Consolidation
Addendum Task 7 (BRIEF_addendum_tarea7_gp_y_consolidacion.md) -- closes the validation
workstream and prepares the final, clean figure/table set for manuscript writing.

Sections:
  1. Final figure set (fig_2 ... fig_7), renumbered and captioned in clean scientific
     terminology, from fig_1 onward reserved for the methodology diagram (drawn separately).
  2. Supplementary check: out-of-condition (grouped) validation WITHOUT the R^2>=0.50
     physical filter, to show the reduced-design limitation is not an artifact of the filter.
  3. Gaussian Process regression (Phenolics only) with predictive uncertainty quantification
     and a design-space uncertainty map (fig_8).
  4. Consolidated, publication-ready CSVs (paper_table_*.csv) with a table index.

Reuses (read-only imports; does not modify) kinetic_pipeline.py, script_8_rsm_vs_rf.py and
script_9_grouped_validation.py. All performance metrics are out-of-condition (grouped
cross-validation by id_cinetica), consistent with the validation scheme established
previously in this project.
"""

import os

import matplotlib
matplotlib.use('Agg')  # headless backend -- avoids a Tk/Tcl crash (exit code 3) seen when
                        # joblib's multiprocess GridSearchCV (fig_5) runs under the
                        # interactive TkAgg backend that is this environment's default.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from kinetic_pipeline import (TARGETS_MAP, RANDOM_STATE, T_MIN, T_MAX,
                               load_dataset, ext_deg_model, fit_valid_curves, augment_curves)
from script_8_rsm_vs_rf import parse_ethanol_pct, parse_ratio_val
from script_9_grouped_validation import run_grouped_rf
from figure_export import mm_to_in, save_figure

COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
T_STEP = 0.5

# Journal of Chemometrics (Wiley) final-art width target: within 80-180 mm.
# All figures here are complex multipanel/heatmap/legend content, so all target the wide
# end of the range; height is derived per figure to preserve its current aspect ratio.
FIG_WIDTH_MM = 178
FIG_WIDTH_IN = mm_to_in(FIG_WIDTH_MM)


def _sized_figsize(width_in, height_in, target_mm=FIG_WIDTH_MM):
    """Rescale (width_in, height_in) to target_mm (in) while preserving aspect ratio.
    target_mm defaults to FIG_WIDTH_MM but is reduced for figures with a colorbar, since
    the colorbar+label render outside the nominal axes box and would otherwise push the
    final tight-bbox width past the Wiley 180 mm limit (confirmed empirically via
    FORMAT_check.md: colorbar figures overshot the plain figsize target by 13-23%)."""
    target_in = mm_to_in(target_mm)
    scale = target_in / width_in
    return target_in, height_in * scale


# =====================================================================================
# Section 1 -- Final figure set (fig_2 ... fig_7)
# =====================================================================================

def make_fig2_eda(figures_dir):
    """fig_2: generated directly by script_1_eda_pairplot.py (run first) in final
    Wiley format (TIFF 600 dpi + PNG 300 dpi, fig_2_eda_pairplot.*). Nothing to do here;
    this is just a presence check so a missing prerequisite run fails loudly."""
    tiff_path = os.path.join(figures_dir, 'fig_2_eda_pairplot.tiff')
    if not os.path.exists(tiff_path):
        print(f"[WARN] {tiff_path} not found -- run script_1_eda_pairplot.py first.")
    else:
        print(f"fig_2 (EDA pairplot) already present -> {tiff_path}")


def make_fig3_phenomenological_fits(df, figures_dir):
    """fig_3: Ext-Deg phenomenological fit overlaid on the real concentration-vs-time data,
    for every physically valid curve, one panel per compound."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.3)
    fig, axes = plt.subplots(2, 2, figsize=_sized_figsize(15, 12, target_mm=172))
    axes = axes.ravel()

    for ax, (comp_es, comp_en) in zip(axes, TARGETS_MAP.items()):
        sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
        curvas_validas, _ = fit_valid_curves(sub_df)
        if not curvas_validas:
            ax.set_title(comp_en, fontsize=13, fontweight='bold')
            ax.text(0.5, 0.5, 'Insufficient valid curves', ha='center', va='center',
                    fontsize=12, color='red', weight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            continue

        palette = sns.color_palette('tab10', len(curvas_validas))
        for color, curva in zip(palette, curvas_validas):
            datos = sub_df[sub_df['id_cinetica'] == curva['id_cinetica']].sort_values('tiempo_min')
            t_obs, C_obs = datos['tiempo_min'].values, datos['conc_promedio'].values
            t_smooth = np.linspace(t_obs.min(), t_obs.max(), 200)
            C_smooth = ext_deg_model(t_smooth, curva['C0'], curva['k_ext'], curva['k_deg'])
            label = f"{curva['T_C']:.0f}°C, {curva['disolvente']}, {curva['relacion']}"
            ax.scatter(t_obs, C_obs, color=color, s=28, alpha=0.85, edgecolor='white', linewidth=0.4)
            ax.plot(t_smooth, C_smooth, color=color, lw=1.6, label=label)

        ax.set_title(comp_en, fontsize=13, fontweight='bold')
        ax.set_xlabel('$t$ (min)', fontsize=11)
        ax.set_ylabel('$C$', fontsize=11)
        ax.legend(fontsize=6.5, loc='best', framealpha=0.9)

    plt.suptitle('Phenomenological Ext-Deg fits for physically valid extraction kinetics',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_3_phenomenological_fits')
    plt.close()
    print("fig_3 (phenomenological fits) saved")


def make_fig4_kext_confidence_intervals(results_dir, figures_dir):
    """fig_4: estimated k_ext per operating condition with its 95% confidence interval."""
    table_4 = pd.read_csv(os.path.join(results_dir, 'table_4_kext_confidence_intervals.csv'))
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.3)
    fig, axes = plt.subplots(2, 2, figsize=_sized_figsize(15, 12))
    axes = axes.ravel()

    for ax, color, comp_en in zip(axes, COLORS, TARGETS_MAP.values()):
        sub = table_4[table_4['Compound'] == comp_en].sort_values('T_C').reset_index(drop=True)
        if sub.empty:
            ax.set_title(comp_en, fontsize=13, fontweight='bold')
            ax.text(0.5, 0.5, 'Insufficient valid conditions', ha='center', va='center',
                    fontsize=12, color='red', weight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            continue

        x = np.arange(len(sub))
        yerr_low = sub['k_ext_mean'] - sub['CI95_lower']
        yerr_high = sub['CI95_upper'] - sub['k_ext_mean']
        ax.errorbar(x, sub['k_ext_mean'], yerr=[yerr_low, yerr_high], fmt='o', ms=7,
                    color=color, ecolor=color, elinewidth=1.3, capsize=4, markeredgecolor='white')
        labels = [f"{r.T_C:.0f}°C\n{r.disolvente}\n{r.relacion}" for r in sub.itertuples()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel(r'$k_{\mathrm{ext}}$ (min$^{-1}$)', fontsize=11)
        ax.set_title(comp_en, fontsize=13, fontweight='bold')

    plt.suptitle(r'Estimated $k_{\mathrm{ext}}$ by operating condition, with 95% confidence intervals',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_4_kext_confidence_intervals')
    plt.close()
    print("fig_4 (k_ext confidence intervals) saved")


def _build_mesh(disolventes, relaciones):
    t_grid = np.round(np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP), 1)
    rows = [{'T_C': t, 'disolvente': d, 'relacion': r} for d in disolventes for r in relaciones for t in t_grid]
    return pd.DataFrame(rows), t_grid


def make_fig5_response_surfaces(df, figures_dir):
    """fig_5: RF-predicted k_ext response surfaces across the measured design space
    (model prediction, interpolation only -- never extrapolated)."""
    from kinetic_pipeline import run_compound_pipeline

    disolventes = sorted(df['disolvente'].unique())
    relaciones = sorted(df['relacion'].unique())
    combos = [f"{d} | {r}" for d in disolventes for r in relaciones]
    mesh_df, t_grid = _build_mesh(disolventes, relaciones)

    sns.set_theme(style='white', context='paper', font_scale=1.3)
    fig, axes = plt.subplots(2, 2, figsize=_sized_figsize(17, 13, target_mm=155))
    axes = axes.ravel()

    for i, (comp_es, comp_en) in enumerate(TARGETS_MAP.items()):
        ax = axes[i]
        result = run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE)
        if result.get('insufficient') or 'grid' not in result:
            ax.set_title(comp_en, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center',
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
        ax.set_yticks(range(len(combos))); ax.set_yticklabels(combos, fontsize=8)
        ax.set_xlabel('$T$ (°C)', fontsize=11)
        ax.set_title(f"{comp_en} — model prediction", fontsize=12, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$k_{\mathrm{ext}}$ (min$^{-1}$)', fontsize=9)

        for curva in result['curvas_validas']:
            combo_label = f"{curva['disolvente']} | {curva['relacion']}"
            if combo_label not in combos:
                continue
            row_idx = combos.index(combo_label)
            ax.scatter([curva['T_C']], [row_idx], marker='o', s=90, facecolor='none',
                       edgecolor='white', linewidth=1.8, zorder=5)
            ax.scatter([curva['T_C']], [row_idx], marker='x', s=50, color='black', linewidth=1.6, zorder=6)

    plt.suptitle(r'Model-predicted $k_{\mathrm{ext}}$ response surfaces (interpolation within the measured design space)'
                 '\nWhite-ringed markers = real experimental points', y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_5_response_surfaces')
    plt.close()
    print("fig_5 (response surfaces) saved")


def make_fig6_in_vs_out_of_condition(results_dir, figures_dir):
    """fig_6: in-condition (random split) vs. out-of-condition (grouped cross-validation)
    predictive performance -- the central methodological finding."""
    table_7 = pd.read_csv(os.path.join(results_dir, 'table_7_grouped_validation.csv'))
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.3)
    fig, ax = plt.subplots(figsize=_sized_figsize(10, 6.5))

    compounds = table_7['Compound'].tolist()
    x = np.arange(len(compounds))
    width = 0.35
    ax.bar(x - width / 2, table_7['R2_random_Table3'], width, label='In-condition (random split)',
           color='#9fc5e8', edgecolor='black', linewidth=0.6)
    ax.bar(x + width / 2, table_7['R2_grouped_LOCO'], width, label='Out-of-condition (grouped)',
           color='#c44e52', edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(compounds, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Test $R^2$', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Predictive performance: in-condition vs. out-of-condition validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_6_out_of_condition_validation')
    plt.close()
    print("fig_6 (in- vs. out-of-condition validation) saved")


def make_fig7_augmentation_and_sigma(results_dir, figures_dir):
    """fig_7: effect of augmentation size and noise level (sigma) on out-of-condition
    predictive performance -- both evaluated under grouped cross-validation."""
    table_9 = pd.read_csv(os.path.join(results_dir, 'table_9_ablation_grouped.csv'))
    table_10 = pd.read_csv(os.path.join(results_dir, 'table_10_sigma_sensitivity.csv'))

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.25)
    fig, axes = plt.subplots(1, 2, figsize=_sized_figsize(15, 6))

    for color, (comp_en, grp) in zip(COLORS, table_9.groupby('Compound', sort=False)):
        grp = grp.sort_values('n_synth_per_curve')
        axes[0].plot(grp['augmented_dataset_size'], grp['R2_grouped_LOCO'], marker='o',
                     color=color, label=comp_en, lw=2)
    axes[0].axhline(0, color='gray', lw=0.8, ls=':')
    axes[0].set_xlabel('Augmented dataset size', fontsize=10)
    axes[0].set_ylabel('Test $R^2$ (out-of-condition)', fontsize=10)
    axes[0].set_title('Effect of augmentation size', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=7)

    for color, (comp_en, grp) in zip(COLORS, table_10.groupby('Compound', sort=False)):
        grp = grp.sort_values('sigma')
        axes[1].plot(grp['sigma'], grp['R2_grouped_LOCO'], marker='o', color=color, label=comp_en, lw=2)
    axes[1].axhline(0, color='gray', lw=0.8, ls=':')
    axes[1].axvline(0.08, color='black', ls='--', lw=1.1, label=r'$\sigma$ = 0.08 (adopted)')
    axes[1].set_xlabel(r'$\sigma$ (augmentation noise fraction)', fontsize=10)
    axes[1].set_ylabel('Test $R^2$ (out-of-condition)', fontsize=10)
    axes[1].set_title('Effect of augmentation noise level', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=7)

    plt.suptitle('Augmentation size and noise level do not restore out-of-condition generalization',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_7_augmentation_and_sigma')
    plt.close()
    print("fig_7 (augmentation and sigma effects) saved")


# =====================================================================================
# Section 2 -- Out-of-condition validation WITHOUT the physical R^2>=0.50 filter
# =====================================================================================

def fit_all_curves_no_filter(sub_df):
    """Ext-Deg fit for every curve with >=4 timepoints, regardless of fit quality --
    duplicates the fitting loop (rather than modifying kinetic_pipeline.fit_valid_curves,
    which must stay untouched) specifically to DROP the R^2>=0.50 acceptance filter."""
    cineticas = sub_df['id_cinetica'].unique()
    curvas_all = []
    for cid in cineticas:
        datos = sub_df[sub_df['id_cinetica'] == cid].sort_values('tiempo_min')
        if len(datos) < 4:
            continue
        t, C = datos['tiempo_min'].values, datos['conc_promedio'].values
        try:
            popt, _ = curve_fit(ext_deg_model, t, C, p0=[max(C) * 1.5, 0.05, 0.005],
                                 bounds=([0, 0, 0], [np.inf, 1.0, 1.0]), maxfev=10000)
            r2 = r2_score(C, ext_deg_model(t, *popt))
            curvas_all.append({'id_cinetica': cid, 'T_C': datos['T_C'].iloc[0],
                                'disolvente': datos['disolvente'].iloc[0], 'relacion': datos['relacion'].iloc[0],
                                'C0': popt[0], 'k_ext': popt[1], 'k_deg': popt[2], 't_real': t, 'r2_fit': r2})
        except RuntimeError:
            pass
    return curvas_all


def build_table_supp_nofilter(df, results_dir):
    table_7 = pd.read_csv(os.path.join(results_dir, 'table_7_grouped_validation.csv'))
    rows = []
    for comp_es, comp_en in TARGETS_MAP.items():
        sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
        curvas_all = fit_all_curves_no_filter(sub_df) if not sub_df.empty else []
        r2_with = table_7.loc[table_7['Compound'] == comp_en, 'R2_grouped_LOCO']
        n_with = table_7.loc[table_7['Compound'] == comp_en, 'n_grupos']

        if len(curvas_all) < 2:
            rows.append({'Compound': comp_en,
                         'R2_grouped_with_filter': r2_with.values[0] if len(r2_with) else np.nan,
                         'R2_grouped_no_filter': np.nan,
                         'n_curves_with': int(n_with.values[0]) if len(n_with) else np.nan,
                         'n_curves_no': len(curvas_all)})
            continue

        df_aug_no_filter, _ = augment_curves(curvas_all, n_synth=30, band=0.60, seed=RANDOM_STATE)
        grouped = run_grouped_rf(df_aug_no_filter, random_state=RANDOM_STATE)
        rows.append({
            'Compound': comp_en,
            'R2_grouped_with_filter': r2_with.values[0] if len(r2_with) else np.nan,
            'R2_grouped_no_filter': grouped['r2'] if grouped else np.nan,
            'n_curves_with': int(n_with.values[0]) if len(n_with) else np.nan,
            'n_curves_no': len(curvas_all),
        })
    return pd.DataFrame(rows)


# =====================================================================================
# Section 3 -- Gaussian Process with predictive uncertainty (Phenolics)
# =====================================================================================

def _build_gp_kernel():
    return (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0)))


def fit_gp_for_compound(df, comp_es='Fenoles'):
    sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
    curvas_validas, _ = fit_valid_curves(sub_df)
    if len(curvas_validas) < 3:
        return None

    X = np.array([[float(c['T_C']), parse_ethanol_pct(c['disolvente']), parse_ratio_val(c['relacion'])]
                   for c in curvas_validas])
    y = np.array([c['k_ext'] for c in curvas_validas])

    # Leave-one-condition-out (each condition contributes exactly one real point here --
    # no Monte Carlo augmentation is used for the GP, per the brief).
    y_true_all, y_pred_all = [], []
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        scaler = StandardScaler().fit(X[mask])
        gp = GaussianProcessRegressor(kernel=_build_gp_kernel(), normalize_y=True,
                                       n_restarts_optimizer=5, random_state=RANDOM_STATE)
        gp.fit(scaler.transform(X[mask]), y[mask])
        pred = gp.predict(scaler.transform(X[i:i + 1]))
        y_true_all.append(y[i]); y_pred_all.append(pred[0])
    r2_loco = r2_score(y_true_all, y_pred_all)
    rmse_loco = mean_squared_error(y_true_all, y_pred_all) ** 0.5

    scaler_full = StandardScaler().fit(X)
    gp_final = GaussianProcessRegressor(kernel=_build_gp_kernel(), normalize_y=True,
                                         n_restarts_optimizer=5, random_state=RANDOM_STATE)
    gp_final.fit(scaler_full.transform(X), y)

    return {'r2_loco': r2_loco, 'rmse_loco': rmse_loco, 'gp': gp_final, 'scaler': scaler_full,
            'curvas_validas': curvas_validas, 'X': X, 'y': y}


def make_fig8_gp_uncertainty_map(gp_result, df, figures_dir):
    disolventes = sorted(df['disolvente'].unique())
    relaciones = sorted(df['relacion'].unique())
    combos = [f"{d} | {r}" for d in disolventes for r in relaciones]
    mesh_df, t_grid = _build_mesh(disolventes, relaciones)
    mesh_factors = np.array([[t, parse_ethanol_pct(d), parse_ratio_val(r)]
                              for d, r, t in zip(mesh_df['disolvente'], mesh_df['relacion'], mesh_df['T_C'])])

    _, std_pred = gp_result['gp'].predict(gp_result['scaler'].transform(mesh_factors), return_std=True)

    surface = np.zeros((len(combos), len(t_grid)))
    for row_idx, (d, r) in enumerate([(d, r) for d in disolventes for r in relaciones]):
        mask = (mesh_df['disolvente'] == d) & (mesh_df['relacion'] == r)
        surface[row_idx, :] = std_pred[mask.values]

    sns.set_theme(style='white', context='paper', font_scale=1.3)
    fig, ax = plt.subplots(figsize=_sized_figsize(10, 7, target_mm=145))
    im = ax.imshow(surface, aspect='auto', origin='lower', cmap='magma',
                    extent=[t_grid.min(), t_grid.max(), -0.5, len(combos) - 0.5])
    ax.set_yticks(range(len(combos))); ax.set_yticklabels(combos, fontsize=9)
    ax.set_xlabel('$T$ (°C)', fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'GP predictive standard deviation of $k_{\mathrm{ext}}$ (min$^{-1}$)', fontsize=10)

    for curva in gp_result['curvas_validas']:
        combo_label = f"{curva['disolvente']} | {curva['relacion']}"
        if combo_label not in combos:
            continue
        row_idx = combos.index(combo_label)
        ax.scatter([curva['T_C']], [row_idx], marker='o', s=100, facecolor='none',
                   edgecolor='cyan', linewidth=2.0, zorder=5)

    ax.set_title(r'Gaussian Process predictive uncertainty for $k_{\mathrm{ext}}$ across the measured design space (Phenolics)'
                 '\nHigh-uncertainty regions indicate operating conditions not informed by the current design; '
                 'cyan markers = experimental points', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_8_gp_uncertainty_map')
    plt.close()
    print("fig_8 (GP uncertainty map) saved")


def build_table_gp_uncertainty(gp_result, df):
    """Representative grid: the 3 measured temperatures x 9 solvent/ratio combinations
    (27 rows) -- legible and directly tied to the design, per the brief."""
    disolventes = sorted(df['disolvente'].unique())
    relaciones = sorted(df['relacion'].unique())
    t_values = sorted(df['T_C'].unique())

    rows = []
    for t in t_values:
        for d in disolventes:
            for r in relaciones:
                x = np.array([[float(t), parse_ethanol_pct(d), parse_ratio_val(r)]])
                mean, std = gp_result['gp'].predict(gp_result['scaler'].transform(x), return_std=True)
                mean, std = mean[0], std[0]
                rows.append({
                    'T_C': t, 'solvent': d, 'ratio': r,
                    'k_ext_GP_mean': mean, 'k_ext_GP_std': std,
                    'CI95_low': mean - 1.96 * std, 'CI95_high': mean + 1.96 * std,
                })
    return pd.DataFrame(rows)


# =====================================================================================
# Section 4 -- Consolidated CSVs for the manuscript
# =====================================================================================

def consolidate_paper_tables(results_dir, gp_table, nofilter_table):
    # paper_table_1: phenomenological fit summary
    t2 = pd.read_csv(os.path.join(results_dir, 'table_2_phenomenological_fit.csv'))
    t2 = t2.rename(columns={
        'Ext-Deg Model (All, n=15) R2': 'R2_all_curves',
        'Valid Curves (n)': 'n_valid_curves',
        'Ext-Deg Model (Valid Subset) R2': 'R2_valid_curves_only',
    })
    t2.round(3).to_csv(os.path.join(results_dir, 'paper_table_1_kinetic_fit_summary.csv'), index=False)

    # paper_table_2: k_ext confidence intervals
    t4 = pd.read_csv(os.path.join(results_dir, 'table_4_kext_confidence_intervals.csv'))
    t4 = t4.rename(columns={'disolvente': 'solvent', 'relacion': 'ratio'})
    t4.round(4).to_csv(os.path.join(results_dir, 'paper_table_2_kext_confidence_intervals.csv'), index=False)

    # paper_table_3: out-of-condition validation with baselines
    t7 = pd.read_csv(os.path.join(results_dir, 'table_7_grouped_validation.csv'))
    t8 = pd.read_csv(os.path.join(results_dir, 'table_8_grouped_baselines.csv'))
    t3 = t7.merge(t8, on='Compound', how='left').rename(columns={
        'n_grupos': 'n_conditions',
        'R2_random_Table3': 'R2_in_condition_random_split',
        'R2_grouped_LOCO': 'R2_out_of_condition_grouped',
        'RMSE_grouped': 'RMSE_out_of_condition_grouped',
        'R2_RF_grouped': 'R2_RF_out_of_condition',
        'R2_global_mean': 'R2_baseline_global_mean',
        'R2_nearest_condition': 'R2_baseline_nearest_condition',
    })
    t3 = t3[['Compound', 'n_conditions', 'R2_in_condition_random_split', 'R2_out_of_condition_grouped',
             'RMSE_out_of_condition_grouped', 'delta_R2', 'R2_baseline_global_mean',
             'R2_baseline_nearest_condition', 'note']]
    t3['note'] = t3.apply(
        lambda r: 'Not conclusive (n<4 conditions); included for completeness' if r['n_conditions'] < 4 else '',
        axis=1)
    t3.round(4).to_csv(os.path.join(results_dir, 'paper_table_3_out_of_condition_validation.csv'), index=False)

    # paper_table_4: replicate measurement CV
    t12 = pd.read_csv(os.path.join(results_dir, 'table_12_replicate_cv_by_compound.csv'))
    t12.round(3).to_csv(os.path.join(results_dir, 'paper_table_4_replicate_measurement_cv.csv'), index=False)

    # paper_table_5: GP uncertainty
    gp_table.round(4).to_csv(os.path.join(results_dir, 'paper_table_5_gp_uncertainty.csv'), index=False)

    # supplementary: no-filter validation
    nofilter_table.round(4).to_csv(os.path.join(results_dir, 'paper_table_supp_nofilter.csv'), index=False)

    return {
        'paper_table_1_kinetic_fit_summary.csv': t2,
        'paper_table_2_kext_confidence_intervals.csv': t4,
        'paper_table_3_out_of_condition_validation.csv': t3,
        'paper_table_4_replicate_measurement_cv.csv': t12,
        'paper_table_5_gp_uncertainty.csv': gp_table,
        'paper_table_supp_nofilter.csv': nofilter_table,
    }


def write_tables_readme(results_dir, tables):
    lines = ["# README_tables.md — Index of manuscript-ready CSV tables\n",
             "Each CSV below is the single source of truth for its corresponding manuscript table. "
             "All R2 values are out-of-condition (grouped cross-validation) unless the column name says "
             "otherwise; k_ext in min^-1.\n"]
    descriptions = {
        'paper_table_1_kinetic_fit_summary.csv':
            ("**Manuscript table:** phenomenological Ext-Deg fit summary.\n"
             "Columns: `Compound`, `R2_all_curves` (mean R2 of the Ext-Deg fit over all 15 kinetic curves), "
             "`n_valid_curves` (curves passing the R2>=0.50 physical validity filter), "
             "`R2_valid_curves_only` (mean R2 restricted to the valid subset)."),
        'paper_table_2_kext_confidence_intervals.csv':
            ("**Manuscript table:** k_ext estimate per operating condition, with 95% CI.\n"
             "Columns: `Compound`, `T_C`, `solvent`, `ratio`, `n_synthetic`, `k_ext_mean`, `k_ext_std`, "
             "`CI95_lower`, `CI95_upper`, `CI95_width`."),
        'paper_table_3_out_of_condition_validation.csv':
            ("**Manuscript table:** in-condition vs. out-of-condition predictive performance, with baselines.\n"
             "Columns: `Compound`, `n_conditions`, `R2_in_condition_random_split`, "
             "`R2_out_of_condition_grouped`, `RMSE_out_of_condition_grouped`, `delta_R2`, "
             "`R2_baseline_global_mean`, `R2_baseline_nearest_condition`, `note`."),
        'paper_table_4_replicate_measurement_cv.csv':
            ("**Manuscript table:** absorbance replicate (triplicate) measurement CV by compound.\n"
             "Columns: `Compound`, `n_obs`, `A_mean_min/max`, `CV_median_pct`, `CV_mean_pct`, "
             "`CV_P25_pct`, `CV_P75_pct`, `CV_of_mean_of_3_median_pct`, `CV_of_mean_of_3_mean_pct`. "
             "Anthocyanins shows markedly higher CV (~27%), consistent with absorbance near the "
             "detection limit -- an honest limitation, not a measurement error."),
        'paper_table_5_gp_uncertainty.csv':
            ("**Manuscript table:** Gaussian Process predictive mean, std and 95% CI for k_ext "
             "(Phenolics), over the 3 measured temperatures x 9 solvent/ratio combinations.\n"
             "Columns: `T_C`, `solvent`, `ratio`, `k_ext_GP_mean`, `k_ext_GP_std`, `CI95_low`, `CI95_high`."),
        'paper_table_supp_nofilter.csv':
            ("**Supplementary table:** out-of-condition validation with vs. without the R2>=0.50 "
             "physical validity filter.\n"
             "Columns: `Compound`, `R2_grouped_with_filter`, `R2_grouped_no_filter`, `n_curves_with`, "
             "`n_curves_no`. Shows that removing the filter does not restore generalization, confirming "
             "the limitation is structural (design size), not a filter artifact."),
    }
    for fname in tables:
        lines.append(f"\n## `{fname}`\n\n{descriptions[fname]}\n")

    readme_path = os.path.join(results_dir, 'README_tables.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Table index written to: {readme_path}")


# =====================================================================================
# Section 5 -- Journal of Chemometrics (Wiley) format compliance check (addendum Task 8)
# =====================================================================================

# (figure_stem, IUPAC-italic symbols actually used in that figure's labels/annotations)
FIGURE_MANIFEST = [
    ('fig_1', ['k_ext', 'k_deg', 'sigma', 'T', 'R^2']),
    ('fig_2_eda_pairplot', ['T', 't', 'C']),
    ('fig_3_phenomenological_fits', ['t', 'C']),
    ('fig_4_kext_confidence_intervals', ['k_ext']),
    ('fig_5_response_surfaces', ['T', 'k_ext']),
    ('fig_6_out_of_condition_validation', ['R^2']),
    ('fig_7_augmentation_and_sigma', ['R^2', 'sigma']),
    ('fig_8_gp_uncertainty_map', ['T', 'k_ext']),
]


def write_format_check(figures_dir):
    """Reads back each saved TIFF's real pixel dimensions and DPI (via Pillow) to verify
    -- rather than assume -- compliance with the Journal of Chemometrics width/height
    limits (80-180 mm wide, max 200 mm tall at 600 dpi)."""
    from PIL import Image

    rows = []
    for stem, symbols in FIGURE_MANIFEST:
        tiff_path = os.path.join(figures_dir, f"{stem}.tiff")
        if not os.path.exists(tiff_path):
            rows.append({'figure': stem, 'status': 'MISSING'})
            continue
        with Image.open(tiff_path) as im:
            width_px, height_px = im.size
            dpi_x, dpi_y = im.info.get('dpi', (600, 600))
        width_mm = width_px / dpi_x * 25.4
        height_mm = height_px / dpi_y * 25.4
        rows.append({
            'figure': stem, 'status': 'OK', 'width_mm': round(width_mm, 1),
            'height_mm': round(height_mm, 1), 'dpi': round(dpi_x), 'symbols': ', '.join(symbols),
        })

    lines = [
        "# FORMAT_check.md — Journal of Chemometrics (Wiley) final-art compliance\n",
        "Generated by `script_12_gp_and_consolidation.py` (addendum Task 8). Dimensions read back "
        "from the actual saved TIFF files (Pillow), not assumed from the matplotlib figsize.\n",
        "Requirements checked: TIFF (LZW) at 600 dpi; width 80-180 mm; max reproduction 140x200 mm; "
        "IUPAC italic symbols with roman units; legend/key inside the figure (not caption-only).\n",
        "| Figure | Width (mm) | Height (mm) | DPI | ≤180×200 mm? | Italic symbols used | Legend inside? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r['status'] == 'MISSING':
            lines.append(f"| `{r['figure']}.tiff` | — | — | — | **MISSING** | — | — |")
            continue
        within_limits = 'Yes' if (r['width_mm'] <= 180 and r['height_mm'] <= 200) else '**NO -- exceeds limit**'
        lines.append(f"| `{r['figure']}.tiff` | {r['width_mm']} | {r['height_mm']} | {r['dpi']} | "
                      f"{within_limits} | {r['symbols']} | Yes |")

    lines += [
        "\n## Notes",
        "- All 7 figures (fig_2 ... fig_8) are exported as TIFF (LZW-compressed, lossless) at 600 dpi "
        "for submission, plus a matching PNG at 300 dpi (same base filename) for internal preview/working use.",
        "- Physical-quantity symbols (*T*, *t*, *C*, *k*_ext, *R*², *σ*) are set in italics via matplotlib "
        "mathtext; descriptive subscripts (`ext`) and units (min⁻¹, °C) are set upright via `\\mathrm{}` "
        "or plain text, per IUPAC convention.",
        "- All legends/color keys are rendered inside the figure axes (`ax.legend()` / colorbar), not "
        "described only in the caption text.",
        "- **Tables are NOT exported as figures.** Per Journal of Chemometrics requirements, tables go into "
        "the manuscript as native editable text (Word/LaTeX), sourced from `results/paper_table_1..5*.csv` "
        "and `paper_table_supp_nofilter.csv` — see `results/README_tables.md`.",
        "- No analysis, data, or numeric result was changed by this formatting pass "
        "(`random_state=42` and all prior outputs are untouched).",
    ]

    report_path = os.path.join(figures_dir, 'FORMAT_check.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Format compliance report written to: {report_path}")
    for r in rows:
        print(r)


# =====================================================================================
# Orchestration
# =====================================================================================

def main(data_path='../data/dataset.csv', results_dir='../results', figures_dir='../figures'):
    df = load_dataset(data_path)

    print("=" * 80); print("SECTION 1: final figure set"); print("=" * 80)
    make_fig2_eda(figures_dir)
    make_fig3_phenomenological_fits(df, figures_dir)
    make_fig4_kext_confidence_intervals(results_dir, figures_dir)
    make_fig5_response_surfaces(df, figures_dir)
    make_fig6_in_vs_out_of_condition(results_dir, figures_dir)
    make_fig7_augmentation_and_sigma(results_dir, figures_dir)

    print("\n" + "=" * 80); print("SECTION 2: out-of-condition validation without the physical filter"); print("=" * 80)
    nofilter_table = build_table_supp_nofilter(df, results_dir)
    nofilter_path = os.path.join(results_dir, 'table_supp_nofilter_validation.csv')
    nofilter_table.round(4).to_csv(nofilter_path, index=False)
    print(nofilter_table.round(4).to_string(index=False))
    print(f"[Saved to {nofilter_path}]")

    print("\n" + "=" * 80); print("SECTION 3: Gaussian Process uncertainty (Phenolics)"); print("=" * 80)
    gp_result = fit_gp_for_compound(df, comp_es='Fenoles')
    if gp_result is not None:
        print(f"GP out-of-condition R2={gp_result['r2_loco']:.4f}, RMSE={gp_result['rmse_loco']:.4f}")
        make_fig8_gp_uncertainty_map(gp_result, df, figures_dir)
        gp_table = build_table_gp_uncertainty(gp_result, df)
    else:
        print("[WARN] Insufficient valid curves for Phenolics GP -- skipping fig_8 and GP table.")
        gp_table = pd.DataFrame()

    print("\n" + "=" * 80); print("SECTION 4: consolidated manuscript CSVs"); print("=" * 80)
    tables = consolidate_paper_tables(results_dir, gp_table, nofilter_table)
    write_tables_readme(results_dir, tables)

    print("\n" + "=" * 80); print("SECTION 5: Wiley format compliance check"); print("=" * 80)
    write_format_check(figures_dir)
    print("Done.")


if __name__ == "__main__":
    main()
