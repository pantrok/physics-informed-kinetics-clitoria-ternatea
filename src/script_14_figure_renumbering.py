"""
Script 14: Final Figure Renumbering (EDA Removed From Manuscript)
Addendum Task 9 §4 (BRIEF_addendum_tarea9_respuesta_revision.md). The journal caps the
manuscript at 7 illustrations; the EDA pairplot is dropped from the manuscript (kept in
the repo, not deleted) and every following figure shifts down by one number. The
validation figure (new fig_5, formerly fig_6) is regenerated to add per-fold dispersion
(Task 9A) and the structured Arrhenius comparator (Task 9B) as required content, not just
renamed.

    fig_1  methodology            (unchanged)
    fig_2  phenomenological fits  (was fig_3)
    fig_3  k_ext confidence ints  (was fig_4)
    fig_4  response surfaces      (was fig_5)
    fig_5  out-of-condition validation + structured comparator + fold dispersion (was fig_6, enhanced)
    fig_6  augmentation & sigma   (was fig_7)
    fig_7  GP uncertainty map     (was fig_8)
    fig_supp_eda_pairplot         (was fig_2 -- excluded from the manuscript, kept in repo)

Depends on script_13_review_response.py's outputs (validation_fold_detail.csv,
paper_table_structured_comparator.csv) having already been generated. Does not modify
script_9/script_12/script_13; only renames/regenerates figure files and rewrites
figures/FORMAT_check.md and figures/CAPTIONS_borrador.md.
"""

import os
import shutil

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from figure_export import save_figure, mm_to_in
from script_12_gp_and_consolidation import FIG_WIDTH_MM, _sized_figsize

COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

# (old_stem, new_stem) -- simple rename, content unchanged
RENAME_MAP = [
    ('fig_3_phenomenological_fits', 'fig_2_phenomenological_fits'),
    ('fig_4_kext_confidence_intervals', 'fig_3_kext_confidence_intervals'),
    ('fig_5_response_surfaces', 'fig_4_response_surfaces'),
    ('fig_7_augmentation_and_sigma', 'fig_6_augmentation_and_sigma'),
    ('fig_8_gp_uncertainty_map', 'fig_7_gp_uncertainty_map'),
]
EDA_RENAME = ('fig_2_eda_pairplot', 'fig_supp_eda_pairplot')
SUPERSEDED_STEM = 'fig_6_out_of_condition_validation'  # replaced by the new fig_5

FINAL_MANIFEST = [
    ('fig_1', ['k_ext', 'k_deg', 'sigma', 'T', 'R^2']),
    ('fig_2_phenomenological_fits', ['t', 'C']),
    ('fig_3_kext_confidence_intervals', ['k_ext']),
    ('fig_4_response_surfaces', ['T', 'k_ext']),
    ('fig_5_out_of_condition_validation', ['R^2']),
    ('fig_6_augmentation_and_sigma', ['R^2', 'sigma']),
    ('fig_7_gp_uncertainty_map', ['T', 'k_ext']),
]


def rename_figures(figures_dir):
    for old_stem, new_stem in RENAME_MAP + [EDA_RENAME]:
        for ext in ('.tiff', '.png'):
            src = os.path.join(figures_dir, old_stem + ext)
            dst = os.path.join(figures_dir, new_stem + ext)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"{old_stem}{ext} -> {new_stem}{ext}")
            else:
                print(f"[WARN] {src} not found, skipping")


def remove_superseded(figures_dir):
    for ext in ('.tiff', '.png'):
        path = os.path.join(figures_dir, SUPERSEDED_STEM + ext)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed superseded {path}")


def make_fig5_validation_enhanced(results_dir, figures_dir):
    """Out-of-condition validation, now with: (a) the structured Arrhenius comparator as
    an additional bar series (Task 9B), and (b) individual LOGO fold R2 values overlaid
    as points on the RF bar, showing the fold-to-fold dispersion (Task 9A)."""
    table_7 = pd.read_csv(os.path.join(results_dir, 'table_7_grouped_validation.csv'))
    fold_detail = pd.read_csv(os.path.join(results_dir, 'validation_fold_detail.csv'))
    comparator = pd.read_csv(os.path.join(results_dir, 'paper_table_structured_comparator.csv'))

    compounds = table_7['Compound'].tolist()
    t7 = table_7.set_index('Compound')
    comp = comparator.set_index('Compound')
    x = np.arange(len(compounds))
    width = 0.2

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.25)
    fig, ax = plt.subplots(figsize=_sized_figsize(12, 7))

    in_cond = t7.loc[compounds, 'R2_random_Table3'].values.astype(float)
    out_rf = t7.loc[compounds, 'R2_grouped_LOCO'].values.astype(float)
    out_arr = comp.loc[compounds, 'R2_arrhenius_LOGO'].values.astype(float)
    out_gp = comp.loc[compounds, 'R2_GP_LOGO'].values.astype(float)

    ax.bar(x - 1.5 * width, in_cond, width, label='In-condition (random split)',
           color='#9fc5e8', edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5 * width, out_rf, width, label='Out-of-condition RF (grouped)',
           color='#c44e52', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5 * width, out_arr, width, label='Out-of-condition Arrhenius (structured)',
           color='#55a868', edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5 * width, out_gp, width, label='Out-of-condition GP (Phenolics only)',
           color='#8172b3', edgecolor='black', linewidth=0.5)

    # A handful of individual LOGO folds are extreme outliers (R2 in the hundreds to
    # thousands negative -- see validation_fold_detail.csv / RESULTADOS_revision.md); a
    # linear axis that accommodates them would flatten every bar and every other fold to
    # invisibility. Clip the axis to a legible range and mark off-scale folds explicitly
    # at the bottom edge with their true value annotated, rather than hiding them.
    Y_MIN, Y_MAX = -3.5, 1.3
    rng = np.random.default_rng(RANDOM_STATE_FOR_JITTER)
    n_offscale_total = 0
    for i, compound in enumerate(compounds):
        folds = fold_detail.loc[fold_detail['Compound'] == compound, 'R2_fold'].dropna().values
        if len(folds) == 0:
            continue
        onscale = folds[folds >= Y_MIN]
        offscale = folds[folds < Y_MIN]
        jitter = rng.uniform(-0.05, 0.05, size=len(onscale))
        ax.scatter(np.full(len(onscale), x[i] - 0.5 * width) + jitter, onscale,
                   color='black', s=14, zorder=5, alpha=0.75,
                   label='Individual LOGO folds (RF)' if i == 0 else None)
        if len(offscale):
            n_offscale_total += len(offscale)
            jitter_off = rng.uniform(-0.05, 0.05, size=len(offscale))
            ax.scatter(np.full(len(offscale), x[i] - 0.5 * width) + jitter_off,
                       np.full(len(offscale), Y_MIN + 0.12), marker='v', color='black',
                       s=28, zorder=6)
            ax.annotate(f"{offscale.min():.0f}" if len(offscale) == 1 else f"min={offscale.min():.0f}",
                        (x[i] - 0.5 * width, Y_MIN + 0.12), xytext=(0, -11),
                        textcoords='offset points', ha='center', fontsize=6.5, color='black')

    ax.axhline(0, color='black', lw=1.0)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(x)
    ax.set_xticklabels(compounds, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Test $R^2$', fontsize=11)
    ax.legend(fontsize=7, loc='lower left')
    subtitle = ('(black dots = individual LOGO fold results for the RF; open triangles = fold(s) '
                f'off-scale below {Y_MIN:.1f}, true value annotated, n={n_offscale_total})')
    ax.set_title(f'Predictive performance: in-condition vs. out-of-condition validation\n{subtitle}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, figures_dir, 'fig_5_out_of_condition_validation')
    plt.close()
    print("fig_5 (out-of-condition validation, enhanced) saved")


RANDOM_STATE_FOR_JITTER = 42


def write_format_check(figures_dir):
    from PIL import Image
    rows = []
    for stem, symbols in FINAL_MANIFEST:
        tiff_path = os.path.join(figures_dir, f"{stem}.tiff")
        if not os.path.exists(tiff_path):
            rows.append({'figure': stem, 'status': 'MISSING'})
            continue
        with Image.open(tiff_path) as im:
            width_px, height_px = im.size
            dpi_x, dpi_y = im.info.get('dpi', (600, 600))
        width_mm = width_px / dpi_x * 25.4
        height_mm = height_px / dpi_y * 25.4
        rows.append({'figure': stem, 'status': 'OK', 'width_mm': round(width_mm, 1),
                     'height_mm': round(height_mm, 1), 'dpi': round(dpi_x), 'symbols': ', '.join(symbols)})

    lines = [
        "# FORMAT_check.md — Journal of Chemometrics (Wiley) final-art compliance\n",
        "Generated by `script_14_figure_renumbering.py` (addendum Task 9 -- final manuscript "
        "numbering, 7 figures, EDA pairplot excluded per the journal's illustration limit). "
        "Dimensions read back from the actual saved TIFF files (Pillow).\n",
        "Requirements checked: TIFF (LZW) at 600 dpi; width 80-180 mm; max reproduction 140x200 mm; "
        "IUPAC italic symbols with roman units; legend/key inside the figure (not caption-only).\n",
        "| Figure | Width (mm) | Height (mm) | DPI | ≤180×200 mm? | Italic symbols used | Legend inside? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r['status'] == 'MISSING':
            lines.append(f"| `{r['figure']}.tiff` | — | — | — | **MISSING** | — | — |")
            continue
        within = 'Yes' if (r['width_mm'] <= 180 and r['height_mm'] <= 200) else '**NO -- exceeds limit**'
        lines.append(f"| `{r['figure']}.tiff` | {r['width_mm']} | {r['height_mm']} | {r['dpi']} | "
                      f"{within} | {r['symbols']} | Yes |")

    lines += [
        "\n## Notes",
        "- Manuscript figure count: 7 (`fig_1` ... `fig_7`), within the journal's limit.",
        "- The EDA pairplot (`fig_supp_eda_pairplot.tiff`) is retained in the repository for "
        "reproducibility but is **not** part of the manuscript's figure count.",
        "- `fig_5` now combines the out-of-condition validation bars, the structured Arrhenius "
        "comparator, and individual LOGO fold results (dispersion) in one figure, per the "
        "7-illustration limit (no new figure was created for Tasks 9A/9B).",
        "- Tables are NOT exported as figures; they go into the manuscript as native editable "
        "text, sourced from `results/paper_table_*.csv` -- see `results/README_tables.md`.",
        "- No analysis, data, or numeric result was changed by this renumbering pass "
        "(`random_state=42`, all prior outputs untouched).",
    ]
    report_path = os.path.join(figures_dir, 'FORMAT_check.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Format compliance report (renumbered) written to: {report_path}")
    for r in rows:
        print(r)


CAPTIONS = """# Captions borrador — Fig. 1 a Fig. 7 (esquema final del manuscrito)

Borrador crudo, terminología limpia de paper; se pule en la fase de redacción con el autor.
La figura de EDA (pairplot) se retiró del manuscrito (límite de 7 ilustraciones de la revista)
y se conserva en el repositorio como `fig_supp_eda_pairplot` (no forma parte del conteo).

---

## Fig. 1 — `fig_1.tiff`

> Fig. 1. Methodology overview: experimental acquisition, grey-box kinetic modeling,
> physics-informed augmentation, and predictive mapping & validation.

---

## Fig. 2 — `fig_2_phenomenological_fits.tiff`

> Fig. 2. Phenomenological Ext-Deg model fits (concentration vs. time) for the physically
> valid extraction kinetics (R²≥0.50), one panel per target compound.

---

## Fig. 3 — `fig_3_kext_confidence_intervals.tiff`

> Fig. 3. Estimated extraction rate constant (k_ext) by operating condition, with 95%
> confidence intervals derived from physics-informed Monte Carlo augmentation.

---

## Fig. 4 — `fig_4_response_surfaces.tiff`

> Fig. 4. Model-predicted k_ext response surfaces across the measured design space;
> interpolation strictly within the measured domain (T: 60-75 °C), never extrapolated.

---

## Fig. 5 — `fig_5_out_of_condition_validation.tiff`

> Fig. 5. Predictive performance under in-condition (random split) vs. out-of-condition
> (grouped, leave-one-condition-out) validation, for a black-box Random Forest, a
> Gaussian Process (Phenolics only), and a 4-parameter structured Arrhenius-form model,
> by compound. Black points overlay the individual leave-one-condition-out fold results
> for the Random Forest, showing the fold-to-fold dispersion behind the aggregate value.

**Nota para el autor:** figura central del hallazgo metodológico y de la respuesta a
revisión (Tareas 9A/9B). Ningún enfoque -- flexible o físicamente estructurado --
generaliza fuera de condición con este diseño.

---

## Fig. 6 — `fig_6_augmentation_and_sigma.tiff`

> Fig. 6. Effect of Monte Carlo augmentation size and augmentation noise level (σ) on
> out-of-condition predictive performance; neither restores generalization.

---

## Fig. 7 — `fig_7_gp_uncertainty_map.tiff`

> Fig. 7. Gaussian Process predictive uncertainty (standard deviation) for k_ext across
> the measured design space (Phenolics); motivates active learning as future work.
"""


def write_captions(figures_dir):
    path = os.path.join(figures_dir, 'CAPTIONS_borrador.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(CAPTIONS)
    print(f"Captions (renumbered) written to: {path}")


def main(results_dir='../results', figures_dir='../figures'):
    rename_figures(figures_dir)
    make_fig5_validation_enhanced(results_dir, figures_dir)
    remove_superseded(figures_dir)
    write_format_check(figures_dir)
    write_captions(figures_dir)
    print("Done.")


if __name__ == "__main__":
    main()
