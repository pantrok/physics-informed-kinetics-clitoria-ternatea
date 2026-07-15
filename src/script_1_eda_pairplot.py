"""
Script 1: Exploratory Data Analysis (EDA) - Bivariate Relationships
Description: Generates a pair plot to visualize the distribution and correlations between
operational extraction variables, UV-Vis absorbance features, and target outputs for
Clitoria ternatea extracts. Exported as fig_2 in Journal of Chemometrics (Wiley) final-art
format: TIFF 600 dpi (LZW) for submission + PNG 300 dpi for working/preview use, width
within the 80-180 mm range, IUPAC italic symbols with roman (upright) units.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend -- avoids Tk/Tcl crashes in batch figure generation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from figure_export import save_figure

FIG_WIDTH_MM = 178  # target width, within Wiley's 80-180 mm range (see facet_height_in below)


def generate_eda_pairplot(data_path: str, output_dir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    
    for col in ['flor', 'compuesto', 'relacion', 'disolvente', 'metrica_conc']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required_columns = ['A_replicate_1', 'A_replicate_2', 'A_replicate_3', 'T_C', 'tiempo_min', 'conc_promedio']
    df_clean = df.dropna(subset=required_columns).copy()
    df_clean = df_clean[df_clean['compuesto'] != '']

    # Short forms for the legend only (standard assay abbreviations) -- keeps the legend
    # narrow enough that seaborn's corner-plot legend addon stays within the Wiley width
    # limit; full compound names are used throughout every other figure/table.
    compound_mapping = {
        'Antocianinas': 'Anthocyanins',
        'Fenoles': 'Phenolics',
        'Flavonoides (ABTS)': 'ABTS',
        'Flavonoides (DPPH)': 'DPPH'
    }
    df_clean['compuesto'] = df_clean['compuesto'].replace(compound_mapping)

    # IUPAC: symbols in italics (mathtext default), units/descriptive subscripts in roman (\mathrm{})
    feature_mapping = {
        'T_C': '$T$ (°C)',
        'tiempo_min': '$t$ (min)',
        'A_replicate_1': 'Abs. 1',
        'A_replicate_2': 'Abs. 2',
        'A_replicate_3': 'Abs. 3',
        'conc_promedio': '$C$',
    }

    columns_to_plot = list(feature_mapping.keys())
    df_plot = df_clean[columns_to_plot + ['compuesto']].rename(columns=feature_mapping)

    # seaborn's corner=True pairplot widens the canvas beyond the n_vars-square grid to fit
    # the auto-placed legend outside it (confirmed empirically: legend addon ~1.46 in wide
    # with these short labels). height=0.92 in/facet was tuned so square+legend addon lands
    # just under the 178 mm target (~177 mm before the final tight-bbox crop).
    facet_height_in = 0.92

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    g = sns.pairplot(
        df_plot, hue='compuesto', corner=True, height=facet_height_in, aspect=1.0,
        plot_kws={'alpha': 0.6, 's': 18, 'edgecolor': 'none'}, palette='husl'
    )

    g.fig.suptitle('EDA: Bivariate Relationships and Distributions', y=1.02, fontsize=13, fontweight='bold')
    # Style the legend in place (do NOT move it with an external bbox_to_anchor -- that
    # pulled the saved canvas far beyond the 180 mm Wiley width limit in an earlier pass).
    if g._legend is not None:
        g._legend.set_title('Compound')
        g._legend.get_title().set_fontsize(8)
        for text in g._legend.get_texts():
            text.set_fontsize(7)
    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(labelsize=6)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=7.5)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=7.5)

    tiff_path, png_path = save_figure(g.fig, output_dir, 'fig_2_eda_pairplot')
    plt.close(g.fig)
    print(f"EDA pairplot generated at: {tiff_path} and {png_path}")


if __name__ == "__main__":
    generate_eda_pairplot(os.path.join('..', 'data', 'dataset.csv'), os.path.join('..', 'figures'))