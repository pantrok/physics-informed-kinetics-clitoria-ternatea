"""
Script 1: Exploratory Data Analysis (EDA) - Bivariate Relationships
Description: Generates a high-resolution pair plot (300 DPI) to visualize 
the distribution and correlations between operational extraction variables, 
UV-Vis absorbance features, and target outputs for Clitoria ternatea extracts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda_pairplot(data_path: str, output_dir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    
    for col in ['flor', 'compuesto', 'relacion', 'disolvente', 'metrica_conc']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required_columns = ['A_band1', 'A_band2', 'A_band3', 'T_C', 'tiempo_min', 'conc_promedio']
    df_clean = df.dropna(subset=required_columns).copy()
    df_clean = df_clean[df_clean['compuesto'] != '']

    compound_mapping = {
        'Antocianinas': 'Anthocyanins',
        'Fenoles': 'Phenolics',
        'Flavonoides (ABTS)': 'Antioxidant Capacity (ABTS)',
        'Flavonoides (DPPH)': 'Antioxidant Capacity (DPPH)'
    }
    df_clean['compuesto'] = df_clean['compuesto'].replace(compound_mapping)

    feature_mapping = {
        'T_C': 'Temperature (°C)',
        'tiempo_min': 'Time (min)',
        'A_band1': 'Abs 1',
        'A_band2': 'Abs 2',
        'A_band3': 'Abs 3',
        'conc_promedio': 'Target Value'
    }
    
    columns_to_plot = list(feature_mapping.keys())
    df_plot = df_clean[columns_to_plot + ['compuesto']].rename(columns=feature_mapping)

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    g = sns.pairplot(
        df_plot, hue='compuesto', corner=True, 
        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'none'}, palette='husl'
    )

    g.fig.suptitle('EDA: Bivariate Relationships and Distributions', y=1.02, fontsize=20, fontweight='bold')
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.95, 0.95), title='Target Output', frameon=True, fontsize=14, title_fontsize=16)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_2_pairplot_eda.png')
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"EDA pairplot generated at: {output_path}")


if __name__ == "__main__":
    generate_eda_pairplot(os.path.join('..', 'data', 'dataset.csv'), os.path.join('..', 'figures'))