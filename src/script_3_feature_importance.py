"""
Script 3: Random Forest Feature Importance
Description: Calculates and visualizes the Gini impurity reduction 
for each descriptor utilized by the optimal Random Forest regressor.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def generate_feature_importance_plot(data_path: str, output_dir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    for col in ['flor', 'compuesto', 'relacion', 'disolvente']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    comp, target = 'Antocianinas', 'conc_promedio'
    numeric_features = ['A_band1', 'A_band2', 'A_band3', 'T_C', 'tiempo_min']
    cat_features = ['disolvente', 'relacion']

    mask_valid = (df['compuesto'] == comp) & df[target].notna() & df['A_band1'].notna()
    sub = df[mask_valid].copy()

    X, y = sub[numeric_features + cat_features], sub[target]

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    rf_pipeline = Pipeline([
        ('prep', preprocess),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X, y)

    cat_encoder = rf_pipeline.named_steps['prep'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(cat_features)
    all_features = numeric_features + list(cat_names)
    importances = rf_pipeline.named_steps['model'].feature_importances_

    imp_df = pd.DataFrame({'Descriptor': all_features, 'Importancia': importances})

    rename_map = {
        'A_band1': 'Abs 1', 'A_band2': 'Abs 2', 'A_band3': 'Abs 3',
        'T_C': 'Temperature (°C)', 'tiempo_min': 'Time (min)',
        'disolvente_Agua100': 'Solvent: Water (100%)',
        'disolvente_Agua50_ETA50': 'Solvent: Water/EtOH (50:50)',
        'disolvente_Agua75_ETA25': 'Solvent: Water/EtOH (75:25)',
        'relacion_1_10': 'Ratio: 1:10', 'relacion_1_20': 'Ratio: 1:20', 'relacion_1_30': 'Ratio: 1:30'
    }

    imp_df['Descriptor'] = imp_df['Descriptor'].map(lambda x: rename_map.get(x, x))
    imp_df = imp_df.sort_values(by='Importancia', ascending=False)

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=imp_df, x='Importancia', y='Descriptor', palette='viridis')

    plt.title('Random Forest Feature Importance (Anthocyanins)', fontsize=20, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance (Impurity Reduction)', fontsize=16)
    plt.ylabel('', fontsize=15)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=14)

    plt.xlim(0, imp_df['Importancia'].max() * 1.15)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_4_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance generated at: {output_path}")

if __name__ == "__main__":
    generate_feature_importance_plot('dataset.csv', 'results_figures')