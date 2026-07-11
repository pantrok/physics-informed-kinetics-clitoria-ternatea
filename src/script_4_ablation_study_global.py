"""
Script 4: Global Impact of Feature Reduction (Ablation Study)
Description: Evaluates the predictive performance across all targets 
using only Top 1, Top 2, and All Features to prove multivariable necessity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

def generate_full_ablation_study(data_path: str, output_dir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    for col in ['flor', 'compuesto', 'relacion', 'disolvente']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    targets_map = {
        'Antocianinas': 'Anthocyanins', 'Fenoles': 'Phenolics',
        'Flavonoides (ABTS)': 'Antioxidant Capacity (ABTS)',
        'Flavonoides (DPPH)': 'Antioxidant Capacity (DPPH)'
    }

    numeric_features = ['A_replicate_1', 'A_replicate_2', 'A_replicate_3', 'T_C', 'tiempo_min']
    cat_features = ['disolvente', 'relacion']
    results = []

    for comp_es, comp_en in targets_map.items():
        mask = (df['compuesto'] == comp_es) & df['conc_promedio'].notna() & df['A_replicate_1'].notna()
        sub = df[mask].copy()
        if len(sub) < 10: continue
            
        sub['id_cinetica'] = pd.to_numeric(sub['id_cinetica'], errors='coerce')
        uniq_ids = sub['id_cinetica'].dropna().unique()
        train_ids, test_ids = train_test_split(uniq_ids, test_size=0.3, random_state=42)

        train_mask = sub['id_cinetica'].isin(train_ids)
        test_mask = sub['id_cinetica'].isin(test_ids)

        X_train_raw = sub.loc[train_mask, numeric_features + cat_features]
        X_test_raw = sub.loc[test_mask, numeric_features + cat_features]
        y_train, y_test = sub.loc[train_mask, 'conc_promedio'], sub.loc[test_mask, 'conc_promedio']

        preprocess = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])

        X_train_prep = preprocess.fit_transform(X_train_raw)
        X_test_prep = preprocess.transform(X_test_raw)

        rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_base.fit(X_train_prep, y_train)
        indices = np.argsort(rf_base.feature_importances_)[::-1]

        subset_labels = ['Top 1 Feature', 'Top 2 Features', 'All Features']
        k_values = [1, 2, len(indices)]

        for k, label in zip(k_values, subset_labels):
            top_k_indices = indices[:k]
            X_tr_k, X_te_k = X_train_prep[:, top_k_indices], X_test_prep[:, top_k_indices]
            
            rf_k = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_k.fit(X_tr_k, y_train)
            r2 = r2_score(y_test, rf_k.predict(X_te_k))
            
            results.append({'Target': comp_en, 'Feature Subset': label, 'R2_real': r2, 'R2_plot': max(0, r2)})

    res_df = pd.DataFrame(results)

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    plt.figure(figsize=(14, 8))
    palette = {'Top 1 Feature': '#c44e52', 'Top 2 Features': '#dd8452', 'All Features': '#55a868'}
    
    ax = sns.barplot(data=res_df, x='Target', y='R2_plot', hue='Feature Subset', palette=palette, edgecolor='black', linewidth=1)
    
    plt.title('Impact of Feature Reduction across Targets', fontsize=20, fontweight='bold', pad=15)
    plt.xlabel('Target Bioactive Compound / Property', fontsize=16)
    plt.ylabel('Predictive Performance ($R^2$)', fontsize=16)
    plt.ylim(0, 1.15)

    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            subset_label = list(palette.keys())[i]
            target_label = res_df['Target'].unique()[j]
            real_r2 = res_df[(res_df['Target'] == target_label) & (res_df['Feature Subset'] == subset_label)]['R2_real'].values[0]
            y_pos = max(bar.get_height(), 0.02)
            ax.annotate(f"{real_r2:.2f}", xy=(bar.get_x() + bar.get_width() / 2, y_pos), xytext=(0, 5),
                        textcoords="offset points", ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.legend(title='Feature Subset', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, title_fontsize=15)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'fig_5_impact_feature_reduction.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global ablation study generated at: {out_path}")

if __name__ == "__main__":
    generate_full_ablation_study('dataset.csv', 'results_figures')