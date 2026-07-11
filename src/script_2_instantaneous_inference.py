"""
Script 2: Instantaneous ML Inference (Table 1 & Parity Plots)
Description: Evaluates the predictive performance (R^2 and RMSE) of four algorithms 
across all target bioactive properties using a Leave-One-Batch-Out strategy.
Outputs a comprehensive metrics table (Table 1) and generates the parity plot 
for the representative case (Anthocyanins).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def generate_inference_metrics_and_plots(data_path: str, output_dir: str) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found.")

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    for col in ['flor', 'compuesto', 'relacion', 'disolvente']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    targets_map = {
        'Antocianinas': 'Anthocyanins',
        'Fenoles': 'Phenolics',
        'Flavonoides (ABTS)': 'Antioxidant Capacity (ABTS)',
        'Flavonoides (DPPH)': 'Antioxidant Capacity (DPPH)'
    }

    numeric_features = ['A_band1', 'A_band2', 'A_band3', 'T_C', 'tiempo_min']
    cat_features = ['disolvente', 'relacion']

    results_table = []
    
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre todos los compuestos para generar la Tabla 1
    for comp_es, comp_en in targets_map.items():
        mask_valid = (df['compuesto'] == comp_es) & df['conc_promedio'].notna() & \
                     df['A_band1'].notna() & df['A_band2'].notna() & df['A_band3'].notna()
        sub = df[mask_valid].copy()
        
        if len(sub) < 10: continue
            
        sub['id_cinetica'] = pd.to_numeric(sub['id_cinetica'], errors='coerce')
        uniq_ids = sub['id_cinetica'].dropna().unique()
        
        # Leave-One-Batch-Out Split
        train_ids, test_ids = train_test_split(uniq_ids, test_size=0.3, random_state=42)
        train_mask, test_mask = sub['id_cinetica'].isin(train_ids), sub['id_cinetica'].isin(test_ids)

        X_train = sub.loc[train_mask, numeric_features + cat_features]
        X_test = sub.loc[test_mask, numeric_features + cat_features]
        y_train, y_test = sub.loc[train_mask, 'conc_promedio'], sub.loc[test_mask, 'conc_promedio']
        groups_train = sub.loc[train_mask, 'id_cinetica']

        # --- Model 1: Beer-Lambert ---
        corr = sub[['A_band1', 'A_band2', 'A_band3', 'conc_promedio']].corr()['conc_promedio'].drop('conc_promedio')
        best_band = corr.abs().idxmax()
        bl = LinearRegression().fit(sub.loc[train_mask, [best_band]], y_train)
        bl_pred = bl.predict(sub.loc[test_mask, [best_band]])

        # --- Model 2: PLS ---
        X_train_num, X_test_num = sub.loc[train_mask, ['A_band1', 'A_band2', 'A_band3']].values, sub.loc[test_mask, ['A_band1', 'A_band2', 'A_band3']].values
        pls = Pipeline([('scaler', StandardScaler()), ('pls', PLSRegression(n_components=min(3, X_train_num.shape[0]-1, X_train_num.shape[1])))])
        pls.fit(X_train_num, y_train)
        pls_pred = np.ravel(pls.predict(X_test_num))

        # --- Preprocessing for Ridge/RF ---
        preprocess = ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
        gkf = GroupKFold(n_splits=min(3, len(train_ids)))

        # --- Model 3: Ridge ---
        pipe_ridge = Pipeline([('prep', preprocess), ('model', Ridge(random_state=42))])
        grid_ridge = GridSearchCV(pipe_ridge, param_grid={'model__alpha': [0.1, 1.0, 10.0]}, cv=gkf, scoring='r2')
        grid_ridge.fit(X_train, y_train, groups=groups_train)
        ridge_pred = grid_ridge.predict(X_test)

        # --- Model 4: Random Forest ---
        pipe_rf = Pipeline([('prep', preprocess), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
        grid_rf = GridSearchCV(pipe_rf, param_grid={'model__n_estimators': [100], 'model__max_depth': [None, 10]}, cv=gkf, scoring='r2')
        grid_rf.fit(X_train, y_train, groups=groups_train)
        rf_pred = grid_rf.predict(X_test)

        # Recopilar métricas
        models_preds = {
            'Beer-Lambert': bl_pred, 'PLS': pls_pred, 
            'Ridge': ridge_pred, 'Random Forest': rf_pred
        }
        
        row = {'Compound': comp_en}
        for m_name, preds in models_preds.items():
            row[f'{m_name} R2'] = r2_score(y_test, preds)
            row[f'{m_name} RMSE'] = mean_squared_error(y_test, preds)**0.5
        results_table.append(row)

        # Generar gráfica SÓLO para Antocianinas (Caso representativo)
        if comp_en == 'Anthocyanins':
            sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes, colors = axes.ravel(), ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
            plot_data = [('Beer-Lambert', bl_pred), ('PLS (3 Wavelengths)', pls_pred), ('Ridge', ridge_pred), ('Random Forest', rf_pred)]
            
            for ax, (name, pred), color in zip(axes, plot_data, colors):
                ax.scatter(y_test, pred, alpha=0.8, color=color, s=80, edgecolor='white', linewidth=0.5)
                min_v, max_v = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
                buffer = max((max_v - min_v) * 0.08, 0.1)
                ax.plot([min_v - buffer, max_v + buffer], [min_v - buffer, max_v + buffer], 'k--', lw=1.5, alpha=0.7)
                ax.set_title(name, fontsize=18, fontweight='bold', pad=12)
                ax.set_xlabel('True Concentration (mg/L)', fontsize=16)
                if name == 'Beer-Lambert': ax.set_ylabel('Predicted Concentration (mg/L)', fontsize=16)
                ax.set_xlim(min_v - buffer, max_v + buffer); ax.set_ylim(min_v - buffer, max_v + buffer)
                ax.text(0.05, 0.95, f"R² = {r2_score(y_test, pred):.3f}\nRMSE = {mean_squared_error(y_test, pred)**0.5:.3f}", 
                        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=14)

            plt.suptitle('Parity Plots: Instantaneous Inference for Anthocyanins', y=1.08, fontsize=22, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_3_parity_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Guardar e imprimir la Tabla 1
    df_results = pd.DataFrame(results_table)
    df_results = df_results.round(3)
    table_path = os.path.join(output_dir, 'table_1_inference_metrics.csv')
    df_results.to_csv(table_path, index=False)
    
    print("\n" + "="*80)
    print("TABLE 1: Instantaneous Calibration/Inference Metrics")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80 + f"\n[Saved to {table_path}]\n")

if __name__ == "__main__":
    generate_inference_metrics_and_plots('dataset.csv', 'results_figures')