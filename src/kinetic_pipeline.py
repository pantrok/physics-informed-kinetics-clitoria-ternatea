"""
Shared physics-informed kinetic pipeline: Ext-Deg phenomenological fit -> physical
R^2 filter -> Monte Carlo augmentation with variance filter -> RF mapping of
k_ext ~ [T_C, disolvente, relacion].

Extracted from script_5_augmented_kinetic_prediction.py so that the confidence-interval,
response-surface, augmentation-validation and RSM-comparison analyses (Tasks 1-4 of the
ENBIS reinforcement brief) reuse the exact same fitting/augmentation/training logic and
stay numerically consistent with Table 2 / Table 3 / Fig. 6.

Design-space bounds (never extrapolate beyond these): T_C in [60, 75] deg C,
the 3 disolvente levels and 3 relacion levels of the Box-Behnken design.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

TARGETS_MAP = {
    'Antocianinas': 'Anthocyanins',
    'Fenoles': 'Phenolics',
    'Flavonoides (ABTS)': 'Antioxidant Capacity (ABTS)',
    'Flavonoides (DPPH)': 'Antioxidant Capacity (DPPH)',
}

T_MIN, T_MAX = 60.0, 75.0


def ext_deg_model(t, C0, k_ext, k_deg):
    denom = k_deg - k_ext
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return C0 * (k_ext / denom) * (np.exp(-k_ext * t) - np.exp(-k_deg * t))


def load_dataset(data_path):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    for c in ['flor', 'compuesto', 'relacion', 'disolvente']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df['id_cinetica'] = pd.to_numeric(df['id_cinetica'], errors='coerce')
    return df


def fit_valid_curves(sub_df):
    """Phase A+B of script_5: phenomenological Ext-Deg fit per kinetic curve, then the
    physical validation filter (R^2 >= 0.50). Returns (curvas_validas, r2_all_fits)."""
    cineticas = sub_df['id_cinetica'].unique()
    r2_all_fits, curvas_validas = [], []
    for cid in cineticas:
        datos = sub_df[sub_df['id_cinetica'] == cid].sort_values(by='tiempo_min')
        if len(datos) < 4:
            continue
        t, C = datos['tiempo_min'].values, datos['conc_promedio'].values
        try:
            popt, _ = curve_fit(ext_deg_model, t, C, p0=[max(C) * 1.5, 0.05, 0.005],
                                 bounds=([0, 0, 0], [np.inf, 1.0, 1.0]), maxfev=10000)
            r2 = r2_score(C, ext_deg_model(t, *popt))
            r2_all_fits.append(r2)
            if r2 >= 0.50:
                curvas_validas.append({
                    'id_cinetica': cid,
                    'T_C': datos['T_C'].iloc[0], 'disolvente': datos['disolvente'].iloc[0],
                    'relacion': datos['relacion'].iloc[0], 'C0': popt[0],
                    'k_ext': popt[1], 'k_deg': popt[2], 't_real': t, 'r2_fit': r2,
                })
        except RuntimeError:
            pass
    return curvas_validas, r2_all_fits


def augment_curves(curvas_validas, n_synth=30, band=0.60, noise_frac=0.08, seed=RANDOM_STATE):
    """Phase C of script_5: Monte Carlo augmentation with the phenomenological variance
    filter (accept synthetic k_ext only within +/- `band` of the real curve's k_ext).

    Reproduces script_5's exact random-draw sequence (one np.random.seed(seed) call before
    looping over curves, n_synth inner draws per curve): n_synth=30, band=0.60 exactly
    reproduces the original Table 3 / Fig. 6 results.

    Returns (df_aug, audit_df). audit_df carries accepted/rejected counts per curve, needed
    for the Task 4 variance-filter audit.
    """
    np.random.seed(seed)
    datos_aug, audit = [], []
    for curva in curvas_validas:
        datos_aug.append({'T_C': curva['T_C'], 'disolvente': curva['disolvente'],
                           'relacion': curva['relacion'], 'k_ext': curva['k_ext'],
                           'id_cinetica': curva['id_cinetica'], 'synthetic': False})
        accepted, rejected = 0, 0
        for _ in range(n_synth):
            C_ideal = ext_deg_model(curva['t_real'], curva['C0'], curva['k_ext'], curva['k_deg'])
            C_sint = np.clip(C_ideal + np.random.normal(0, noise_frac * np.mean(C_ideal), len(curva['t_real'])), 0, None)
            try:
                popt_sint, _ = curve_fit(ext_deg_model, curva['t_real'], C_sint,
                                          p0=[curva['C0'], curva['k_ext'], curva['k_deg']],
                                          bounds=([0, 0, 0], [np.inf, 1.0, 1.0]), maxfev=10000)
                k_ext_sintetico = popt_sint[1]
                limite_inferior = curva['k_ext'] * (1 - band)
                limite_superior = curva['k_ext'] * (1 + band)
                if limite_inferior <= k_ext_sintetico <= limite_superior:
                    datos_aug.append({'T_C': curva['T_C'], 'disolvente': curva['disolvente'],
                                       'relacion': curva['relacion'], 'k_ext': k_ext_sintetico,
                                       'id_cinetica': curva['id_cinetica'], 'synthetic': True})
                    accepted += 1
                else:
                    rejected += 1
            except RuntimeError:
                rejected += 1
        audit.append({
            'id_cinetica': curva['id_cinetica'], 'T_C': curva['T_C'],
            'disolvente': curva['disolvente'], 'relacion': curva['relacion'],
            'n_attempted': n_synth, 'n_accepted': accepted, 'n_rejected': rejected,
            'rejection_rate': (rejected / n_synth) if n_synth > 0 else np.nan,
        })
    return pd.DataFrame(datos_aug), pd.DataFrame(audit)


def build_rf_pipeline(random_state=RANDOM_STATE):
    return Pipeline([
        ('prep', ColumnTransformer([('num', StandardScaler(), ['T_C']),
                                     ('cat', OneHotEncoder(handle_unknown='ignore'), ['disolvente', 'relacion'])])),
        ('model', RandomForestRegressor(random_state=random_state)),
    ])


def train_rf(df_aug, test_size=0.25, random_state=RANDOM_STATE):
    """Phase D of script_5: GridSearchCV-tuned RandomForest mapping k_ext ~ [T_C, disolvente, relacion]."""
    X, y = df_aug[['T_C', 'disolvente', 'relacion']], df_aug['k_ext']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = build_rf_pipeline(random_state)
    grid = GridSearchCV(pipe, {'model__n_estimators': [50, 100, 200],
                                'model__max_depth': [None, 5, 10],
                                'model__min_samples_split': [2, 5]},
                         cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    return {
        'grid': grid, 'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred,
        'r2': r2, 'rmse': rmse,
    }


def run_compound_pipeline(df, comp_es, comp_en, n_synth=30, band=0.60, seed=RANDOM_STATE):
    """Full per-compound pipeline (Phases A-D). Returns a dict; 'insufficient'=True means
    there were no physically-valid curves (R^2>=0.50) or the RF stage could not be trained
    (e.g. too few samples for the CV folds) -- both are legitimate, documented outcomes,
    not errors to silently work around."""
    sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
    result = {'compound_es': comp_es, 'compound_en': comp_en, 'insufficient': True}
    if sub_df.empty:
        return result

    curvas_validas, r2_all_fits = fit_valid_curves(sub_df)
    result.update({
        'curvas_validas': curvas_validas,
        'n_valid': len(curvas_validas),
        'mean_r2_all': np.mean(r2_all_fits) if r2_all_fits else 0.0,
        'mean_r2_valid': np.mean([c['r2_fit'] for c in curvas_validas]) if curvas_validas else 0.0,
    })
    if not curvas_validas:
        return result

    df_aug, audit_df = augment_curves(curvas_validas, n_synth=n_synth, band=band, seed=seed)
    result['df_aug'], result['audit'] = df_aug, audit_df
    if df_aug.empty:
        return result

    try:
        rf_result = train_rf(df_aug, random_state=seed)
    except (ValueError, IndexError) as e:
        result['error'] = str(e)
        return result

    result.update(rf_result)
    result['insufficient'] = False
    return result
