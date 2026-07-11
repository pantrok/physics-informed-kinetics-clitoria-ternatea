"""
Script 8: Classical RSM (Box-Behnken) vs. Random Forest comparison (Table 5 + ANOVA)
Task 3 of the ENBIS analytical-reinforcement brief.

Fits a classical second-order Response Surface polynomial (statsmodels OLS, with the
standard Box-Behnken ANOVA) to k_ext ~ [T_C, ethanol_pct, ratio_val], using ONLY the
real k_ext values from the physically valid Ext-Deg curves (R^2>=0.50) -- the same
population the Random Forest here is compared against, on an IDENTICAL
train_test_split(random_state=42). This is a different, smaller-sample comparison
than Table 3 (which trains the RF on the Monte Carlo-augmented dataset): classical
RSM/ANOVA requires genuine replicated experimental observations, not synthetic
pseudo-replicates, so no augmentation is used in this comparison.

disolvente / relacion are mapped back to their underlying continuous Box-Behnken
factors (ethanol_pct, ratio_val) so the 2nd-order polynomial is meaningful.

Interpretation (documented per the brief, not enforced): if RF outperforms RSM, that
is evidence of captured Arrhenius-type nonlinearity without a pre-specified polynomial
form; if they tie, the case for RF rests on flexibility without added specification
cost.

In practice, every compound here has only 2-7 physically valid curves -- fewer than
the 10 parameters the full 2nd-order model needs -- so the full RSM is NOT estimable.
Rather than silently forcing a fit, this script reports that outcome as-is and falls
back to a main-effects-only linear RSM (4 parameters) so at least one honest, estimable
comparison is produced (see RSM_model_type in Table 5 / 5b). The non-estimability of the
full quadratic model is itself documented in ANOMALIAS.md as a structural finding.
"""

import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from kinetic_pipeline import TARGETS_MAP, RANDOM_STATE, load_dataset, fit_valid_curves, build_rf_pipeline

RSM_FORMULA = ("k_ext ~ T_C + ethanol_pct + ratio_val "
               "+ I(T_C**2) + I(ethanol_pct**2) + I(ratio_val**2) "
               "+ T_C:ethanol_pct + T_C:ratio_val + ethanol_pct:ratio_val")
RSM_N_PARAMS = 10  # intercept + 3 linear + 3 quadratic + 3 interactions


def parse_ethanol_pct(disolvente: str) -> float:
    m = re.search(r'ETA(\d+)', disolvente)
    return float(m.group(1)) if m else 0.0


def parse_ratio_val(relacion: str) -> float:
    parts = relacion.split('_')
    return float(parts[1]) if len(parts) == 2 else np.nan


def build_valid_curve_frame(df, comp_es):
    sub_df = df[(df['compuesto'] == comp_es) & df['conc_promedio'].notna()].copy()
    if sub_df.empty:
        return pd.DataFrame()
    curvas_validas, _ = fit_valid_curves(sub_df)
    if not curvas_validas:
        return pd.DataFrame()
    rows = [{
        'id_cinetica': c['id_cinetica'], 'T_C': float(c['T_C']),
        'disolvente': c['disolvente'], 'relacion': c['relacion'],
        'ethanol_pct': parse_ethanol_pct(c['disolvente']),
        'ratio_val': parse_ratio_val(c['relacion']),
        'k_ext': c['k_ext'],
    } for c in curvas_validas]
    return pd.DataFrame(rows)


LINEAR_FORMULA = "k_ext ~ T_C + ethanol_pct + ratio_val"
LINEAR_N_PARAMS = 4  # intercept + 3 linear terms, main-effects-only fallback


def _fit_ols(train_df, formula, n_params):
    """Generic OLS fit guarded against non-identifiable designs (n <= n_params or
    non-positive residual degrees of freedom). Returns (model, anova_table) or (None, None)."""
    if len(train_df) <= n_params:
        return None, None
    model = smf.ols(formula, data=train_df).fit()
    if model.df_resid <= 0:
        return None, None
    return model, anova_lm(model, typ=2)


def evaluate_rsm_rf(df):
    """For each compound: attempt the full 2nd-order Box-Behnken RSM (9 terms + intercept,
    as specified by the brief) on the REAL valid-curve k_ext values only. If -- as actually
    happens here, since only 2-7 curves survive the R^2>=0.50 physical filter per compound --
    the full model is not identifiable (n <= 10 parameters), fall back to a main-effects-only
    linear RSM (4 parameters) so at least one honest, estimable RSM comparison is reported.
    Both outcomes are recorded via RSM_model_type; see ANOMALIAS.md for the full account."""
    table_5_rows, anova_frames = [], []
    for comp_es, comp_en in TARGETS_MAP.items():
        curve_df = build_valid_curve_frame(df, comp_es)
        n_valid = len(curve_df)
        row = {'Compound': comp_en, 'n_valid_curves': n_valid, 'RSM_model_type': 'none',
               'RSM_R2': np.nan, 'RSM_RMSE': np.nan, 'RF_R2': np.nan, 'RF_RMSE': np.nan, 'note': ''}

        if n_valid < 4:
            row['note'] = 'Too few valid curves for any meaningful train/test split (n<4)'
            table_5_rows.append(row)
            continue

        train_df, test_df = train_test_split(curve_df, test_size=0.25, random_state=RANDOM_STATE)
        if len(test_df) < 2:
            row['note'] = f'Test split too small (n_test={len(test_df)}) for a well-defined R2'
            table_5_rows.append(row)
            continue

        model, anova_table, model_type = None, None, 'none'
        try:
            model, anova_table = _fit_ols(train_df, RSM_FORMULA, RSM_N_PARAMS)
            model_type = 'quadratic_full' if model is not None else 'none'
        except Exception as e:
            row['note'] += f'Full RSM fit failed: {e}. '

        if model is None:
            row['note'] += (f'Full 2nd-order RSM not estimable: n_valid={n_valid} <= '
                             f'{RSM_N_PARAMS} parameters -- falling back to a linear main-effects RSM. ')
            try:
                model, anova_table = _fit_ols(train_df, LINEAR_FORMULA, LINEAR_N_PARAMS)
                model_type = 'linear_fallback' if model is not None else 'none'
            except Exception as e:
                row['note'] += f'Linear fallback RSM fit failed: {e}. '
            if model is None:
                row['note'] += f'Linear fallback also not estimable: n_valid={n_valid} <= {LINEAR_N_PARAMS} parameters.'

        row['RSM_model_type'] = model_type
        if model is not None:
            y_pred_rsm = model.predict(test_df)
            row['RSM_R2'] = r2_score(test_df['k_ext'], y_pred_rsm)
            row['RSM_RMSE'] = mean_squared_error(test_df['k_ext'], y_pred_rsm) ** 0.5
            anova_table = anova_table.reset_index().rename(columns={'index': 'term'})
            anova_table.insert(0, 'Compound', comp_en)
            anova_table.insert(1, 'RSM_model_type', model_type)
            anova_frames.append(anova_table)

        try:
            rf_pipe = build_rf_pipeline(random_state=RANDOM_STATE)
            rf_pipe.fit(train_df[['T_C', 'disolvente', 'relacion']], train_df['k_ext'])
            y_pred_rf = rf_pipe.predict(test_df[['T_C', 'disolvente', 'relacion']])
            row['RF_R2'] = r2_score(test_df['k_ext'], y_pred_rf)
            row['RF_RMSE'] = mean_squared_error(test_df['k_ext'], y_pred_rf) ** 0.5
        except Exception as e:
            row['note'] += f'RF on valid-curve subset failed: {e}'

        table_5_rows.append(row)

    table_5 = pd.DataFrame(table_5_rows)
    anova_all = pd.concat(anova_frames, ignore_index=True) if anova_frames else pd.DataFrame()
    return table_5, anova_all


def generate_rsm_vs_rf(data_path: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    df = load_dataset(data_path)
    table_5, anova_all = evaluate_rsm_rf(df)

    t5_path = os.path.join(results_dir, 'table_5_rsm_vs_rf.csv')
    anova_path = os.path.join(results_dir, 'table_5b_rsm_anova.csv')
    table_5.round(4).to_csv(t5_path, index=False)
    anova_all.round(4).to_csv(anova_path, index=False)

    print("=" * 80)
    print("TABLE 5: Classical RSM (2nd-order, Box-Behnken) vs. Random Forest")
    print("(same train/test split, valid curves only -- no MC augmentation)")
    print("=" * 80)
    print(table_5.round(4).to_string(index=False))
    print(f"[Saved to {t5_path}]\n")

    print("=" * 80)
    print("TABLE 5b: RSM ANOVA (quadratic_full where estimable, else linear_fallback -- see RSM_model_type)")
    print("=" * 80)
    print(anova_all.round(4).to_string(index=False) if not anova_all.empty
          else "[No compound had enough valid curves to estimate even the linear fallback RSM]")
    print(f"[Saved to {anova_path}]\n")


if __name__ == "__main__":
    generate_rsm_vs_rf(os.path.join('..', 'data', 'dataset.csv'), os.path.join('..', 'results'))
