# README_tables.md — Index of manuscript-ready CSV tables

Each CSV below is the single source of truth for its corresponding manuscript table. All R2 values are out-of-condition (grouped cross-validation) unless the column name says otherwise; k_ext in min^-1.


## `paper_table_1_kinetic_fit_summary.csv`

**Manuscript table:** phenomenological Ext-Deg fit summary.
Columns: `Compound`, `R2_all_curves` (mean R2 of the Ext-Deg fit over all 15 kinetic curves), `n_valid_curves` (curves passing the R2>=0.50 physical validity filter), `R2_valid_curves_only` (mean R2 restricted to the valid subset).


## `paper_table_2_kext_confidence_intervals.csv`

**Manuscript table:** k_ext estimate per operating condition, with 95% CI.
Columns: `Compound`, `T_C`, `solvent`, `ratio`, `n_synthetic`, `k_ext_mean`, `k_ext_std`, `CI95_lower`, `CI95_upper`, `CI95_width`.


## `paper_table_3_out_of_condition_validation.csv`

**Manuscript table:** in-condition vs. out-of-condition predictive performance, with baselines.
Columns: `Compound`, `n_conditions`, `R2_in_condition_random_split`, `R2_out_of_condition_grouped`, `RMSE_out_of_condition_grouped`, `delta_R2`, `R2_baseline_global_mean`, `R2_baseline_nearest_condition`, `note`.


## `paper_table_4_replicate_measurement_cv.csv`

**Manuscript table:** absorbance replicate (triplicate) measurement CV by compound.
Columns: `Compound`, `n_obs`, `A_mean_min/max`, `CV_median_pct`, `CV_mean_pct`, `CV_P25_pct`, `CV_P75_pct`, `CV_of_mean_of_3_median_pct`, `CV_of_mean_of_3_mean_pct`. Anthocyanins shows markedly higher CV (~27%), consistent with absorbance near the detection limit -- an honest limitation, not a measurement error.


## `paper_table_5_gp_uncertainty.csv`

**Manuscript table:** Gaussian Process predictive mean, std and 95% CI for k_ext (Phenolics), over the 3 measured temperatures x 9 solvent/ratio combinations.
Columns: `T_C`, `solvent`, `ratio`, `k_ext_GP_mean`, `k_ext_GP_std`, `CI95_low`, `CI95_high`.


## `paper_table_supp_nofilter.csv`

**Supplementary table:** out-of-condition validation with vs. without the R2>=0.50 physical validity filter.
Columns: `Compound`, `R2_grouped_with_filter`, `R2_grouped_no_filter`, `n_curves_with`, `n_curves_no`. Shows that removing the filter does not restore generalization, confirming the limitation is structural (design size), not a filter artifact.
