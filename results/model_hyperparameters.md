# model_hyperparameters.md — RF and GP hyperparameters (addendum, review response)

## Random Forest (k_ext regressor)

Search grid (`GridSearchCV`, `cv=3`, `scoring='r2'`), tuned on the training fold each time (random split for Table 3; grouped inner CV for every LOGO fold elsewhere):

- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 5, 10]
- `min_samples_split`: [2, 5]

Selected values (best_params_, full-data fit on the augmented dataset, per compound):

- **Anthocyanins**: `{'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 50}`
- **Phenolics**: `{'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 200}`
- **Antioxidant Capacity (ABTS)**: `{'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 50}`
- **Antioxidant Capacity (DPPH)**: `{'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 200}`

## Gaussian Process (k_ext regressor, Phenolics only)

Kernel: `ConstantKernel() * RBF(length_scale=[1,1,1]) + WhiteKernel()`, anisotropic over `[T, ethanol_pct, ratio_val]` (standardized). Hyperparameters (kernel amplitude, RBF length-scales, white-noise level) are optimized by maximizing the log marginal likelihood (`GaussianProcessRegressor` default, L-BFGS-B, `n_restarts_optimizer=5`) — not manually chosen.


## RF hyperparameter stability sweep under out-of-condition (LOGO) validation

Fixed (untuned) RF at each candidate value, evaluated under the same LeaveOneGroupOut scheme as the main result, to check the negative out-of-condition R2 is not an artifact of the specific GridSearchCV grid. Full data in `results/rf_hyperparameter_sweep_logo.csv`.

- **Anthocyanins**: R2 range across the sweep = [-0.583, -0.575] (stays negative throughout)
- **Phenolics**: R2 range across the sweep = [-0.451, -0.395] (stays negative throughout)
- **Antioxidant Capacity (ABTS)**: R2 range across the sweep = [-1.102, -1.019] (stays negative throughout)
- **Antioxidant Capacity (DPPH)**: R2 range across the sweep = [-2.912, -2.828] (stays negative throughout)