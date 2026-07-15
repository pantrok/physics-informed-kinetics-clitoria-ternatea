# Captions borrador — Fig. 1 a Fig. 7 (esquema final del manuscrito)

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
