# Captions borrador — Fig. 2 a Fig. 8 (esquema final)

Borrador crudo, terminología limpia de paper; se pule en la fase de redacción con el autor.
Fig. 1 está reservada para el diagrama de metodología (se hace aparte, no aquí).

---

## Fig. 2 — `fig_2_eda_pairplot.png`

> Fig. 2. Exploratory data analysis: bivariate relationships among operational extraction variables
> and target compound concentrations, colored by target compound.

---

## Fig. 3 — `fig_3_phenomenological_fits.png`

> Fig. 3. Phenomenological Ext-Deg model fits (concentration vs. time) for the physically valid
> extraction kinetics (R²≥0.50), one panel per target compound. Points are observed concentrations;
> lines are the fitted extraction-degradation model for each operating condition.

---

## Fig. 4 — `fig_4_kext_confidence_intervals.png`

> Fig. 4. Estimated extraction rate constant (k_ext) by operating condition, with 95% confidence
> intervals derived from physics-informed Monte Carlo augmentation, one panel per target compound.

---

## Fig. 5 — `fig_5_response_surfaces.png`

> Fig. 5. Model-predicted k_ext response surfaces across the measured design space, one panel per
> target compound. Heatmaps show the Random Forest's prediction over a dense temperature grid crossed
> with the 9 solvent × solid:liquid ratio combinations of the design; white-ringed markers indicate
> real experimental points. All values are interpolation strictly within the measured domain
> (T: 60-75 °C) — no extrapolation was performed.

**Nota para el autor:** el patrón "escalonado" (no una superficie suave) refleja las particiones
piecewise-constant del Random Forest, no un error de graficado — discutir en Resultados/Discusión.

---

## Fig. 6 — `fig_6_out_of_condition_validation.png`

> Fig. 6. Predictive performance under in-condition (random split) versus out-of-condition (grouped
> cross-validation by operating condition) evaluation, by target compound. The reduced Box-Behnken
> design does not support generalization to unseen operating conditions.

**Nota para el autor:** esta es la figura central del hallazgo metodológico. El contraste debe
presentarse como una evaluación deliberada, no como el descubrimiento de un error.

---

## Fig. 7 — `fig_7_augmentation_and_sigma.png`

> Fig. 7. Effect of Monte Carlo augmentation size (left) and augmentation noise level σ (right) on
> out-of-condition predictive performance, by target compound. Neither factor restores generalization
> to unseen operating conditions; the dashed line marks σ=0.08, the value adopted for the main analysis.

---

## Fig. 8 — `fig_8_gp_uncertainty_map.png`

> Fig. 8. Gaussian Process predictive uncertainty (standard deviation) for k_ext across the measured
> design space (Phenolics). High-uncertainty regions indicate operating conditions not informed by the
> current Box-Behnken design; experimental points overlaid (cyan markers). This uncertainty map
> motivates active learning / Bayesian optimization as a future sampling strategy.

**Nota para el autor:** GP entrenado solo con las curvas reales válidas de Fenoles (sin aumentación);
desempeño agrupado limitado (R²≈0.03) — el valor de esta figura es la incertidumbre, no la precisión
predictiva.
