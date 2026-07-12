# RESULTADOS_para_manuscrito.md — Índice final para la fase de redacción

Consolida figuras, tablas y números clave, con terminología limpia de paper, listos para escribir el
manuscrito. No redacta secciones del texto — eso se trabaja con el autor, turno a turno.

**Framing acordado (una línea):** modelado fenomenológico grey (Ext-Deg) con cuantificación de
incertidumbre para cinéticas de extracción de *Clitoria ternatea*, más un análisis de validación
riguroso que demuestra —por tres vías independientes (RSM clásico no estimable, predicción ML no
generaliza fuera de condición, la aumentación física no supera a un control de ruido simple— que un
diseño Box-Behnken de 15 corridas, tras el filtro físico, no sostiene predicción fuera de muestra; se
propone active learning / optimización bayesiana como ruta futura, fundamentada en el mapa de
incertidumbre del Gaussian Process.

---

## 1. Figuras finales (Fig. 2 – Fig. 8)

Fig. 1 queda reservada para el diagrama de metodología (aparte, no generado por este código).

| Figura | Archivo | Caption limpio |
|---|---|---|
| Fig. 2 | `fig_2_eda_pairplot.png` | Exploratory data analysis: bivariate relationships among operational extraction variables and target compound concentrations, colored by target compound. |
| Fig. 3 | `fig_3_phenomenological_fits.png` | Phenomenological Ext-Deg model fits (concentration vs. time) for the physically valid extraction kinetics (R²≥0.50), one panel per target compound. |
| Fig. 4 | `fig_4_kext_confidence_intervals.png` | Estimated extraction rate constant (k_ext) by operating condition, with 95% confidence intervals derived from physics-informed Monte Carlo augmentation, one panel per target compound. |
| Fig. 5 | `fig_5_response_surfaces.png` | Model-predicted k_ext response surfaces across the measured design space, one panel per compound; interpolation strictly within the measured domain (T: 60–75 °C), never extrapolated. |
| Fig. 6 | `fig_6_out_of_condition_validation.png` | Predictive performance under in-condition (random split) vs. out-of-condition (grouped cross-validation) evaluation, by compound — the central methodological finding. |
| Fig. 7 | `fig_7_augmentation_and_sigma.png` | Effect of Monte Carlo augmentation size and augmentation noise level (σ) on out-of-condition predictive performance; neither restores generalization. |
| Fig. 8 | `fig_8_gp_uncertainty_map.png` | Gaussian Process predictive uncertainty (standard deviation) for k_ext across the measured design space (Phenolics); motivates active learning as future work. |

Captions completos y notas para el autor en [figures/CAPTIONS_borrador.md](figures/CAPTIONS_borrador.md).

## 2. Tablas finales (paper_table_1 – 5 + soporte)

| Tabla | Archivo | Qué muestra |
|---|---|---|
| Tabla 1 | `results/paper_table_1_kinetic_fit_summary.csv` | Resumen del ajuste fenomenológico Ext-Deg: n curvas válidas y R² por compuesto. |
| Tabla 2 | `results/paper_table_2_kext_confidence_intervals.csv` | k_ext por condición operativa, con IC 95%. |
| Tabla 3 | `results/paper_table_3_out_of_condition_validation.csv` | R² in-condition vs. out-of-condition (agrupado), con baselines triviales — hallazgo central. |
| Tabla 4 | `results/paper_table_4_replicate_measurement_cv.csv` | CV de medición por compuesto (triplicado de absorbancia), con nota de Antocianinas cerca del LOD. |
| Tabla 5 | `results/paper_table_5_gp_uncertainty.csv` | Incertidumbre del Gaussian Process (media, std, IC95) para k_ext, Fenoles. |
| Soporte | `results/paper_table_supp_nofilter.csv` | Validación con vs. sin filtro físico — confirma que la limitación es del diseño, no del filtro. |

Descripción completa de columnas en [results/README_tables.md](results/README_tables.md).

## 3. Números clave

### 3.1 Hallazgo central — in-condition vs. out-of-condition (Fig. 6 / Tabla 3)

| Compuesto | n condiciones | R² in-condition | R² out-of-condition | Δ R² |
|---|---|---|---|---|
| Anthocyanins | 6 | 0.944 | **-0.580** | -1.524 |
| Phenolics | 7 | 0.903 | **-0.423** | -1.326 |
| Antioxidant Capacity (ABTS) | 3 | 0.914 | -1.054 | -1.969 (no concluyente, n<4) |
| Antioxidant Capacity (DPPH) | 2 | 0.402 | -2.825 | -3.227 (no concluyente, n<4) |

Los dos casos concluyentes (Anthocyanins, Phenolics, n≥4) muestran una caída dramática y consistente:
la predicción ML no generaliza a condiciones operativas no vistas.

### 3.2 Validación sin filtro físico (soporte, blindaje ante revisor)

| Compuesto | R² con filtro (n=6/7/3/2) | R² sin filtro (n=15 todas) |
|---|---|---|
| Anthocyanins | -0.580 | -1.736 (peor) |
| Phenolics | -0.423 | -0.079 (similar, sigue negativo) |
| Antioxidant Capacity (ABTS) | -1.054 | -2.286 (peor) |
| Antioxidant Capacity (DPPH) | -2.825 | **+0.554** (positivo — hallazgo honesto, no forzado) |

Quitar el filtro físico no rescata la generalización en 3 de 4 compuestos (confirma que la limitación
es del tamaño/diseño experimental, no del filtro). **DPPH es una excepción honesta:** al pasar de 2 a
15 curvas el R² se vuelve positivo — dado el salto tan grande en tamaño de muestra (no solo el filtro),
esto se reporta tal cual, sin forzar una narrativa única; se recomienda mencionarlo explícitamente como
una observación puntual, no generalizable a los otros compuestos.

### 3.3 Gaussian Process — Phenolics

- R² agrupado (out-of-condition): **0.028**, RMSE: 0.056 — desempeño predictivo limitado, como se
  esperaba; el valor de esta pieza es la cuantificación de incertidumbre, no la precisión.
- El mapa de incertidumbre (Fig. 8) identifica las combinaciones solvente×ratio con MAYOR incertidumbre
  predictiva — precisamente las condiciones sin punto experimental cercano (p. ej. las combinaciones con
  50% etanol, `Agua50_ETA50`, no muestreadas en el subconjunto válido de Fenoles) — que un muestreo de
  active learning priorizaría en un siguiente lote experimental.

## 4. Recordatorio de framing (para mantener consistencia al redactar)

> La contribución del manuscrito es metodológica: cuantificación de incertidumbre físicamente informada
> + un diagnóstico de validación honesto que muestra los límites de la generalización ML en un diseño
> Box-Behnken reducido, con una ruta futura de active learning fundamentada en el mapa de incertidumbre
> del GP. El resultado negativo de generalización NO se oculta — es el hallazgo central, presentado como
> evaluación metodológica deliberada (Fig. 6), no como bitácora de un error descubierto a posteriori.

---

*No se redactan aquí secciones del paper. Ver también [RESULTADOS_validacion_agrupada.md](RESULTADOS_validacion_agrupada.md),
[RESULTADOS_sigma_y_fisica.md](RESULTADOS_sigma_y_fisica.md) y [ANOMALIAS.md](ANOMALIAS.md) para el
detalle completo del proceso de validación detrás de estos números.*
