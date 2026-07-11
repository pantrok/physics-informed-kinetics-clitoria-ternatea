# RESULTADOS_resumen.md — Cifras clave para redacción del manuscrito

Fuente única de verdad para todas las cifras citables del eje physics-informed / QbD-PAT.
Todas las cifras provienen de los CSV en `/results` generados por `src/script_5` … `src/script_8`,
con `random_state=42` fijo. R² con 3 decimales, RMSE con 3-4 decimales (según orden de magnitud),
k_ext en min⁻¹.

**IMPORTANTE — qué reemplaza a qué:** las cifras de este documento son las que van al paper.
Las Tablas 1 (inferencia instantánea de absorbancia), los parity plots de absorbancia y el
feature-importance de absorbancia de la versión anterior del manuscrito (generados por los antiguos
`script_2`, `script_3`, `script_4`) quedan **eliminados** — no son parte del eje reenfocado y no deben
citarse. Ningún número de este documento reemplaza esos valores porque esos valores ya no forman parte
del paper.

---

## 1. Conteo de curvas cinéticas válidas (filtro físico R²≥0.50)

Fuente: `results/table_2_phenomenological_fit.csv`

| Compuesto | Curvas válidas (n) | Ext-Deg R² (las 15, promedio) | Ext-Deg R² (solo válidas, promedio) |
|---|---|---|---|
| Anthocyanins | 6 | 0.376 | 0.770 |
| Phenolics | 7 | 0.441 | 0.706 |
| Antioxidant Capacity (ABTS) | 3 | 0.272 | 0.751 |
| Antioxidant Capacity (DPPH) | 2 | 0.233 | 0.719 |

**Coincide con la Tabla 2 original (6/7/3/2)** — el conteo de curvas válidas no cambió al reforzar el análisis.

## 2. Métricas finales del RF de k_ext, dataset aumentado (Tabla 3)

Fuente: `results/table_3_augmented_ml_prediction.csv`. RF entrenado sobre el dataset Monte Carlo
physics-informed (n_synth=30 por curva, filtro de varianza ±60%), `train_test_split(test_size=0.25, random_state=42)`.

| Compuesto | R² (test) | RMSE (test, min⁻¹) |
|---|---|---|
| Anthocyanins | 0.944 | 0.0049 |
| Phenolics | 0.903 | 0.0169 |
| Antioxidant Capacity (ABTS) | 0.914 | 0.0073 |
| Antioxidant Capacity (DPPH) | 0.402 | 0.0489 |

## 3. Intervalos de confianza 95% de k_ext por condición operativa (Tabla 4)

Fuente completa: `results/table_4_kext_confidence_intervals.csv` (17 condiciones operativas, agregando
los k_ext sintéticos aceptados del Monte Carlo). Ejemplos representativos (ver CSV para las 17 filas completas):

| Compuesto | T (°C) | Disolvente \| Ratio | k_ext medio | IC 95% | Ancho IC |
|---|---|---|---|---|---|
| Anthocyanins | 75.0 | Agua100 \| 1_20 | 0.0837 | [0.0648, 0.1073] | 0.0425 |
| Anthocyanins | 60.0 | Agua75_ETA25 \| 1_30 | 0.0384 | [0.0303, 0.0516] | 0.0213 |
| Phenolics | 75.0 | Agua75_ETA25 \| 1_10 | 0.1589 | [0.1089, 0.2545] | 0.1455 |
| Antioxidant Capacity (ABTS) | 67.5 | Agua100 \| 1_10 | 0.0726 | [0.0539, 0.1048] | 0.0510 |
| Antioxidant Capacity (DPPH) | 75.0 | Agua50_ETA50 \| 1_20 | 0.2281 | [0.1500, 0.3943] | 0.2443 |

Los IC de DPPH son notablemente más anchos (ver [ANOMALIAS.md](ANOMALIAS.md) §1) — consistente con solo 2 curvas válidas.

## 4. RSM clásico (Box-Behnken) vs. RF — misma partición, solo curvas reales (Tabla 5)

Fuente: `results/table_5_rsm_vs_rf.csv`, `results/table_5b_rsm_anova.csv`.
**Hallazgo principal: el RSM cuadrático completo (10 parámetros) no fue estimable para NINGÚN compuesto**
(2-7 curvas válidas disponibles). Ver [ANOMALIAS.md](ANOMALIAS.md) §3 para el análisis completo.

| Compuesto | n válidas | Modelo RSM ajustado | RSM R² (test) | RF R² (test, mismos datos) |
|---|---|---|---|---|
| Anthocyanins | 6 | Ninguno (ni cuadrático ni lineal estimable) | — | -0.823 |
| Phenolics | 7 | Lineal (fallback, 1 gdl residual) | 0.545 | -0.213 |
| Antioxidant Capacity (ABTS) | 3 | Ninguno (n<4) | — | — |
| Antioxidant Capacity (DPPH) | 2 | Ninguno (n<4) | — | — |

ANOVA del único modelo estimable (Fenoles, lineal): ningún término significativo a p<0.05
(T_C: p=0.379; ethanol_pct: p=0.213; ratio_val: p=0.356; 1 gdl residual).

**Nota:** esta comparación usa solo las curvas reales (sin aumentación Monte Carlo), por diseño metodológico
(un RSM/ANOVA clásico requiere réplicas experimentales genuinas). No debe confundirse con la Tabla 3, donde
el RF sí se beneficia de la aumentación physics-informed y alcanza R²=0.90-0.94.

## 5. Filtro de varianza ±60% — tasas de aceptación/rechazo (Tabla 6)

Fuente: `results/table_6b_variance_filter_audit_summary.csv` (resumen), `results/table_6_variance_filter_audit.csv` (detalle por condición).

| Compuesto | Intentados | Aceptados | Rechazados | Tasa de rechazo |
|---|---|---|---|---|
| Anthocyanins | 180 | 165 | 15 | 8.3% |
| Phenolics | 210 | 182 | 28 | 13.3% |
| Antioxidant Capacity (ABTS) | 90 | 86 | 4 | 4.4% |
| Antioxidant Capacity (DPPH) | 60 | 43 | 17 | 28.3% |

## 6. Curva de ablación — R² vs. tamaño del dataset aumentado (n_synth por curva)

Fuente: `results/table_ablation_r2_vs_augmentation_size.csv`, `figures/fig_8_augmentation_ablation.png`.

| Compuesto | n_synth=0 (R²) | n_synth=10 | n_synth=30 (usado en Tabla 3) | n_synth=60 | n_synth=90 |
|---|---|---|---|---|---|
| Anthocyanins | **-1.324** | 0.947 | 0.944 | 0.933 | 0.929 |
| Phenolics | **-0.133** | 0.618 | 0.903 | 0.791 | 0.808 |
| Antioxidant Capacity (ABTS) | N/A (n=3) | 0.860 | 0.914 | 0.892 | 0.855 |
| Antioxidant Capacity (DPPH) | N/A (n=2) | 0.807 | 0.402 | 0.893 | 0.690 |

**Meseta:** Anthocyanins y ABTS se estabilizan ya en n_synth=10-30, sin ganancia adicional al aumentar más.
Phenolics alcanza su máximo en n_synth=30 (el valor usado en Tabla 3). DPPH es ruidoso en todo el rango
(ver ANOMALIAS.md §1) y no muestra una meseta clara — coherente con su base de solo 2 curvas reales.

---

*Para el relato completo de los hallazgos inesperados (DPPH, superficies escalonadas del RF, no-estimabilidad
del RSM completo, tasa de rechazo del filtro), ver [ANOMALIAS.md](ANOMALIAS.md).*
*Para los captions borrador de Fig. 7 y Fig. 8, ver `figures/CAPTIONS_borrador.md`.*
