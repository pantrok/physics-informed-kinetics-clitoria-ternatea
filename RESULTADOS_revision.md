# RESULTADOS_revision.md — Respuesta a la revisión por pares (pre-envío)

Consolida, para la fase de redacción, la respuesta a los dos gaps sustantivos identificados por la
revisión simulada (Editor + 3 revisores + Devil's Advocate → Major Revision, sin hallazgos críticos) y
los ítems de reporte menores. `random_state=42` fijo; todo bajo LeaveOneGroupOut (LOGO) agrupado por
`id_cinetica`. No se redactan aquí secciones del paper — solo veredicto y números.

---

## 1. Distribución por fold de la validación LOGO (Tarea 9A)

### 1.1 Esquema exacto (texto listo para Métodos)

La validación agrupada usa **LeaveOneGroupOut exhaustivo**, agrupado por `id_cinetica`: cada condición
operativa actúa como fold de prueba exactamente una vez, y ninguna réplica sintética Monte Carlo de la
condición retenida aparece en el entrenamiento de ese fold (ni en el ajuste del RF ni en la búsqueda
interna de hiperparámetros, que también se agrupa). Número de folds = número de condiciones válidas por
compuesto: Anthocyanins 6, Phenolics 7, ABTS 3, DPPH 2.

### 1.2 Resumen por compuesto

| Compuesto | n folds | R² medio | R² mediana | R² mín | R² máx | R² IQR | R² std |
|---|---|---|---|---|---|---|---|
| Anthocyanins | 6 | -14.69 | -7.61 | -42.28 | **-2.74** | 15.35 | 15.48 |
| Phenolics | 7 | -786.37 | -11.59 | -5302.69 | **-0.69** | 91.28 | 1992.45 |
| Antioxidant Capacity (ABTS) | 3 | -82.12 | -12.17 | -224.05 | **-10.14** | 106.96 | 122.92 |
| Antioxidant Capacity (DPPH) | 2 | -225.87 | -225.87 | -446.31 | **-5.44** | 220.43 | 311.74 |

### 1.3 Nota de honestidad — atemperar el lenguaje de cifras exactas

**La dispersión entre folds es muy amplia**, tal como anticipaba el brief. La media y la desviación
estándar por fold son estadísticos **inestables e inadecuados para citar como cifra única**: un solo
fold con varianza de test casi nula infla el R² negativo a magnitudes absurdas (p. ej. Phenolics
R²=-5302 en un fold), dominando la media. **No se recomienda citar R²=-0.423 (el valor agregado de la
Tabla 3/7) ni el R² medio por fold como una estimación puntual precisa** — ambos son sensibles a folds
individuales atípicos.

**El estadístico robusto y honesto a citar es cualitativo, no la cifra media:** de los 6+7+3+2 = 18
folds evaluados en total, a través de los 4 compuestos, **ni un solo fold —ni siquiera el mejor de cada
compuesto— alcanzó R² positivo** (el máximo por compuesto va de -0.69 a -10.14). Esto es evidencia
robusta e insensible a outliers: la recomendación de los revisores de atemperar el lenguaje a **"colapso
a desempeño no positivo en todas las condiciones evaluadas"** es la formulación correcta, en vez de citar
la cifra puntual -0.580 como si fuera estable.

Ver `results/paper_table_validation_folds.csv` (resumen) y `results/validation_fold_detail.csv` (detalle
por fold, con la condición retenida). Representación visual en Fig. 5 (puntos negros = folds dentro de
escala; triángulos = folds fuera de escala con su valor real anotado, ver §6).

---

## 2. Comparador estructurado tipo Arrhenius (Tarea 9B)

### 2.1 Modelo

`k_ext(T, solvente, ratio) = A · exp(−Ea/(R·T_K)) · (1 + b₁·fracción_etanol + b₂·ratio)`, 4 parámetros
(A, Ea, b₁, b₂), ajustado por mínimos cuadrados no lineales con cotas físicas (A>0, 0≤Ea≤300 kJ/mol),
sobre las curvas reales válidas, **sin aumentación Monte Carlo** (el modelo no la necesita: 4 parámetros
son identificables con muchas menos condiciones que el RF).

### 2.2 Comparación de los tres enfoques bajo LOGO

| Compuesto | n condiciones | R² Arrhenius (LOGO) | R² RF (LOGO) | R² GP (LOGO) |
|---|---|---|---|---|
| Anthocyanins | 6 | -0.653 | -0.580 | N/A (GP solo Fenoles) |
| Phenolics | 7 | -1.285 | -0.423 | +0.028 |
| Antioxidant Capacity (ABTS) | 3 | No estimable (n≤5) | -1.054 | N/A |
| Antioxidant Capacity (DPPH) | 2 | No estimable (n≤5) | -2.825 | N/A |

### 2.3 Veredicto explícito

**El modelo estructurado (Arrhenius) TAMBIÉN falla fuera de condición** — de hecho, en los dos casos
donde es estimable (Anthocyanins, Phenolics), su R² es **igual o peor** que el del RF (Phenolics:
-1.285 vs. -0.423). Para ABTS y DPPH, ni siquiera es estimable (n≤5 condiciones, insuficientes para 4
parámetros más un fold de prueba) — converge exactamente con el resultado de no-estimabilidad del RSM
(Tarea 3).

**Esto refuerza fuertemente la conclusión central del manuscrito:** ni un modelo flexible (RF, GP) ni un
modelo de solo 4 parámetros con física embebida (Arrhenius) generalizan fuera de condición con este
diseño Box-Behnken de 15 corridas tras el filtro físico. Elimina la objeción "solo demostraron que el ML
flexible falla" — el resultado no depende de la flexibilidad del modelo, sino del tamaño/estructura del
diseño experimental. **No se requiere revisar el framing del manuscrito** (el escenario "el estructurado
funciona" no se dio).

Ver `results/paper_table_structured_comparator.csv`.

---

## 3. Hiperparámetros de RF y GP, y estabilidad (Tarea 9C.1)

**RF:** grid `n_estimators∈{50,100,200}`, `max_depth∈{None,5,10}`, `min_samples_split∈{2,5}`
(`GridSearchCV`, `cv=3`, `scoring='r2'`), afinado en cada fold sobre entrenamiento agrupado. Valores
seleccionados por compuesto documentados en `results/model_hyperparameters.md`.

**GP:** kernel `ConstantKernel() × RBF(length_scale=[1,1,1]) + WhiteKernel()`, anisotrópico sobre
`[T, %etanol, ratio]` estandarizados; hiperparámetros optimizados por máxima verosimilitud marginal
(`GaussianProcessRegressor`, L-BFGS-B, `n_restarts_optimizer=5`) — no elegidos a mano.

**Barrido de estabilidad del RF bajo LOGO** (n_estimators∈{50,100,200,300}, max_depth∈{None,5,10,15},
un RF fijo por valor, sin afinar): el R² agrupado **se mantiene negativo en todo el barrido, para los 4
compuestos** (rangos: Anthocyanins [-0.583,-0.575]; Phenolics [-0.451,-0.395]; ABTS [-1.102,-1.019];
DPPH [-2.912,-2.828]). El resultado negativo **no es un artefacto de la malla de búsqueda de
hiperparámetros** — análogo al barrido de σ de la Tarea 6A.

Ver `results/model_hyperparameters.md`, `results/rf_hyperparameter_sweep_logo.csv`.

---

## 4. Sensibilidad al umbral de validez física (Tarea 9C.2)

Barrido del umbral R²≥{0.40, 0.50, 0.60} sobre el filtro fenomenológico. El R² agrupado **se mantiene
negativo en los 4 compuestos, en los 3 umbrales**, sin excepción (rango completo: -0.28 a -2.82). El
número de condiciones válidas (n) cambia moderadamente con el umbral (p. ej. Phenolics: 8→7→5
condiciones de 0.40→0.50→0.60), pero **la conclusión cualitativa no depende del corte exacto** — el
umbral de 0.50 usado en el resto del manuscrito no está "eligiendo" artificialmente el resultado.

Ver `results/threshold_sensitivity.csv`.

---

## 5. Correlación k_ext / k_deg (Tarea 9C.3)

Para las curvas retenidas (R²≥0.50), la correlación entre `k_ext` y `k_deg` estimados (de la matriz de
covarianza `pcov` del ajuste `curve_fit`) es **consistentemente fuerte y negativa** en las 18 curvas
válidas: rango aproximado -0.49 a -1.00, con medias por compuesto de aproximadamente -0.84
(Anthocyanins), -0.65 (Phenolics), -0.78 (ABTS), -0.58 (DPPH).

**Esto indica no-identificabilidad conjunta sustancial entre `k_ext` y `k_deg`** en el modelo Ext-Deg: el
ajuste puede compensar un `k_ext` más alto con un `k_deg` más bajo (o viceversa) sin degradar
notablemente el R² de la curva. `k_deg` en particular muestra intervalos de confianza muy anchos en
varias curvas (p. ej. Phenolics id_cinetica=7: `k_deg`=0.0058 con CI95 **[-82.29, 82.30]** — un caso
extremo de no-identificabilidad práctica para ese parámetro específico). `k_ext` es comparativamente más
estable en la mayoría de las curvas, pero el hallazgo debe declararse como limitación honesta: **los
`k_ext` reportados no están completamente libres de contaminación por el acoplamiento con `k_deg`.**

Ver `results/kinetic_parameter_correlation.csv` (detalle por curva).

---

## 6. Figuras renumeradas (7 ilustraciones, límite de la revista)

La figura de EDA (pairplot) se retiró del manuscrito (se conserva en el repo como
`figures/fig_supp_eda_pairplot.*`, fuera del conteo). Esquema final:

| Figura | Archivo | Contenido |
|---|---|---|
| Fig. 1 | `fig_1.tiff` | Diagrama de metodología |
| Fig. 2 | `fig_2_phenomenological_fits.tiff` | Ajustes fenomenológicos Ext-Deg |
| Fig. 3 | `fig_3_kext_confidence_intervals.tiff` | k_ext por condición con IC 95% |
| Fig. 4 | `fig_4_response_surfaces.tiff` | Superficies de respuesta (predicción del modelo) |
| Fig. 5 | `fig_5_out_of_condition_validation.tiff` | **Validación in- vs. out-of-condition, RF + Arrhenius + GP, con dispersión por fold** |
| Fig. 6 | `fig_6_augmentation_and_sigma.tiff` | Efecto de aumentación y sensibilidad a σ |
| Fig. 7 | `fig_7_gp_uncertainty_map.tiff` | Mapa de incertidumbre del GP |

Captions completos en [figures/CAPTIONS_borrador.md](figures/CAPTIONS_borrador.md); verificación de
formato (TIFF 600dpi, ≤180×200mm, itálicas IUPAC) en [figures/FORMAT_check.md](figures/FORMAT_check.md).

**Nota sobre Fig. 5:** el eje Y se recortó a [-3.5, 1.3] para mantener legibles las barras y la mayoría de
los folds; los folds fuera de ese rango (n=14 de 18 totales) se marcan con un triángulo en el borde
inferior y su valor real se anota en texto — ningún dato se oculta, solo se re-escala visualmente (ver
§1.3 sobre por qué esos valores extremos existen).

---

## 7. Resultados inesperados (documentados sin forzar)

1. **La dispersión entre folds es mucho más extrema de lo que la cifra agregada de la Tabla 3/7 sugiere**
   (folds individuales con R² de hasta -5302). No invalida la conclusión — al contrario, la refuerza,
   porque incluso el fold MENOS malo de cada compuesto sigue siendo negativo — pero exige el cambio de
   lenguaje descrito en §1.3.
2. **El comparador estructurado no solo falla, sino que es peor que el RF** donde ambos son estimables
   (particularmente en Phenolics: -1.285 vs. -0.423). No se esperaba que el modelo con física embebida
   fuera *peor*, no solo "también malo".
3. **La correlación k_ext/k_deg es sistemáticamente fuerte (mediana ~-0.7 a -0.9)**, con al menos un caso
   de no-identificabilidad práctica extrema (CI95 de k_deg mayor a ±80). Esto no se había cuantificado
   antes de esta tarea y es material directo para la sección de limitaciones.

---

*Ver también [RESULTADOS_validacion_agrupada.md](RESULTADOS_validacion_agrupada.md) (Tarea 5),
[RESULTADOS_sigma_y_fisica.md](RESULTADOS_sigma_y_fisica.md) (Tarea 6), y
[RESULTADOS_para_manuscrito.md](RESULTADOS_para_manuscrito.md) (Tarea 7, índice de figuras/tablas
superseded en numeración por este documento — ver §6 arriba para el esquema vigente).*
