# ANOMALIAS.md — Log honesto de hallazgos inesperados

Este documento registra, sin "arreglar a la fuerza", los resultados que se desviaron de lo esperado durante
la ejecución de las Tareas 1-4 del refuerzo analítico. Son material legítimo para la Discusión del manuscrito.

---

## 0. Nota de redacción — terminología "banda" (addendum Tarea 6, §1)

Las columnas `A_band1/2/3` / `conc_band1/2/3` del dataset original **no son bandas espectrales ni longitudes
de onda distintas**: son **tres réplicas de medición de absorbancia** de la misma muestra (triplicado
instrumental, para capturar ruido de medición). Se renombraron a `A_replicate_1/2/3` / `conc_replicate_1/2/3`
(commit `refactor: renombrar A_band*/conc_band* a *_replicate_*`); el archivo original con los nombres viejos
se conserva en `data/dataset_original_backup.csv`.

**Verificar en el texto del manuscrito** que no se describan "tres bandas" ni "tres longitudes de onda" — el
diseño correcto a describir en Métodos es "triplicado de medición de absorbancia por instancia". Ver
`results/table_12_replicate_cv_by_compound.csv` y `RESULTADOS_sigma_y_fisica.md` para la cuantificación del
ruido de este triplicado (CV por compuesto).

## 6. DPPH se vuelve positivo al quitar el filtro físico (addendum Tarea 7, §2) — no forzar una narrativa única

La validación agrupada sin el filtro R²≥0.50 (`results/paper_table_supp_nofilter.csv`) confirma lo
esperado para 3 de 4 compuestos (quitar el filtro empeora o deja igual de negativo el R² agrupado:
Anthocyanins -0.580→-1.736, ABTS -1.054→-2.286, Phenolics -0.423→-0.079). **DPPH es la excepción:**
pasa de R²=-2.825 (2 curvas, con filtro) a R²=+0.554 (15 curvas, sin filtro) — el único resultado
positivo de toda la validación agrupada del proyecto.

**No se interpreta esto como "el filtro perjudicaba a DPPH"** — el salto de 2 a 15 curvas es un cambio
de tamaño de muestra mucho mayor que en los otros 3 compuestos (que solo ganan curvas adicionales de
peor ajuste fenomenológico, sin cambiar tan drásticamente el nº de grupos LOCO). Es más consistente con
"con solo 2 grupos, cualquier LOCO es extremadamente ruidoso e inestable" que con una conclusión sobre
la utilidad del filtro físico. Se registra tal cual, sin forzar una narrativa que lo generalice a los
otros compuestos ni que lo descarte como ruido — es una observación puntual y honesta a mencionar en la
Discusión si se habla de DPPH específicamente.

---

## 1. DPPH sigue sin datos suficientes para el mapeo ML (confirmado, límite conocido)

Solo **2 curvas cinéticas** de Flavonoides (DPPH) superan el filtro físico R²≥0.50 (`table_2_phenomenological_fit.csv`),
igual que en la Tabla 2 original. A diferencia de lo que hacía `script_5` original (mostrar panel "Insufficient data"
cuando `curvas_validas` estaba vacío), aquí SÍ hay 2 curvas, así que el pipeline `run_compound_pipeline` completa el
entrenamiento del RF — pero el resultado es débil y frágil:

- Tabla 3: R² = 0.402, RMSE = 0.049 (el más bajo de los 4 compuestos, muy por debajo de Antocianinas 0.944, Fenoles 0.903, ABTS 0.914).
- Tabla 4 (IC 95%): solo 2 condiciones operativas tienen k_ext sintético agregado, con anchos de IC muy grandes
  (0.2443 y 0.0321 min⁻¹), reflejando la falta de replicación real del diseño para este compuesto.
- Curva de ablación (Tabla ablation / Fig. 8): la línea de DPPH es la más ruidosa y NO monótona
  (R² sube a 0.807 en n=10, cae a 0.402 en n=30, sube a 0.893 en n=60, cae a 0.690 en n=90) — comportamiento
  errático consistente con memorizar variaciones de solo 2 condiciones reales en vez de aprender una tendencia física.
- Tabla 5 (RSM vs RF): DPPH tiene solo 2 curvas válidas, por debajo del mínimo (n≥4) para cualquier partición
  train/test razonable — ni RSM ni RF pudieron evaluarse de forma comparable.

**Conclusión honesta:** DPPH es, con los datos actuales, insuficiente para una conclusión ML robusta sobre k_ext.
Es un límite conocido del diseño experimental (Box-Behnken con 15 corridas), no un error del pipeline.

## 2. Las superficies de respuesta (Fig. 7) son escalonadas, no suaves

El Random Forest predice mediante particiones (árboles), por lo que la malla de interpolación T=60-75°C produce
un patrón de bandas/escalones discretos en vez de una superficie continua tipo Arrhenius. Esto es visualmente
correcto (es exactamente lo que hace un RF) pero **no es monotónico ni suave** como una superficie de respuesta
polinómica clásica. Se observa especialmente en Antocianinas y Fenoles, donde hay un salto abrupto de nivel de
`k_ext` entre ~71°C y ~72°C, en vez de un incremento gradual. Este comportamiento debe discutirse explícitamente
en el manuscrito: el RF captura relaciones no lineales pero al costo de una superficie no suave, mientras que un
RSM polinómico impondría suavidad por construcción (a costa de un posible sesgo de forma funcional).

## 3. El modelo RSM cuadrático completo (Tarea 3) NO es estimable para ningún compuesto

Este es el hallazgo más importante de la Tarea 3. El modelo RSM de 2º orden especificado en el brief
(3 términos lineales + 3 cuadráticos + 3 interacciones = 9 términos + intercepto = **10 parámetros**) requiere
más puntos de diseño independientes de los que sobreviven al filtro físico R²≥0.50:

| Compuesto | Curvas válidas (n) | RSM cuadrático completo | Fallback lineal (4 parámetros) |
|---|---|---|---|
| Antocianinas | 6 | No estimable (n≤10) | No estimable (train=4≤4) |
| Fenoles | 7 | No estimable (n≤10) | Estimable, pero con **1 solo grado de libertad residual** |
| ABTS | 3 | No estimable | No intentado (n<4, sin partición train/test viable) |
| DPPH | 2 | No estimable | No intentado (n<4) |

Solo Fenoles logró un ajuste (lineal, sin términos cuadráticos ni de interacción), y aun así con un ANOVA
extremadamente frágil (`table_5b_rsm_anova.csv`: Residual df=1). Ninguno de los términos (T_C, ethanol_pct,
ratio_val) resultó significativo a p<0.05 con esa muestra tan pequeña (p=0.379, 0.213, 0.356 respectivamente).

En la comparación RF vs RSM sobre la misma partición pequeña (sin aumentación), el RF tampoco generalizó bien
(R²=-0.821 en Antocianinas, R²=-0.213 en Fenoles) — ambos peor que una simple media, porque el conjunto de
entrenamiento real (4-5 muestras) es demasiado pequeño para cualquier modelo. **Esta comparación "en igualdad
de condiciones" (mismos datos reales, sin aumentación) no favorece a ninguno de los dos métodos: ambos fallan
sin suficientes datos reales.**

**Implicación para el manuscrito (argumento estructural, no forzado):** el diseño Box-Behnken de 15 corridas,
después del filtro de validez física, deja muy pocos puntos de diseño independientes (2-7 según el compuesto)
para identificar un RSM de 2º orden clásico. Esto es en sí mismo un argumento a favor de un enfoque no paramétrico
alimentado por aumentación Monte Carlo physics-informed (el RF de la Tabla 3, entrenado sobre el dataset aumentado,
sí alcanza R²=0.90-0.94 para 3 de 4 compuestos) — no porque el RF sea intrínsecamente superior a un RSM en igualdad
de condiciones, sino porque el enfoque aumentado tolera una malla de diseño reducida y no ortogonal mejor que un
polinomio de forma fija que necesita más grados de libertad de los que el filtro físico deja disponibles.

## 4. El filtro de varianza ±60% sí discrimina (ni permisivo ni excesivamente estricto)

Tasas de rechazo por compuesto (`table_6b_variance_filter_audit_summary.csv`):

| Compuesto | Tasa de aceptación | Tasa de rechazo |
|---|---|---|
| ABTS | 95.6% | 4.4% |
| Antocianinas | 91.7% | 8.3% |
| Fenoles | 86.7% | 13.3% |
| DPPH | 71.7% | 28.3% |

Ninguna tasa de rechazo es cercana a 0% (el filtro sí descarta artefactos) ni cercana a 100% (el filtro no es tan
estricto que invalide la aumentación). DPPH tiene la tasa de rechazo más alta (28.3%), coherente con su fit
fenomenológico más inestable (solo 2 curvas válidas, condiciones extremas T=75°C). Una condición particular
(DPPH, id_cinetica=6, T=75°C, Agua50_ETA50, 1_20) tuvo una tasa de rechazo de 56.7%, la más alta de todo el
dataset — vale la pena mencionarla puntualmente si se discute la robustez del filtro por condición.

## 5. La curva de ablación confirma el patrón esperado, con matices

Para Antocianinas y Fenoles, R² sin aumentación (n_synth=0) es **negativo** (-1.324 y -0.133 respectivamente):
el modelo es efectivamente inservible sin aumentación, como anticipaba el brief. Ambos alcanzan una meseta
razonable ya en n_synth=10-30 (R²≈0.90-0.95), sin mejora clara al seguir aumentando hasta n_synth=90 (de hecho,
Fenoles y ABTS muestran una ligera caída en R² en n=60 antes de recuperarse en n=90) — evidencia de que
n_synth=30 (el valor usado en Tabla 3) es una elección razonable, no arbitrariamente insuficiente ni excesiva.
ABTS y DPPH no tienen punto en n_synth=0 en la figura (R² indefinido con solo 3 y 2 muestras reales,
`UndefinedMetricWarning` de scikit-learn) — se registran como NaN en la tabla, no se fuerza un valor.
