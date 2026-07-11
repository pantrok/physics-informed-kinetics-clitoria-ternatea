# RESULTADOS_validacion_agrupada.md — Veredicto de la Tarea 5 (diagnóstico de fuga)

Fuente: `results/table_7_grouped_validation.csv`, `results/table_8_grouped_baselines.csv`,
`results/table_9_ablation_grouped.csv`, `figures/fig_9_ablation_grouped.png`.
`random_state=42` fijo; validación LeaveOneGroupOut (LOCO) agrupada por `id_cinetica`.

---

## Veredicto explícito: **FUGA CONFIRMADA / NO GENERALIZA**

El R²=0.90-0.94 reportado en la Tabla 3 (split aleatorio por fila) **no refleja generalización real**.
Es la firma de fuga de información por identidad de condición: el `train_test_split` aleatorio reparte
réplicas Monte Carlo de la MISMA condición operativa a ambos lados de train/test, y el Random Forest
memoriza el promedio de una condición ya vista en vez de predecir una condición nueva.

## 1. R² aleatorio (Tabla 3) vs. R² agrupado (LOCO) — caída por compuesto

| Compuesto | n grupos | R² aleatorio (Tabla 3) | R² agrupado (LOCO) | Δ R² |
|---|---|---|---|---|
| Anthocyanins | 6 | 0.944 | **-0.580** | -1.524 |
| Phenolics | 7 | 0.903 | **-0.423** | -1.326 |
| Antioxidant Capacity (ABTS) | 3 | 0.914 | **-1.054** | -1.969 (no concluyente, n<4) |
| Antioxidant Capacity (DPPH) | 2 | 0.402 | **-2.825** | -3.227 (no concluyente, n<4) |

**Los 4 compuestos colapsan a R² negativo bajo LOCO** — incluyendo Anthocyanins y Phenolics, que tienen
suficientes grupos (6 y 7) para que el resultado sea concluyente. Un R² negativo significa que el modelo
predice peor que una línea horizontal en la media de entrenamiento.

## 2. RF agrupado vs. baselines triviales (Tabla 8)

| Compuesto | R² RF agrupado | R² media-global | R² condición-más-cercana |
|---|---|---|---|
| Anthocyanins | -0.580 | -0.408 | -0.768 |
| Phenolics | -0.423 | -0.280 | -0.019 |
| Antioxidant Capacity (ABTS) | -1.054 | -1.079 | -1.382 |
| Antioxidant Capacity (DPPH) | -2.825 | -2.879 | -2.879 |

**El RF agrupado NO supera consistentemente al baseline trivial de media global.** En Anthocyanins y
Phenolics, predecir simplemente la media global de entrenamiento da un R² *menos negativo* (mejor) que el
RF entrenado con todas sus variables de proceso. En Phenolics, el baseline de "condición más cercana"
(-0.019) es notablemente mejor que el RF (-0.423) y casi neutral — sugiere que ni siquiera un vecino
físicamente próximo predice bien, y el RF añade ruido en vez de señal. Solo en ABTS y DPPH el RF iguala
aproximadamente a los baselines (todos igual de malos). **En ningún caso el RF demuestra aportar
modelado real fuera de condición.**

## 3. Curva de ablación agrupada vs. aleatoria (Fig. 9)

La Fig. 9 sobrepone ambas curvas por compuesto. El contraste es dramático y consistente:

- **Random split (línea vieja, Tarea 4):** sube rápidamente a R²≈0.8-0.95 y se mantiene ("meseta" que
  se interpretó en la Tarea 4 como suficiencia de datos).
- **Grouped LOCO (línea nueva):** permanece **negativo en todo el rango de n_synth (0, 10, 30, 60, 90)**,
  para los 4 compuestos, sin mostrar una meseta positiva ni una tendencia clara de mejora con más
  aumentación. En ABTS y DPPH incluso empeora en puntos intermedios.

**Esto confirma que la "meseta" de la Tarea 4 era un artefacto de fuga que crece con n_synth (más réplicas
sintéticas de la misma condición = más oportunidades de fuga entre train/test), no evidencia de que la
aumentación aporte generalización real.**

## 4. Conclusión y framing recomendado

Con este veredicto, el eje central del paper ("ML predice k_ext desde variables de proceso fuera de
muestra") **no se sostiene con los datos actuales**. Siguiendo la nota de framing de la Tarea 5 (§6 del
addendum): el manuscrito debe pivotar hacia:

> "Cuantificación de incertidumbre de k_ext + modelado fenomenológico grey + el hallazgo estructural de que
> un diseño Box-Behnken de 15 corridas, tras el filtro físico, no admite identificación de un RSM clásico
> (Tarea 3) NI generalización ML fuera de condición (Tarea 5)."

Esto es coherente con el hallazgo ya documentado en `ANOMALIAS.md` §3 (RSM cuadrático no estimable): dos
líneas de evidencia independientes (RSM clásico y RF fuera de condición) apuntan a la misma limitación
estructural — el diseño experimental, después del filtro físico R²≥0.50, deja muy pocas condiciones
independientes (2-7) para sostener CUALQUIER modelo predictivo fuera de muestra, paramétrico o no.
Esto es una advertencia metodológica genuina y valiosa para la comunidad QbD/PAT, no un resultado
negativo estéril.

**Nota importante para 6A/6B:** con este veredicto de "fuga confirmada", se espera que la sensibilidad a
σ (Tarea 6A) muestre R² agrupado bajo/negativo en todo el barrido, y que physics-informed vs. ingenuo
(Tarea 6B) puedan empatar en valores bajos — ambos siguen siendo diagnósticos informativos de los
límites del enfoque, no se descartan por este resultado.

---

*Este documento NO redacta secciones del paper — es el veredicto y los números para decidir el framing
con el autor. Ver también [ANOMALIAS.md](ANOMALIAS.md) para el hallazgo relacionado de la Tarea 3 (RSM).*
