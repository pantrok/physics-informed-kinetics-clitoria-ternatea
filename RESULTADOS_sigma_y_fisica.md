# RESULTADOS_sigma_y_fisica.md — Veredicto de la Tarea 6 (σ, física, CV de réplicas)

Fuente: `results/table_10_sigma_sensitivity.csv`, `results/table_11_physics_ablation.csv`,
`results/table_12_replicate_cv_by_compound.csv`, `figures/fig_10_sigma_sensitivity.png`.
`random_state=42` fijo; todas las métricas de desempeño bajo LeaveOneGroupOut (LOCO), agrupado por
`id_cinetica`, siguiendo el esquema establecido en la Tarea 5.

**Contexto obligatorio (leer primero):** la Tarea 5 concluyó **"fuga confirmada / no generaliza"**
(ver [RESULTADOS_validacion_agrupada.md](RESULTADOS_validacion_agrupada.md)): el R²=0.90-0.94 de la
Tabla 3 no sobrevive a una validación que respeta la identidad de condición. Los resultados de 6A/6B de
abajo se interpretan a la luz de eso — son diagnósticos de *por qué*, no un intento de "salvar" el R².

---

## 1. Sensibilidad a σ bajo LOCO (Tarea 6A)

### 1.1 Tabla resumen

| Compuesto | σ=0.02 | σ=0.057 (CV/√3) | σ=0.08 (actual) | σ=0.10 | σ=0.20 | Rango de R² en todo el barrido |
|---|---|---|---|---|---|---|
| Anthocyanins | -0.546 | -0.556 | -0.580 | -0.622 | -0.528 | [-0.622, -0.528] |
| Phenolics | -0.013 | -0.315 | -0.423 | -0.392 | -0.035 | [-0.423, -0.013] |
| Antioxidant Capacity (ABTS) | -1.046 | -1.345 | -1.054 | -1.087 | -1.044 | [-1.345, -0.889] |
| Antioxidant Capacity (DPPH) | -3.110 | -2.839 | -2.825 | -3.008 | -3.463 | [-3.753, -2.825] |

### 1.2 ¿Existe una meseta?

**Sí existe una meseta — pero es una meseta de fracaso, no de éxito.** El R² agrupado permanece
**negativo en absolutamente todo el rango de σ probado (0.02 a 0.20)**, para los 4 compuestos (Fig. 10,
panel izquierdo). La variación de R² dentro de cada compuesto a través de todo el barrido es pequeña
comparada con la brecha frente al split aleatorio (p. ej. Anthocyanins varía solo entre -0.53 y -0.62
en todo el rango, frente a +0.94 del split aleatorio). **Esto significa que la elección exacta de σ
genuinamente no es crítica — tal como anticipaba el brief — pero no porque el modelo sea robusto y
funcione bien en un rango amplio, sino porque no funciona en NINGÚN punto del rango.**

### 1.3 Posición de σ=0.08 respecto al CV empírico

σ=0.08 sigue siendo **conservador** frente al CV empírico de la media-de-tres (~5.7% global, Tarea 6C):
0.08 > 0.057. Como el R² es plano en todo el barrido, no hay ninguna ganancia de desempeño al mover σ
hacia el valor empírico — pero tampoco hay ninguna pérdida. **No hay evidencia, con los datos actuales,
para preferir un σ uniforme sobre uno anclado por compuesto, ni viceversa: la pregunta es discutible en
el texto como elección de diseño razonada (anclaje empírico al CV/√3), no como algo que el desempeño
LOCO pueda arbitrar.**

### 1.4 Auditoría del filtro y dispersión (paneles 2 y 3 de Fig. 10)

Ambos se comportan como cabía esperar (sanity check superado): la tasa de rechazo del filtro ±60% crece
monótonamente con σ (de ~0% en σ=0.02 a 26-43% en σ=0.20), y la dispersión media de k_ext sintéticos por
condición también crece con σ. El filtro y la aumentación funcionan mecánicamente como se diseñaron; el
problema no está ahí, está en la validación fuera de condición (Tarea 5).

---

## 2. Veredicto Tarea 6B — ¿physics-informed aporta sobre ruido ingenuo? (σ=0.08, LOCO)

| Compuesto | R² physics-informed | R² ingenuo | Δ R² (physics − ingenuo) |
|---|---|---|---|
| Anthocyanins | -0.580 | -0.605 | +0.025 |
| Phenolics | -0.423 | **+0.041** | **-0.464** |
| Antioxidant Capacity (ABTS) | -1.054 | -0.980 | -0.075 |
| Antioxidant Capacity (DPPH) | -2.825 | -2.949 | +0.124 |

**Veredicto: NO hay evidencia consistente de que el anclaje físico aporte sobre el ruido ingenuo bajo
LOCO — el resultado es mixto y, en el caso más señalado, desfavorable a "physics-informed".**

- En Anthocyanins y DPPH, physics-informed es marginalmente mejor (Δ ≈ +0.02 a +0.12), pero ambos R²
  siguen siendo fuertemente negativos — la diferencia no es prácticamente relevante.
- En Antioxidant Capacity (ABTS), el control ingenuo es marginalmente mejor (Δ = -0.075), también sobre
  una base negativa.
- **En Phenolics, el control ingenuo obtiene el ÚNICO R² positivo de todo el ejercicio de validación
  agrupada (+0.041), superando claramente al physics-informed (-0.423, Δ = -0.464).** Esto cae en la
  categoría más fuerte de advertencia del brief (§3): *"si el control ingenuo ya reproduce o supera el
  R², es una advertencia fuerte de que el resultado no depende de la física."*

**Conclusión honesta:** el anclaje físico (reajuste Ext-Deg + filtro ±60%) NO está demostrando aportar
generalización fuera de condición sobre una alternativa mucho más simple (perturbar k_ext directamente).
Dado que ninguno de los dos métodos generaliza de forma útil (Tarea 5), esta comparación no puede
"salvar" el término "physics-informed" en el título apoyándose en la generalización — su justificación
debe descansar en otro terreno (p. ej. que el reajuste Ext-Deg SÍ impone consistencia dinámica interna
verificable, aunque eso no se traduzca en mejor R² fuera de condición con este diseño).

---

## 3. CV de réplicas de absorbancia por compuesto (Tarea 6C)

| Compuesto | CV mediano | CV medio | P25–P75 | CV de la media-de-3 (mediana/√3) |
|---|---|---|---|---|
| Antocianinas | **26.75%** | 29.71% | 16.7%–38.3% | 15.44% |
| Fenoles | 7.60% | 8.99% | 4.3%–11.8% | 4.39% |
| ABTS | 3.49% | 5.08% | 2.0%–6.1% | 2.01% |
| DPPH | 12.39% | 14.33% | 7.6%–19.0% | 7.15% |
| **Global** | **9.89%** | 14.49% | 4.1%–19.4% | **5.71%** |

**Antocianinas destaca con CV≈26.7%** — muy por encima del resto (ABTS 3.5%, Fenoles 7.6%, DPPH 12.4%).
Rango de absorbancia en Antocianinas: 0.006–0.060, cerca del límite de detección espectrofotométrico, lo
que eleva el ruido relativo mecánicamente (división entre valores pequeños). **Es una limitación honesta
del diseño experimental, no un error de medición ni de código** — debe documentarse así en Métodos/Limitaciones.

El global (mediana 9.89%, CV-de-media-de-3 ≈ 5.71%) confirma los valores ya conocidos citados en el
brief y usados como ancla para σ en la Tarea 6A.

---

## 4. Recomendaciones de una línea por hallazgo (para la fase de redacción)

- **Elección de σ:** mantener σ=0.08 es defendible (conservador frente al CV empírico, y el desempeño
  LOCO es indiferente a la elección exacta en todo el rango probado) — pero no reportar σ=0.08 como
  "óptimo": es "conservador y sin efecto detectable en el rango 0.02–0.20 bajo la validación agrupada".
- **¿Se sostiene "physics-informed" en el título?** No sobre la base de generalización fuera de
  condición (Tarea 6B es mixto/desfavorable) — la justificación del término, si se mantiene, debe
  apoyarse explícitamente en la consistencia dinámica del reajuste Ext-Deg, no en una ganancia de R²
  demostrada, y debe leerse junto con el veredicto de fuga de la Tarea 5.
- **Cómo reportar la incertidumbre de medición:** reportar el CV por compuesto (Tabla 12), no un CV
  global único — la variación de 3.5% (ABTS) a 26.7% (Antocianinas) es en sí misma un hallazgo relevante
  para la Discusión (proximidad al límite de detección en Antocianinas).
- **Framing global recomendado (combinando Tareas 3, 5 y 6):** el manuscrito tiene evidencia convergente
  de tres análisis independientes (RSM no estimable, RF no generaliza fuera de condición bajo LOCO, y la
  ventaja de "physics-informed" sobre ruido ingenuo no se sostiene bajo la misma validación) de que el
  diseño Box-Behnken de 15 corridas, tras el filtro físico R²≥0.50, es estructuralmente insuficiente para
  sostener predicción ML fuera de muestra. Esto es material legítimo y publicable como advertencia
  metodológica para la comunidad QbD/PAT (ver §6 de `BRIEF_addendum_tarea5_validacion_agrupada.md`).

---

*No se redactan aquí secciones del paper — solo veredicto y números, para decidir el framing con el autor.*
*Ver también [RESULTADOS_validacion_agrupada.md](RESULTADOS_validacion_agrupada.md) (Tarea 5) y
[ANOMALIAS.md](ANOMALIAS.md) (Tarea 3, no-estimabilidad del RSM).*
