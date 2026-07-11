# Captions borrador — Fig. 7 y Fig. 8

Borrador crudo; se pule en la fase de redacción con el autor.

---

## Fig. 7 — `fig_7_response_surfaces.png`

**Borrador de caption:**

> Fig. 7. Physics-informed k_ext response surfaces across the measured design space, one panel per
> target compound (Anthocyanins, Phenolics, Antioxidant Capacity ABTS, Antioxidant Capacity DPPH).
> Each panel is a heatmap of the Random Forest's k_ext prediction, with temperature (T, °C) on the
> x-axis and the 9 solvent × solid:liquid ratio combinations of the Box-Behnken design on the y-axis.
> Color encodes predicted k_ext (min⁻¹); white-ringed black-cross markers indicate the real experimental
> (T, k_ext) points from the physically valid Ext-Deg fits (R²≥0.50), overlaid for visual contrast.
> **All values are model predictions from interpolation strictly within the measured design space
> (T: 60-75 °C); no extrapolation was performed.** Antioxidant Capacity (DPPH) is shown with an
> "insufficient data" panel where applicable (see Table 2 / ANOMALIAS.md).

**Notas para el autor:**
- Cada panel es una malla T=60-75°C cada 0.5°C x 9 combinaciones disolvente×ratio.
- El patrón "escalonado" (no una superficie suave) es un artefacto esperado del Random Forest
  (particiones piecewise-constant), no un error de graficado — discutir en Resultados/Discusión
  (ver ANOMALIAS.md §2).
- Verificar si el eje Y (9 combinaciones) debe reordenarse por relevancia física (p.ej. de mayor a
  menor contenido de etanol) en vez de orden alfabético.

---

## Fig. 8 — `fig_8_augmentation_ablation.png`

**Borrador de caption:**

> Fig. 8. Augmentation ablation curve: test R² of the k_ext Random Forest as a function of the
> augmented dataset size, obtained by varying the number of Monte Carlo synthetic replicates accepted
> per physically valid curve (0, 10, 30, 60, 90), one line per compound. n_synth=0 (real curves only,
> no augmentation) yields a negative R² for Anthocyanins and Phenolics, i.e. an unusable model; R²
> stabilizes into a plateau by n_synth≈10-30, which justifies the n_synth=30 augmentation level used
> to report Table 3. Antioxidant Capacity (DPPH), with only 2 physically valid curves, does not show
> a stable plateau (see ANOMALIAS.md §5).

**Notas para el autor:**
- Eje X = tamaño total del dataset aumentado (reales + sintéticos aceptados), no directamente n_synth
  (por eso el espaciado entre puntos no es uniforme).
- Mencionar explícitamente en el texto que n_synth=30 (usado en Tabla 3 / Fig. 6 / Fig. 7) no es un
  punto arbitrario sino el punto de meseta post-ablación para 3 de 4 compuestos.
- DPPH es ruidoso en toda la curva — no forzar una narrativa de "meseta" para ese compuesto.
