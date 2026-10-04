# Galaxy3 — Decisiones del piloto (Fase 11, sección 16.1)

> Registro de las decisiones tomadas tras el piloto del baseline (A). Estas decisiones fijan
> el protocolo de los tres runs principales (A, B, C) y no se modifican después de verlos.
> El piloto usó solo train y validation; el test no se usó para ninguna decisión.

## Decisiones

| Decisión | Valor | Aplica a |
|---|---|---|
| `center_crop` | `None` (sin `CenterCrop`) | train, validation y test; runs A, B y C |
| `epochs` | **40** | runs A, B y C |

Ambas decisiones las tomó el docente. El resto del protocolo sigue la sección 15 de la spec
sin cambios: AdamW, `lr = 1e-3`, `weight_decay = 0.0`, sin scheduler ni early stopping,
`train_seed = 2026`, `split_seed = 2026` y best checkpoint por `argmin val_loss`.

Como no hay crop, `dataset_stats.json` (media y desviación estándar de train tras
`Resize((128, 128))` y `ToTensor()`) sigue siendo válido y no se recalcula.

Nota: el default de `ExperimentConfig.epochs` sigue siendo 30 (provisional, sección 9). Los
runs principales deben pasar `--epochs 40` de forma explícita.

## Piloto ejecutado

```bash
uv run python -m galaxy3.train --run-name pilot_baseline --epochs 50 \
    --mlflow-experiment galaxy3-pilot
```

- Configuración: baseline A (`use_batchnorm=False`, `dropout_p=0.0`), 50 épocas, el resto
  con los defaults.
- MLflow: experimento `galaxy3-pilot`, run_id `848709cc9a884bad9f3868c3a1b420ad`.
- Entorno: RTX 3060 (CUDA), torch 2.14.0+cu132, mlflow 3.16.1.
- Tiempo: ≈ 2.7 s por época (3.5 s la primera, por el arranque de workers y CUDA).
- Figura: `galaxy3/outputs/figures/pilot_baseline_training_curves.png` (no versionada).

## Resultados en la mejor época (best checkpoint, evaluado en modo eval)

| Métrica | Valor |
|---|---|
| Época del mínimo de `val_loss` | **14** |
| `best_val_loss` | 0.2140 |
| `val_acc` | 0.9193 |
| `val_macro_f1` | 0.9220 |
| `train_acc_eval` | 0.9853 |
| `generalization_gap` | +0.0660 |

El mínimo es poco pronunciado: la época 8 alcanzó 0.222, y entre las épocas 6 y 16
`val_loss` oscila entre 0.21 y 0.26.

## Comportamiento después del mínimo

| Época | 14 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| `val_loss` | 0.214 | 0.314 | 0.329 | 0.372 | 0.432 | 0.464 |
| `val_acc` | 0.919 | 0.928 | 0.943 | 0.947 | 0.943 | 0.947 |
| `train_loss` | 0.079 | 0.032 | 0.007 | 0.0003 | 0.0001 | 0.0000 |

- Desde la época 22, `val_loss` sube de forma casi monótona hasta más del doble de su
  mínimo. Desde la época 27, `train_acc` vale 1.0: el modelo memoriza train.
- `val_acc` no empeora: sube hasta una meseta de ≈ 0.946. La subida de `val_loss` refleja
  exceso de confianza en los errores que el modelo todavía comete, no más errores. Elegir
  por `val_loss` (protocolo fijado de antemano) da una `val_acc` ≈ 2.7 puntos menor que la
  de épocas tardías. Esto se discute en clase; no cambia el protocolo.

## Justificación de `epochs = 40`

1. **Margen tras el mínimo.** 40 ≈ 2.9 × 14. En A, la divergencia de `val_loss` ya es
   inequívoca hacia la época 30, así que las curvas muestran el sobreajuste completo.
2. **Mínimos desplazados en B y C.** BatchNorm puede adelantar la convergencia y Dropout
   puede retrasarla. Con 40 épocas, un mínimo que se desplace hasta las épocas 20–25 todavía
   deja unas 15 épocas de historia posterior.
3. **Costo despreciable.** Unos 1.8 minutos por run en la RTX 3060.
4. **50 épocas no aportan información.** Después de la época 30, A solo prolonga la misma
   tendencia.

## Regla de efecto techo

No aplica. `val_acc` es 0.919 en la mejor época y su máximo en todo el run es 0.948, ambos
por debajo de 0.98. Con 917 imágenes de validation, 1 punto de accuracy ≈ 9 imágenes.

## Determinismo y `num_workers` (decisión previa a la Fase 13)

Los runs oficiales A, B y C usan `--deterministic` y `--num-workers 0`.

### `--deterministic`

Sin determinismo, cuDNN elige algoritmos no deterministas y dos ejecuciones idénticas
divergen. Evidencia en el experimento MLflow `galaxy3-determinism`: run C, 3 épocas,
`num_workers=8`, ejecutados en serie en la RTX 3060.

| Modo | `val_loss` época 2 (ejecución 1 / 2) | ¿Par idéntico? | `epoch_time` medio (épocas 2–3) |
|---|---|---|---|
| `deterministic=False` | 0.4576 / 0.3216 | No | 2.71 s |
| `deterministic=True` | 0.34598816650124387 en ambas | Sí, bit a bit en las 3 épocas | 2.78 s |

- Sin determinismo, la diferencia entre dos ejecuciones con la misma `train_seed` es
  comparable a las diferencias que se quieren atribuir a BatchNorm o Dropout.
- El costo del modo determinista es ≈ 2–3 %, dentro del ruido de medición.
- Con `cudnn.deterministic=True` basta en este modelo; no hizo falta
  `torch.use_deterministic_algorithms`.

### `--num-workers 0`

Se usa para evitar un deadlock de los workers del `DataLoader` (procesos creados con
fork). En la verificación, 1 de 5 ejecuciones con `num_workers=8` quedó bloqueada antes del
primer batch: el proceso principal esperaba en la cola del `DataLoader` y un worker estaba
detenido en un `futex`. La causa raíz no se identificó.

`num_workers` **sí cambia los resultados**, aun con `deterministic=True`. Verificado con el
run `c_det_nw0` (`num_workers=0`) contra `c_det_1` (`num_workers=8`): la época 1 es
idéntica bit a bit, pero desde la época 2 el orden de los batches difiere (`val_loss`
época 2: 0.4123 frente a 0.3460). La causa probable es que cada iterador del `DataLoader`
toma una semilla base del mismo `torch.Generator` que el shuffle. Con `num_workers=0` se
crea un iterador por época; con `persistent_workers=True`, uno solo.

Consecuencias:

- A, B y C son reproducibles y comparables entre sí, porque comparten
  `num_workers=0`.
- No son reproducibles bit a bit frente a runs con `num_workers > 0`, incluido el piloto.
- El costo es ≈ 12.8 s por época frente a ≈ 2.8 s (≈ 9–10 min por run de 40 épocas).

## Alcance del piloto

- El piloto no se usó para elegir entre A, B y C ni para probar combinaciones.
- El run de piloto vive en `galaxy3-pilot`, separado de `galaxy3-cnn-from-scratch`.
