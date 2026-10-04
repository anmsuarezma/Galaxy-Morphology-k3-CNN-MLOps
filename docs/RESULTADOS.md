# Galaxy3 — Resultados del Experimento 1 (Fase 13)

> Runs definitivos A, B y C y selección de configuración **con validation**
> (secciones 16 y 17 de la spec). Todavía **sin resultados de test**: `evaluate_on_test` no
> se ha ejecutado.

## Runs verificados

Experimento MLflow `galaxy3-cnn-from-scratch` (contiene exactamente estos tres runs).
Verificado en MLflow:

- los tres están `FINISHED`, con 40 épocas registradas;
- parámetros: `epochs=40`, `deterministic=True`, `num_workers=0`, `train_seed=2026`;
- ninguno tiene métricas de test;
- el JSON local de métricas de cada run tiene el mismo `run_id`, las mismas métricas
  finales y la misma curva de `val_loss` que MLflow.

| Run | `use_batchnorm` | `dropout_p` | run_id |
|---|---|---|---|
| A `baseline` | False | 0.0 | `caa07a916d7a471ebe7ddfe52e366407` |
| B `batchnorm` | True | 0.0 | `1b942309c97c4dbb91b6443b55c55980` |
| C `batchnorm_dropout` | True | 0.3 | `24df078485004e82a8d5a04ed27a15eb` |

Protocolo común (ver `PILOTO.md`): AdamW, `lr = 1e-3`, `weight_decay = 0.0`, sin scheduler
ni early stopping, 40 épocas, `split_seed = 2026`, sin crop, best checkpoint por
`argmin val_loss`. Cada run tardó ≈ 8.4 min de entrenamiento en la RTX 3060.

## Validation en la mejor época

Métricas del best checkpoint, con train y validation evaluados en modo eval. Hay 917
imágenes de validation, así que 1 punto de accuracy ≈ 9 imágenes.

| Run | `best_epoch` | `best_val_loss` | `val_acc` | `val_macro_f1` | `train_acc_eval` | `generalization_gap` | Errores en val |
|---|---|---|---|---|---|---|---|
| A `baseline` | 7 | 0.2159 | 0.9269 | 0.9303 | 0.9502 | +0.0233 | 67 |
| B `batchnorm` | 16 | 0.1384 | 0.9586 | 0.9578 | 0.9883 | +0.0297 | 38 |
| C `batchnorm_dropout` | 12 | 0.1253 | 0.9597 | **0.9594** | 0.9705 | +0.0109 | 37 |

F1 por clase en validation:

| Run | `barred_spiral` | `edge_on_no_bulge` | `round_smooth` |
|---|---|---|---|
| A | 0.890 | 0.967 | 0.933 |
| B | 0.942 | 0.962 | 0.970 |
| C | 0.942 | 0.967 | 0.969 |

En los tres runs la confusión dominante es entre `barred_spiral` y `round_smooth`. En A son
53 de los 67 errores; en B, 22 de 38; en C, 23 de 37.

## Selección según el protocolo

La métrica primaria, fijada antes de correr (sección 16), es el **macro F1 de validation en
la mejor época**. Gana el mayor valor:

> **Configuración seleccionada: C (`batchnorm_dropout`), `val_macro_f1 = 0.9594`.**

La diferencia con B (0.9578) es de 0.0016 en macro F1 y de **una sola imagen** de
validation (37 frente a 38 errores). Con una sola `train_seed` esa diferencia **no es
interpretable**: es mucho menor que la variación entre ejecuciones que cambian solo el
orden de los batches. Por ejemplo, el baseline A del piloto (`num_workers=8`) y el A
oficial (`num_workers=0`) tienen la misma configuración y difieren en la mejor época
(14 frente a 7) y en `val_acc` (0.919 frente a 0.927). La selección de C se aplica
porque así lo fija el protocolo, no como evidencia de que Dropout mejore sobre B.

## Lectura descriptiva (sin test)

- **A → B (BatchNorm).** `val_acc` sube de 0.927 a 0.959 (29 errores menos) y
  `best_val_loss` baja de 0.216 a 0.138. Esta diferencia sí es grande frente a 1 imagen.
  Aun así, es una sola semilla.
- **B → C (Dropout).** Prácticamente igual en `val_acc` y macro F1. El cambio visible está
  en el gap: `train_acc_eval` baja de 0.988 a 0.971 y el gap de +0.030 a +0.011. Es
  consistente con un efecto regularizador, sin mejora apreciable en validation.
- **`val_loss` muy ruidosa en B y C.** En las primeras ~25 épocas hay picos de hasta 1.28
  (B) y 2.39 (C). A tiene como máximo 0.83, en la época 1. La mejor época de C (12,
  0.125) es un mínimo aislado entre las épocas 11 (0.65) y 13 (0.84). La de B (16, 0.138)
  está seguida por 0.60 en la época 17. Que el checkpoint caiga en un valle aislado o no
  depende en parte del azar, otra razón para no interpretar la diferencia B–C.
- **Después del mínimo.** En A, `val_loss` diverge como en el piloto (0.455 en la época 40)
  mientras `val_acc` se mantiene en ≈ 0.94. En B, `val_loss` se estabiliza cerca de su
  mínimo (media de 0.148 en las épocas 31–40). C queda en un punto intermedio (media de
  0.256).

## Figuras

Generadas con `plot_runs_comparison` y `plot_confusion_matrix` a partir de los JSON de
métricas de cada run (sin recalcular nada ni abrir test). Están en
`galaxy3/outputs/figures/fase13/` (no versionadas):

- `runs_comparison_val.png`: `val_loss` y `val_acc` de A, B y C superpuestas, con la mejor
  época marcada.
- `val_confusion_matrix_baseline.png`
- `val_confusion_matrix_batchnorm.png`
- `val_confusion_matrix_batchnorm_dropout.png`

## Pendiente

- `evaluate_on_test` sobre el checkpoint de C, una sola vez, tras aprobación explícita.
  Sus resultados se agregarán aquí sin modificar la selección.
