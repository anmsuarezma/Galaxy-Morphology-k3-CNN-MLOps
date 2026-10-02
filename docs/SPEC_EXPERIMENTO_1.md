# Galaxy3 — Especificación técnica y pedagógica para Claude Code
## Experimento 1: CNN propia entrenada desde cero

> **Propósito**
>
> Este archivo es la especificación de arquitectura, ingeniería, MLOps, HPC y estilo de código
> para construir el **primer experimento del proyecto Galaxy3**.
>
> Claude Code debe tratar este documento como el contrato del proyecto.
>
> **No rediseñar la arquitectura general salvo que exista una incompatibilidad técnica real.**
> Si aparece una incompatibilidad, explicarla antes de hacer un cambio estructural.
>
> **Versión 3.** Cambios respecto a v1: protocolo experimental simplificado (sin scheduler,
> sin early stopping y sin weight decay en la comparación principal; épocas fijadas por un
> piloto mínimo), seeds separadas para split y entrenamiento, checkpoint compatible con
> `torch.load(weights_only=True)`, generalization gap medido en modo eval, transformaciones
> construidas desde la configuración, visualización de imágenes y de feature maps (forward
> hooks) en el notebook, y trabajo fase por fase con aprobación explícita.

---

# 1. Contexto del curso

Proyecto del curso unificado **Sistemas Inteligentes II + Introducción al Deep Learning**.

Los estudiantes ya han trabajado conceptualmente:

- tensores de imágenes `B x C x H x W`;
- `Conv2d`, kernels, feature maps, canales, padding y stride;
- pooling;
- `Flatten`;
- capas `Linear`;
- logits, softmax y entropía cruzada;
- backpropagation;
- train / validation / test;
- underfitting y overfitting;
- Dropout;
- early stopping;
- checkpoints;
- validación para selección de hiperparámetros;
- schedulers, especialmente `ReduceLROnPlateau`;
- comparación de configuraciones.

Antes del proyecto también se trabajará:

- Batch Normalization;
- `BatchNorm2d`;
- `model.train()` y `model.eval()`;
- running statistics;
- modelos preentrenados;
- transfer learning;
- feature extraction;
- fine-tuning;
- `requires_grad`;
- ResNet.

El proyecto no debe introducir esos conceptos desde cero.

La lógica pedagógica general es:

`concepto -> implementación -> experimento -> comparación`

---

# 2. Alcance de esta fase

Construir únicamente:

**GalaxyCNN propia, entrenada desde cero.**

No implementar todavía:

- ResNet18;
- feature extraction;
- fine-tuning;
- Docker;
- APIs;
- cloud computing;
- serving;
- model registry;
- DDP;
- multi-GPU;
- custom CUDA kernels;
- Automatic Mixed Precision;
- barridos de hiperparámetros (learning rate, `dropout_p`);
- submuestreo de train (`train_fraction`);
- data augmentation;
- scheduler y early stopping (se documentan como extensión futura; no se implementan).

Estas ideas quedan como extensiones posteriores.

La infraestructura sí debe quedar reusable para la futura ResNet.

---

# 3. Pregunta experimental

La pregunta general del bloque es:

> **¿Qué estrategia resulta más adecuada cuando tenemos un conjunto relativamente pequeño de imágenes astronómicas?**

En esta primera fase construimos una referencia:

**CNN propia entrenada desde cero**

y observamos, sobre **la misma arquitectura** y con el mismo protocolo de entrenamiento,
el cambio asociado a:

- Batch Normalization (run A → run B);
- Dropout (run B → run C).

Principio:

> **No diseñamos el problema para que BatchNorm o Dropout "ganen".**
> Construimos un problema correcto y observamos qué ocurre.

Una conclusión como "en este problema, con esta arquitectura y esta cantidad de datos,
Dropout no produjo una mejora apreciable" es válida.

Scheduler y early stopping **no se implementan** en esta fase: mezclarlos con Dropout
impediría atribuir los cambios observados. Quedan documentados como extensión futura.

La selección de configuraciones se hace con validation.

El test se reserva para la evaluación final.

---

# 4. Dataset: Galaxy3

## 4.1 Fuente

Galaxy3 se construirá a partir de **Galaxy10 DECaLS**.

Clases decididas para el proyecto:

1. `Round Smooth`
2. `Barred Spiral`
3. `Edge-on without Bulge`

Usar **todas las imágenes disponibles** de esas tres clases.

No submuestrear.

Si al inspeccionar el archivo fuente los nombres o índices de clases no coinciden exactamente con esta
especificación, no inventar un mapeo. Reportar primero el esquema real.

Referencia esperada (documentación de astroNN; verificar contra el archivo real):

- `Galaxy10_DECals.h5` contiene 17 736 imágenes de 256×256 con 3 bandas (g, r, z);
- índice 2: Round Smooth, 2645 imágenes;
- índice 5: Barred Spiral, 2043 imágenes;
- índice 8: Edge-on without Bulge, 1423 imágenes;
- total esperado de Galaxy3: 6111 imágenes.

Consecuencias:

- hay un desbalance moderado (≈1.9:1 entre la clase mayor y la menor); reportar siempre
  macro F1 junto a accuracy;
- las imágenes son una composición en falso color de las bandas g, r, z, no una fotografía
  RGB; documentarlo en el notebook y en los títulos de las figuras.

## 4.2 Estructura final

```text
data/
└── galaxy3/
    ├── train/
    │   ├── round_smooth/
    │   ├── barred_spiral/
    │   └── edge_on_no_bulge/
    ├── val/
    │   ├── round_smooth/
    │   ├── barred_spiral/
    │   └── edge_on_no_bulge/
    └── test/
        ├── round_smooth/
        ├── barred_spiral/
        └── edge_on_no_bulge/
```

Usar `torchvision.datasets.ImageFolder`.

La preparación se realiza una sola vez mediante:

`scripts/prepare_galaxy3.py`

El script debe:

1. leer el dataset fuente;
2. seleccionar únicamente las tres clases;
3. usar todas las imágenes de esas clases;
4. construir un split reproducible;
5. exportar las imágenes con estructura `ImageFolder`;
6. generar `manifest.csv`;
7. registrar conteos por clase y split;
8. usar una semilla explícita.

No descargar varios GB automáticamente durante tests ni al importar módulos.

## 4.3 Split

Usar:

- 70% train;
- 15% validation;
- 15% test.

El split debe ser:

- estratificado;
- reproducible;
- realizado durante preparación;
- independiente del entrenamiento.

Semilla del split:

```python
split_seed = 2026
```

`split_seed` pertenece a la preparación de datos y se registra en `dataset_stats.json`.
Es distinta de `train_seed` (sección 9), que controla inicialización, orden de batches y
máscaras de Dropout. **Cambiar `train_seed` nunca debe cambiar el split.**

Tamaños esperados: aproximadamente 4280 train, 915 validation y 915 test.
Con ~915 imágenes de validation, 1 punto de accuracy equivale a ~9 imágenes.

**El test no puede entrar en `fit()`.**

---

# 5. Preprocesamiento

GalaxyCNN trabajará con imágenes:

`(3, 128, 128)`

Pipeline baseline:

```text
Resize((128, 128))
ToTensor()
Normalize(mean_train, std_train)
```

`CenterCrop` **no** forma parte del baseline. Solo se agrega, antes de `Resize`, si la
inspección visual del piloto (sección 16.1) muestra demasiado cielo vacío alrededor de la
galaxia. En ese caso:

- se controla con el campo `center_crop: int | None` de la configuración;
- se aplica igual a train, validation y test;
- se documenta como decisión del piloto, no como variable experimental.

`mean_train` y `std_train` (por canal) deben calcularse **solo con train**, sobre las
imágenes **después** de las transformaciones geométricas (crop si aplica, y `Resize`) y de
`ToTensor()`, es decir, en la misma escala `[0, 1]` que recibirá `Normalize`.

Implementar el cálculo como una función reutilizable, llamada por el script de preparación,
y guardar el resultado en:

`dataset_stats.json`

junto con `split_seed` y la configuración de transformaciones usada. Si cambia el crop o la
resolución, las estadísticas deben recalcularse.

Las transformaciones se construyen a partir de la configuración (resolución, crop,
estadísticas de normalización) y no quedan fijadas dentro de `data.py`. La futura ResNet
preentrenada usará otra normalización y posiblemente otra resolución.

Para visualizar imágenes normalizadas hay que desnormalizarlas antes de `imshow`
(ver `plots.py`).

En esta primera fase no usar data augmentation. Queremos aislar arquitectura, BatchNorm y
Dropout.

---

# 6. GalaxyCNN

Debe existir **una sola clase configurable**.

Interfaz:

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=False,
    dropout_p=0.0,
)
```

También debe funcionar:

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=True,
    dropout_p=0.0,
)
```

y:

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=True,
    dropout_p=0.3,
)
```

sin cambiar el engine ni el contrato público.

## 6.1 Arquitectura

Entrada:

`(B, 3, 128, 128)`

Bloque 1:

```text
Conv2d(3, 32, kernel_size=3, padding=1)
BatchNorm2d(32) o Identity
ReLU
MaxPool2d(2, 2)
```

Salida:

`(B, 32, 64, 64)`

Bloque 2:

```text
Conv2d(32, 64, kernel_size=3, padding=1)
BatchNorm2d(64) o Identity
ReLU
MaxPool2d(2, 2)
```

Salida:

`(B, 64, 32, 32)`

Bloque 3:

```text
Conv2d(64, 128, kernel_size=3, padding=1)
BatchNorm2d(128) o Identity
ReLU
MaxPool2d(2, 2)
```

Salida:

`(B, 128, 16, 16)`

Bloque 4:

```text
Conv2d(128, 256, kernel_size=3, padding=1)
BatchNorm2d(256) o Identity
ReLU
MaxPool2d(2, 2)
```

Salida:

`(B, 256, 8, 8)`

Después:

```text
AdaptiveAvgPool2d((2, 2))
```

Salida:

`(B, 256, 2, 2)`

Después:

```text
Flatten
Linear(1024, 128)
ReLU
Dropout(dropout_p) o Identity
Linear(128, 3)
```

Salida:

`(B, 3)`

El modelo devuelve **logits**, no softmax.

## 6.2 BatchNorm

Si `use_batchnorm=True`:

```python
nn.BatchNorm2d(channels)
```

Si `False`:

```python
nn.Identity()
```

Cuando BN está activo, preferir:

```python
nn.Conv2d(..., bias=False)
```

Cuando está apagado:

```python
bias=True
```

es válido.

## 6.3 Dropout

Solo inicialmente en el clasificador.

Si `dropout_p == 0.0`:

```python
nn.Identity()
```

Si `dropout_p > 0.0`:

```python
nn.Dropout(dropout_p)
```

Validar:

`0 <= dropout_p < 1`.

Si no se cumple, lanzar `ValueError`.

## 6.4 Conteo de parámetros esperado

Con la arquitectura de 6.1:

| Configuración | Parámetros entrenables |
|---|---|
| `use_batchnorm=False` (convoluciones con bias) | 520 003 |
| `use_batchnorm=True` (convoluciones sin bias + BN) | 520 483 |

Dropout no agrega parámetros. Los buffers de BN (`running_mean`, `running_var`,
`num_batches_tracked`) no son parámetros.

El clasificador (`1024 → 128 → 3`) concentra ≈25 % de los parámetros.

Si el conteo implementado no coincide, reportar la diferencia antes de continuar.

## 6.5 Nombres de módulos

Organizar el modelo con nombres explícitos y estables, por ejemplo:

```text
features.block1.conv / .norm / .act / .pool
features.block2.conv / .norm / .act / .pool
features.block3.conv / .norm / .act / .pool
features.block4.conv / .norm / .act / .pool
pool         (AdaptiveAvgPool2d)
classifier   (Flatten, Linear, ReLU, Dropout o Identity, Linear)
```

Como `norm` es `BatchNorm2d` o `Identity`, y el Dropout es `Dropout` o `Identity`, los
nombres no cambian entre A, B y C. Así el notebook puede inspeccionar la misma capa en los
tres modelos.

El engine sigue sin acceder a estos nombres (sección 7); solo los usa la inspección
pedagógica.

---

# 7. Contrato externo del modelo

El engine solo puede asumir:

```python
logits = model(X_batch)
```

Contrato:

```text
X_batch.shape == (B, 3, H, W)      # en esta fase H = W = config.image_size = 128
logits.shape  == (B, num_classes)  # en esta fase num_classes = 3
```

El engine no debe asumir literalmente 128 ni 3. GalaxyCNN sí está diseñada para 128×128.

El engine no debe acceder a:

- `model.conv1`;
- `model.features`;
- `model.classifier`;
- detalles internos.

Esto permitirá sustituir GalaxyCNN por ResNet18 más adelante sin reescribir el engine.

---

# 8. Workspace y estructura del repositorio

El proyecto Galaxy3 vive dentro de una raíz común que contiene los experimentos computacionales
del curso. El ambiente Python es **compartido a nivel de esa raíz** y se llama:

```text
mlops_unbosque
```

La estructura conceptual es:

```text
Experimentos_MLOPs/
├── pyproject.toml          # configuración/dependencias compartidas del workspace
├── uv.lock                 # lockfile compartido
├── mlops_unbosque/         # ambiente virtual administrado con uv
│
├── reto_cnn_fashionmnist/
│   └── ...
│
└── galaxy3/
    ├── pyproject.toml      # metadata del paquete si Galaxy3 es miembro del workspace
    ├── README.md
    ├── .gitignore
    │
    ├── scripts/
    │   └── prepare_galaxy3.py
    │
    ├── src/
    │   └── galaxy3/
    │       ├── __init__.py
    │       ├── config.py
    │       ├── data.py
    │       ├── models.py
    │       ├── engine.py
    │       ├── metrics.py
    │       ├── plots.py
    │       ├── experiment.py
    │       ├── utils.py
    │       ├── inspection.py
    │       └── train.py
    │
    ├── notebooks/
    │   └── 01_galaxy3.ipynb
    │
    ├── tests/
    │   ├── test_data.py
    │   ├── test_models.py
    │   ├── test_engine.py
    │   ├── test_smoke.py
    │   ├── test_plots.py
    │   └── test_inspection.py
    │
    ├── data/
    │   └── galaxy3/
    │
    └── outputs/
        ├── checkpoints/
        ├── figures/
        └── metrics/
```

## 8.1 Regla del ambiente

No crear un `.venv` dentro de `galaxy3/`.

No crear otro ambiente virtual local para Galaxy3.

La fuente de verdad del ambiente común debe estar en la raíz `Experimentos_MLOPs/`.

El ambiente compartido se llama:

```text
mlops_unbosque
```

Claude Code debe inspeccionar primero cómo está configurado actualmente `uv` en la raíz y adaptar
el proyecto a esa configuración sin romper los proyectos existentes.

Cuando sea necesario configurar explícitamente el ambiente:

```bash
export UV_PROJECT_ENVIRONMENT=mlops_unbosque
```

En PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "mlops_unbosque"
```

Los comandos de sincronización del ambiente deben ejecutarse desde la raíz común, no creando
un entorno independiente dentro de Galaxy3.

Si el workspace ya tiene una estructura `uv` válida, preservarla. No reconstruirla sin necesidad.

No crear todavía:

- `callbacks/`;
- `pipelines/`;
- `services/`;
- `registry/`;
- `trainers/`;

salvo necesidad real.

Evitar sobreingeniería.

---

# 9. Responsabilidad de los módulos

## `config.py`

Configuración tipada del experimento.

Preferir una `dataclass` simple.

Campos mínimos:

```text
run_name
split_seed
train_seed
num_classes
image_size
center_crop
batch_size
num_workers
use_batchnorm
dropout_p
learning_rate
weight_decay
epochs
device_preference
data_root
output_root
mlflow_experiment_name
```

Defaults:

```text
split_seed = 2026
train_seed = 2026
num_classes = 3
image_size = 128
center_crop = None
batch_size = 64
num_workers = 8
use_batchnorm = False
dropout_p = 0.0
learning_rate = 1e-3
weight_decay = 0.0
epochs = 30                 # provisional; el valor definitivo lo fija el piloto (16.1)
device_preference = "auto"
mlflow_experiment_name = "galaxy3-cnn-from-scratch"
```

Son defaults, no valores óptimos.

`split_seed` debe coincidir con la registrada en `dataset_stats.json`; si no coincide,
`run_experiment` lanza `ValueError`.

La configuración debe poder serializarse a un `dict` de tipos primitivos
(`dataclasses.asdict` y `Path → str`) para el checkpoint y para MLflow.

## `data.py`

Responsabilidad:

- `ImageFolder`;
- transforms;
- train/val/test loaders;
- `class_names`;
- contrato de shapes;
- reproducibilidad.

No sabe nada del modelo.

Interfaz conceptual:

```python
make_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
)
```

No usar test durante entrenamiento.

## `models.py`

Responsabilidad:

- `GalaxyCNN`;
- construcción de bloques;
- conteo de parámetros totales;
- conteo de parámetros entrenables.

No incluir:

- optimizer;
- loaders;
- MLflow;
- training loops.

## `engine.py`

Responsabilidad:

- `train_one_epoch`;
- `evaluate`;
- `fit`.

`train_one_epoch`:

```text
model.train()
for batch:
    mover batch al device
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    acumular métricas
```

`evaluate`:

```text
model.eval()
torch.inference_mode()
forward
loss
métricas
```

Sin `backward` ni `optimizer.step`.

`fit`:

- train;
- validation;
- número fijo de épocas (`epochs`);
- best checkpoint por `val_loss`;
- history;
- learning rate por época;
- tiempo por época;
- **sin test**.

`fit` entrena exactamente `epochs` épocas, sin scheduler ni early stopping, y conserva el
checkpoint de la época con menor `val_loss`.

History mínimo:

```python
{
    "train_loss": [],   # promedio acumulado durante la época, en modo train
    "train_acc": [],    # idem
    "val_loss": [],     # modo eval
    "val_acc": [],      # modo eval
    "lr": [],
    "epoch_time": [],
}
```

más `best_epoch` y `best_val_loss`.

`train_loss` y `train_acc` del history se miden con Dropout activo, BN usando estadísticas
del batch y pesos que cambian durante la época. Sirven para las curvas, pero **no** para
calcular el generalization gap (sección 17).

## `metrics.py`

Responsabilidad:

- accuracy;
- precision por clase;
- recall por clase;
- F1 por clase;
- macro F1;
- confusion matrix;
- generalization gap (definido en la sección 17, con train evaluado en modo eval);
- análisis final.

`metrics.py` **no debe conocer MLflow**.

Evitar conversiones GPU -> NumPy dentro de cada batch si no son necesarias.

## `plots.py`

Responsabilidad:

- generar curvas train/validation de loss;
- generar curvas train/validation de accuracy;
- visualizar learning rate si resulta útil;
- construir la matriz de confusión;
- construir figuras con ejemplos de errores de clasificación;
- mostrar grillas de imágenes originales por clase, antes de cualquier transformación
  (`plot_original_galaxies`);
- mostrar grillas de batches transformados (`plot_image_batch`), desnormalizados solo para
  visualización;
- mostrar predicciones (`plot_predictions`): imagen, clase real, clase predicha y
  probabilidad;
- visualizar feature maps (`plot_feature_maps(feature_maps, sample_index=0,
  max_channels=16)`), en escala de grises y con normalización visual por mapa;
- devolver objetos `matplotlib.figure.Figure`.

Detalles técnicos:

- la desnormalización es `x * std + mean` seguida de `clamp(0, 1)`; con tensores
  `(C, H, W)`, `mean` y `std` deben tener forma `(3, 1, 1)`; después de
  `permute(1, 2, 0)`, forma `(3,)`;
- las probabilidades se obtienen con `softmax` sobre los logits **solo para
  visualización**; el modelo sigue devolviendo logits;
- los títulos deben aclarar que las imágenes son composiciones en falso color (g, r, z).

Las funciones de `plots.py` deben ser independientes de MLflow.

Ejemplo conceptual:

```python
fig = plot_training_curves(history)
```

La misma figura debe poder:

1. mostrarse en el notebook;
2. guardarse localmente;
3. registrarse como artifact en MLflow;

sin duplicar la lógica de construcción de la figura.

## `utils.py`

Responsabilidad transversal:

- device;
- seeds;
- timing;
- paths;
- creación de carpetas;
- serialización simple de config.

No convertir `utils.py` en un cajón de lógica sin dueño.

## `inspection.py`

Herramientas de inspección pedagógica. No participan en el entrenamiento.

```python
capture_feature_maps(
    model: nn.Module,
    X_batch: torch.Tensor,
    layer_name: str,
    device: torch.device,
) -> torch.Tensor
```

Requisitos:

- registrar un **forward hook temporal** sobre el módulo `layer_name` de
  `model.named_modules()`;
- lanzar `ValueError` si `layer_name` no existe;
- usar `model.eval()` y `torch.inference_mode()` durante la captura;
- retirar el hook en un bloque `finally`, incluso si ocurre una excepción;
- restaurar el modo train/eval previo del modelo;
- devolver las activaciones desacopladas del grafo, en CPU, con forma `(B, C, H, W)`;
- no modificar el `forward` público de `GalaxyCNN`.

Capa recomendada: `features.blockN.act` (salida de la ReLU). Engancharse en `conv` muestra
la señal antes de BN y ReLU, que es menos interpretable; si se hace, indicarlo en la figura.

Formas esperadas con entrada de 128×128:

```text
features.block1.act -> (B,  32, 128, 128)
features.block2.act -> (B,  64,  64,  64)
features.block3.act -> (B, 128,  32,  32)
features.block4.act -> (B, 256,  16,  16)
```

No asignar automáticamente significado semántico a cada canal. Los feature maps sirven para
observar cómo cambia la representación interna con la profundidad, no para afirmar que un
canal concreto "detecta" una estructura astronómica específica.

## `experiment.py`

Orquestador reusable.

Debe exponer:

```python
run_experiment(config)
```

Flujo:

```text
config
-> train_seed
-> device
-> dataloaders (train, validation)
-> model
-> loss
-> optimizer
-> fit
-> cargar best checkpoint
-> evaluar train y validation en modo eval
-> metrics
-> plots
-> MLflow
```

`run_experiment` **no** evalúa test.

La evaluación final en test es una función separada y explícita:

```python
evaluate_on_test(checkpoint_path, config)
```

Se llama una sola vez, después de seleccionar la configuración con validation
(sección 17). Registra sus resultados en el run de MLflow que produjo el checkpoint;
para eso, guardar el `run_id` junto al checkpoint.

No duplicar training loops.

## `train.py`

Entry point pequeño.

Objetivo:

```bash
uv run python -m galaxy3.train
```

Debe:

1. leer/crear configuración;
2. procesar argumentos CLI si se implementan;
3. llamar `run_experiment(config)`.

No contiene training loop.

Puede usar `argparse`.

Ejemplo:

```bash
uv run python -m galaxy3.train \
    --run-name batchnorm_dropout \
    --use-batchnorm \
    --dropout-p 0.3 \
    --train-seed 2026
```

## Notebook

`notebooks/01_galaxy3.ipynb` es el orquestador pedagógico.

Debe servir para:

- mostrar imágenes originales (256×256) y transformadas (128×128) lado a lado;
- inspeccionar batches con `plot_image_batch`, como en el reto FashionMNIST;
- revisar shapes;
- inspeccionar arquitectura;
- contar parámetros;
- demostrar BN on/off;
- demostrar Dropout on/off;
- lanzar `run_experiment(config)`;
- analizar history;
- visualizar curvas;
- revisar MLflow;
- analizar errores con `plot_predictions`;
- capturar y visualizar feature maps de distintas profundidades con `capture_feature_maps`
  y `plot_feature_maps`.

Las funciones de visualización e inspección viven en `plots.py` e `inspection.py`; el
notebook solo las llama. Así pueden testearse y, cuando convenga, registrarse en MLflow.

No reimplementar:

```python
for epoch in ...
```

Notebook y CLI son dos entradas al mismo núcleo:

```text
notebook ───────┐
                ▼
        run_experiment(config)
                ▲
train.py ───────┘
```

---

# 10. Stack PyTorch de esta fase

Usar explícitamente:

```text
torch.Tensor
torch.nn.Module
nn.Conv2d
nn.BatchNorm2d
nn.ReLU
nn.MaxPool2d
nn.AdaptiveAvgPool2d
nn.Flatten
nn.Linear
nn.Dropout
nn.Identity
nn.CrossEntropyLoss
loss.backward()
torch.optim.AdamW
nn.Module.register_forward_hook             # solo para inspección pedagógica
model.train()
model.eval()
torch.inference_mode()
model.state_dict()
optimizer.state_dict()
torch.save
torch.load(..., weights_only=True)
```

No aplicar softmax antes de `CrossEntropyLoss`.

---

# 11. Device: CUDA, MPS y CPU

El proyecto debe ser device-agnostic.

Orden automático:

```text
CUDA -> MPS -> CPU
```

Implementar algo equivalente a:

```python
def get_device(preference: str = "auto") -> torch.device:
    ...
```

Permitir:

```text
auto
cuda
mps
cpu
```

No usar `.cuda()` dentro del engine.

Usar:

```python
.to(device)
```

El mismo engine debe correr en:

- NVIDIA;
- Apple Silicon;
- CPU.

---

# 12. HPC — alcance hasta nivel 3

## Nivel 1 — vectorización

No usar loops Python sobre:

- píxeles;
- features;
- observaciones individuales;

si la operación puede expresarse con tensores/batches.

## Nivel 2 — CPU/GPU pipeline

Default:

```python
num_workers = 8
```

Es configurable.

`num_workers=0` debe seguir funcionando para debugging o sistemas con problemas de multiprocessing.

Usar:

```python
persistent_workers = num_workers > 0
```

### `pin_memory`

Solo por defecto en CUDA:

```python
pin_memory = device.type == "cuda"
```

MPS y CPU:

```text
False
```

### `non_blocking`

Solo por defecto en CUDA:

```python
non_blocking = device.type == "cuda"
```

Transferencia:

```python
X_batch = X_batch.to(
    device,
    non_blocking=non_blocking,
)
```

igual para etiquetas.

### Reproducibilidad de workers

Usar `torch.Generator` sembrado con `train_seed` para el shuffle.

Sembrar Python y NumPy en cada worker con `worker_init_fn` (por ejemplo, derivando la
semilla de `torch.initial_seed()`).

`persistent_workers=True` solo es válido con `num_workers > 0`.

## Nivel 3 — paralelismo entre experimentos

Las corridas deben ser independientes para poder lanzarse en paralelo desde:

- shells distintos;
- scheduler externo;
- varias GPUs/máquinas en el futuro.

No construir aún un orquestador distribuido.

No usar DDP.

Ejemplos de runs independientes:

```text
A: BN=False, dropout=0.0
B: BN=True,  dropout=0.0
C: BN=True,  dropout=0.3
```

Todos deben poder registrar resultados en el mismo experimento MLflow.

El nivel 3 es una **capacidad de la arquitectura**, no una obligación del curso: basta con
mostrar que dos o tres runs pueden lanzarse como procesos independientes.

Si varios runs comparten la misma GPU al mismo tiempo, `epoch_time` deja de ser comparable
entre ellos; los tiempos solo se comparan entre runs ejecutados en serie. Con
`num_workers=8` por run, varios runs simultáneos pueden saturar la CPU; en ese caso reducir
`num_workers`.

---

# 13. Visualización y registro de resultados

Matplotlib y MLflow tienen responsabilidades distintas y complementarias.

## 13.1 Matplotlib

**Matplotlib es el motor de visualización.**

Debe utilizarse para construir:

- curvas de `train_loss` y `val_loss`;
- curvas de `train_acc` y `val_acc`;
- evolución del learning rate cuando sea útil;
- matriz de confusión;
- ejemplos de errores de clasificación;
- figuras pedagógicas utilizadas en el notebook;
- grillas de galaxias originales y de batches transformados;
- feature maps de capas convolucionales seleccionadas.

Las funciones de plotting deben vivir principalmente en `plots.py`.

No deben registrar directamente nada en MLflow.

Deben devolver objetos `Figure`, por ejemplo:

```python
fig = plot_training_curves(history)
```

En el notebook esa figura puede mostrarse directamente.

## 13.2 MLflow

**MLflow es el sistema de tracking, persistencia y comparación de experimentos.**

Las métricas escalares deben registrarse directamente como métricas MLflow.

Ejemplo conceptual:

```python
mlflow.log_metric(
    "val_loss",
    val_loss,
    step=epoch,
)
```

Esto permite que MLflow construya y compare series por época entre distintos runs.

Las figuras creadas con Matplotlib deben registrarse además como artifacts.

Ejemplo conceptual:

```python
mlflow.log_figure(
    fig,
    "figures/training_curves.png",
)
```

La responsabilidad de registrar figuras y métricas en MLflow corresponde a la capa de
orquestación experimental, principalmente `experiment.py`.

`plots.py` no debe depender de MLflow.

`metrics.py` no debe depender de MLflow.

La separación conceptual es:

```text
PyTorch
   ↓
metrics.py
   ↓
plots.py / Matplotlib
   ↓
notebook para inspección visual
   ↓
experiment.py
   ↓
MLflow para persistencia y comparación
```

En términos prácticos:

- **PyTorch** produce logits, losses y resultados del entrenamiento;
- **`metrics.py`** calcula métricas;
- **`plots.py` + Matplotlib** generan figuras;
- **el notebook** muestra las figuras para discusión pedagógica;
- **`experiment.py`** registra métricas y figuras en MLflow;
- **MLflow** permite comparar runs y conservar artifacts.

No elegir entre Matplotlib y MLflow: se usan juntos.

---

# 14. MLflow

MLflow entra en esta fase **antes de Docker**.

Motivación:

- rastrear configuraciones;
- hiperparámetros;
- métricas;
- curvas;
- mejor época;
- checkpoint;
- duración;
- número de parámetros.

No usar todavía:

- Model Registry;
- serving;
- deployment;
- `mlflow.autolog()` como mecanismo principal.

Queremos logging explícito.

## Experimento

Nombre:

```text
galaxy3-cnn-from-scratch
```

Cada configuración es un run.

## Parámetros a registrar

Como mínimo:

```text
split_seed
train_seed
model_name
num_classes
image_size
batch_size
num_workers
use_batchnorm
dropout_p
optimizer
learning_rate
weight_decay
epochs
center_crop
device
total_parameters
trainable_parameters
train_samples
val_samples
test_samples
```

## Métricas por época

Con `step=epoch`:

```text
train_loss
train_acc
val_loss
val_acc
learning_rate
epoch_time
```

Se pueden registrar después de `fit()` recorriendo `history`.

## Métricas finales del run

Sin `step`:

```text
best_epoch
best_val_loss
train_loss_eval
train_acc_eval
val_acc_best
val_macro_f1
generalization_gap
```

`train_*_eval` se calculan recargando el best checkpoint y evaluando train en modo eval.

No introducir callbacks solo por MLflow.

## Artefactos

Registrar:

- config serializada;
- best checkpoint;
- curvas train/validation;
- tabla final de métricas;
- matriz de confusión cuando se haga evaluación final;
- ejemplos de errores cuando se haga evaluación final.

No es obligatorio registrar en MLflow cada feature map inspeccionado durante la clase;
pueden quedar como visualizaciones del notebook.

## Tracking URI

No hardcodear servidor remoto.

Preferir:

1. `MLFLOW_TRACKING_URI` si existe;
2. fallback local dentro de `galaxy3/` (por ejemplo, un directorio `mlruns/` o una base
   `sqlite:///.../mlflow.db`).

Claude Code debe verificar la versión de MLflow instalada y usar el backend local
recomendado para esa versión. Si se van a lanzar runs en paralelo, preferir un backend que
tolere escrituras concurrentes.

Agregar `data/`, `outputs/` y el backend local de MLflow a `.gitignore`.

Debe funcionar sin cloud.

---

# 15. Estrategia de entrenamiento

Protocolo de la comparación principal:

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)
# sin scheduler
# sin early stopping
# epochs = número fijo definido en el piloto (sección 16.1)
```

Razón: con un número fijo de épocas se observa la historia completa de train y validation,
incluida la posible divergencia después del óptimo. Early stopping, scheduler y weight decay
son regularizadores o modificadores adicionales; mezclarlos con Dropout impediría atribuir
los cambios observados. El modelo que se conserva es el de

```text
argmin_t  val_loss(t)
```

## 15.1 Checkpoint

Guardar si:

```text
val_loss actual < best_val_loss
```

Contenido mínimo:

```python
{
    "epoch": int,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": float,          # usar .item(), no un tensor con grafo
    "config": dict,             # asdict(config) con Path convertidos a str
}
```

Solo tipos primitivos, tensores y `state_dict`. **No** guardar la dataclass ni otros objetos
arbitrarios: desde PyTorch 2.6, `torch.load` usa `weights_only=True` por defecto y los
rechazaría. La carga debe funcionar con:

```python
torch.load(path, map_location=device, weights_only=True)
```

sin recurrir a `weights_only=False`.

## 15.2 Extensiones futuras (no implementar)

`ReduceLROnPlateau` y early stopping, ambos monitoreando `val_loss`, quedan documentados como
extensión posterior. No crear campos de configuración, ramas de código ni tests para ellos
en esta fase.

No usar test para:

- elegir la mejor época;
- seleccionar configuración;
- decidir el número de épocas;
- ninguna otra decisión de entrenamiento.

---

# 16. Runs del Experimento 1

## A — baseline

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=False,
    dropout_p=0.0,
)
```

## B — BatchNorm

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=True,
    dropout_p=0.0,
)
```

## C — BatchNorm + Dropout

```python
GalaxyCNN(
    num_classes=3,
    use_batchnorm=True,
    dropout_p=0.3,
)
```

Los tres runs comparten clases, split, arquitectura base, optimizer, learning rate,
`weight_decay = 0.0`, número de épocas y `train_seed`.

Lectura:

- A → B: cambio asociado a BatchNorm;
- B → C: cambio asociado a Dropout.

No es un diseño factorial completo (falta Dropout sin BN); se acepta por legibilidad.

Nota técnica: la misma `train_seed` no implica los mismos pesos iniciales entre A y B.
Con BN las convoluciones no tienen bias, y cambia la secuencia de números aleatorios
consumida al construir el modelo. Entre B y C la inicialización sí es idéntica (Dropout no
tiene parámetros). Las máscaras de Dropout consumen el RNG global, pero el shuffle del
`DataLoader` y las semillas de sus workers dependen de su propio `torch.Generator`
(sección 12), así que el orden de los batches es el mismo en los tres runs.

Registrar como runs separados en MLflow: `baseline`, `batchnorm`, `batchnorm_dropout`.

Elegir configuración con validation. Métrica primaria, fijada antes de correr:
**macro F1 de validation en la mejor época** (la de menor `val_loss`).

No elegir usando test.

## 16.1 Fase piloto

Antes de los tres runs definitivos, usando solo train y validation:

1. inspeccionar imágenes originales por clase (256×256);
2. inspeccionar las mismas imágenes transformadas a 128×128 y decidir si hace falta
   `CenterCrop`;
3. entrenar solo el baseline (A) durante suficientes épocas para identificar
   aproximadamente su convergencia y el comportamiento de `val_loss`;
4. fijar un presupuesto común de épocas para los tres runs principales.

La duración del piloto y el margen que se deja después del mínimo de `val_loss` son
criterio del docente, no parte de este contrato. El piloto no se usa para probar
combinaciones hasta encontrar un resultado esperado.

Registrar el piloto en un experimento MLflow separado: `galaxy3-pilot`.

Regla de decisión acordada de antemano: si el baseline alcanza `val_acc ≥ 0.98`, las
diferencias entre A, B y C quedarán dentro de unas pocas imágenes de validation. En ese
caso se mantiene el experimento y se discute explícitamente el efecto techo; no se cambian
las clases a posteriori.

El piloto **no** se usa para elegir entre A, B y C.

## 16.2 Interpretación

La comparación usa una sola `train_seed` para mantener el experimento manejable. Las
diferencias se interpretan de forma **descriptiva**, no como evidencia estadística de
superioridad. Con ~915 imágenes de validation, 1 punto de accuracy ≈ 9 imágenes.

Extensión opcional: repetir los tres runs con `train_seed` ∈ {11, 22, 33} y observar en
MLflow cuánto varía el resultado.

---

# 17. Evaluación final

Después de seleccionar configuración con validation:

1. cargar best checkpoint con `torch.load(..., weights_only=True)`;
2. evaluar una vez en test mediante `evaluate_on_test`;
3. reportar:
   - test loss;
   - test accuracy;
   - precision por clase;
   - recall por clase;
   - F1 por clase;
   - macro F1;
   - confusion matrix;
   - ejemplos de errores;
   - generalization gap.

Definición:

```text
generalization_gap = train_acc_eval - val_acc
```

en la mejor época, donde `train_acc_eval` se obtiene recargando el best checkpoint y
evaluando **todo train en modo eval** (`model.eval()`, `torch.inference_mode()`), igual que
validation. No usar el `train_acc` acumulado del history: se mide con Dropout activo y BN
en modo train, que son justamente las variables que se comparan.

Este gap se calcula en `run_experiment`, sin test. En la evaluación final puede reportarse
además `train_acc_eval - test_acc`, con un nombre propio para no confundirlo.

---

# 18. Estándar de programación

## Type hints

Toda función pública, método público y constructor relevante debe tener type hints.

## Docstrings

Usar docstrings estilo **Google**.

Incluir cuando corresponda:

- `Args`;
- `Returns`;
- `Raises`;
- shapes;
- comportamiento relevante.

No llenar el código de comentarios obvios.

## Contratos

Para cada función pública debe quedar claro:

1. entradas;
2. tipos;
3. shapes;
4. valores válidos;
5. salidas;
6. mutación o no;
7. error ante violación.

Usar:

```python
raise ValueError(...)
```

cuando el dominio es inválido.

No corregir silenciosamente entradas incorrectas.

## Responsabilidad única

Evitar una función que a la vez:

- valide;
- cargue datos;
- entrene;
- grafique;
- guarde;
- registre MLflow.

Separar responsabilidades.

## Complejidad

Objetivo:

- funciones pequeñas;
- pocos niveles de anidamiento;
- pocas ramas;
- sin duplicación.

Guía:

- 20–30 líneas de lógica real suele ser razonable;
- una función puede ser mayor si conserva una sola responsabilidad;
- refactorizar cuando la complejidad conceptual aumenta.

Ruff debe revisar complejidad ciclomática.

Target:

```text
max-complexity = 10
```

No sacrificar claridad solo para satisfacer una métrica.

---

# 19. Ruff

Configurar en `pyproject.toml`.

Reglas mínimas:

```text
E
F
I
UP
B
SIM
C90
```

Puede añadirse `PL` si no genera ruido innecesario.

Line length:

```text
100
```

Comandos:

```bash
uv run ruff check .
uv run ruff format --check .
```

Durante desarrollo:

```bash
uv run ruff format .
```

No usar `# noqa` indiscriminadamente.

---

# 20. Tests

Usar `pytest`.

Principio:

```text
test de software != experimento de ML
```

## `test_data.py`

Comprobar contratos como:

```text
X_batch.ndim == 4
channels == 3
y_batch.ndim == 1
B de X == B de y
3 clases
loaders no vacíos
```

Evitar depender del dataset completo si puede usarse un fixture temporal.

## `test_models.py`

Usar:

```python
X = torch.randn(4, 3, 128, 128)
```

Verificar:

```python
logits = model(X)
assert logits.shape == (4, 3)
```

Probar:

```text
BN=False, dropout=0
BN=True, dropout=0
BN=True, dropout=0.3
```

Comprobar presencia/ausencia de:

```text
BatchNorm2d
Dropout
```

según configuración.

## `test_engine.py`

Usar dataset sintético pequeño.

Verificar:

- una época train;
- una evaluación;
- history correcto;
- métricas finitas;
- `fit` corre exactamente `epochs` épocas;
- el checkpoint guardado se carga con `torch.load(..., weights_only=True)` y restaura el
  modelo.

## `test_smoke.py`

Ejecutar:

```text
forward
loss
backward
optimizer.step
```

Verificar:

- loss finita;
- gradientes presentes;
- parámetros entrenables conectados.

Debe ser rápido.

## `test_plots.py`

Verificar:

- `plot_image_batch` y `plot_predictions` devuelven `Figure` sin depender de MLflow;
- la desnormalización de un tensor sintético normalizado recupera valores en `[0, 1]`;
- usar backend no interactivo (`Agg`) y cerrar las figuras con `plt.close(fig)`.

## `test_inspection.py`

Con un modelo y datos sintéticos, verificar:

- `capture_feature_maps` devuelve la forma esperada para `features.block1.act`;
- después de la captura el módulo no conserva hooks registrados;
- el modo train/eval previo del modelo se restaura;
- un `layer_name` inexistente lanza `ValueError`;
- el tensor devuelto está en CPU y no requiere gradiente.

## No testear

No hacer:

```python
assert accuracy > 0.90
```

No asumir como test que:

- BatchNorm mejora;
- Dropout mejora;
- una configuración gana.

Eso se responde experimentalmente.

---

# 21. Reproducibilidad

Implementar una utilidad central de seed, llamada con `train_seed` al inicio de
`run_experiment`. `split_seed` solo se usa en la preparación de datos.

Sembrar:

```text
random
numpy
torch
torch.cuda si aplica
```

Configurar `torch.Generator` para DataLoader.

Objetivo:

- reproducibilidad razonable;
- no prometer identidad bit a bit entre CUDA, MPS y CPU.

Si hay modo determinista, documentar el costo potencial.

---

# 22. Buenas prácticas HPC

Regla:

```text
correctitud -> profiling -> optimización -> paralelización
```

No paralelizar funciones pequeñas porque sí.

Evitar:

```python
tensor.cpu().numpy()
```

dentro de cada batch solo para accuracy.

Mantener cálculos en PyTorch cuando sea posible.

No crear multiprocessing manual para batches.

No escribir kernels CUDA propios.

---

# 23. Dependencias y ambiente compartido

La configuración de dependencias debe integrarse con el `pyproject.toml` del workspace en
`Experimentos_MLOPs/`, preservando la estructura existente.

No crear un entorno independiente dentro de `galaxy3/`.

El ambiente compartido es:

```text
mlops_unbosque
```

El workspace debe disponer, como mínimo, de:
```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
mlflow
h5py
jupyterlab
pytest
ruff
```

Gestión con `uv`.

Los comandos que administran el ambiente compartido deben ejecutarse desde:

```text
Experimentos_MLOPs/
```

Antes de sincronizar, usar la configuración existente del workspace. Si es necesario definir
explícitamente el nombre del ambiente:

```bash
export UV_PROJECT_ENVIRONMENT=mlops_unbosque
```

Comandos conceptuales desde la raíz del workspace:

```bash
uv sync
uv run jupyter lab
uv run python -m galaxy3.train
uv run pytest -q galaxy3/tests
uv run ruff check galaxy3
```

Si la configuración concreta del workspace `uv` exige seleccionar explícitamente el paquete
Galaxy3, Claude Code debe usar la sintaxis apropiada de la configuración existente en vez de
crear un `.venv` local.

Dentro de un notebook, verificar:

```python
import sys
print(sys.executable)
```

El intérprete debe pertenecer a:

```text
.../Experimentos_MLOPs/mlops_unbosque/
```

y no a un ambiente dentro de `galaxy3/`.

---

# 24. Notebook vs CLI

Decisión central:

## Notebook

Para:

- enseñar;
- inspeccionar;
- visualizar;
- experimentar;
- discutir;
- lanzar runs;
- mostrar con Matplotlib las figuras generadas por `plots.py`;
- visualizar batches e imágenes con predicciones, como en el reto FashionMNIST;
- inspeccionar feature maps mediante forward hooks temporales;
- consultar y comparar resultados persistidos en MLflow.

## `train.py`

Para:

- ejecutar el experimento sin interfaz gráfica;
- reproducir una corrida;
- permitir ejecución desde shell/scheduler.

Ambos usan:

```python
run_experiment(config)
```

No hay dos implementaciones del entrenamiento.

---

# 25. MLflow y paralelismo de runs

Debe ser posible ejecutar:

```bash
uv run python -m galaxy3.train --run-name baseline ...
uv run python -m galaxy3.train --run-name batchnorm ...
uv run python -m galaxy3.train --run-name bn_dropout ...
```

en procesos independientes.

Todos bajo:

```text
galaxy3-cnn-from-scratch
```

No construir aún scheduler distribuido propio.

---

# 26. Forma de trabajo esperada de Claude Code

Antes de cambiar código:

1. inspeccionar archivos existentes;
2. identificar necesidad concreta;
3. proponer cambio mínimo;
4. indicar archivos a modificar;
5. implementar;
6. verificar shapes;
7. ejecutar smoke tests;
8. comprobar gradientes cuando corresponda;
9. ejecutar Ruff;
10. ejecutar pytest.

**Trabajar una fase a la vez (sección 27).** Al terminar cada fase: detenerse, reportar qué
se hizo, los resultados de Ruff y pytest y cualquier desviación, y **esperar aprobación
explícita** antes de comenzar la siguiente. No encadenar fases.

Si el ambiente existente en la raíz no se llama `mlops_unbosque` o está configurado de otra
manera, reportarlo en la Fase 0 antes de modificar nada.

No rediseñar el proyecto completo sin necesidad.

No introducir dependencias nuevas sin justificación.

No usar `sys.path.append(...)` como solución estructural.

El paquete debe instalarse correctamente con `uv`.

---

# 27. Orden de implementación

## Fase 0 — inspección

- revisar directorio;
- revisar `pyproject.toml`;
- no sobrescribir trabajo existente sin avisar.

## Fase 1 — esqueleto

- carpetas;
- `pyproject.toml`;
- Ruff;
- pytest;
- paquete `galaxy3`.

Verificar:

```bash
uv sync
uv run python -c "import galaxy3; print(galaxy3.__file__)"
```

## Fase 2 — config y utils

- config;
- device;
- seeds;
- paths.

Tests.

## Fase 3 — data

- preparation script;
- transforms;
- `ImageFolder`;
- loaders;
- workers;
- pinning CUDA;
- reproducibilidad.

Tests.

## Fase 4 — GalaxyCNN

- arquitectura;
- conteo de parámetros;
- shapes;
- BN/Dropout toggles.

Tests sintéticos.

## Fase 5 — engine

- train;
- evaluate;
- fit;
- número fijo de épocas;
- best checkpoint por `val_loss`, compatible con `weights_only=True`.

No implementar scheduler ni early stopping.

Tests sintéticos.

## Fase 6 — smoke end-to-end

```text
data -> model -> loss -> backward -> optimizer.step
```

Confirmar gradientes.

## Fase 7 — metrics y plots

Implementar:

- métricas numéricas en `metrics.py`;
- visualizaciones Matplotlib en `plots.py`;
- funciones de plotting independientes de MLflow;
- retorno de objetos `Figure` reutilizables en notebook y MLflow;
- `plot_original_galaxies`, `plot_image_batch` y `plot_predictions` con desnormalización;
- `plot_feature_maps` en `plots.py` y `capture_feature_maps` en `inspection.py`.

## Fase 8 — experiment

Implementar:

```python
run_experiment(config)
evaluate_on_test(checkpoint_path, config)
```

`evaluate_on_test` se implementa y se prueba con datos sintéticos, pero no se ejecuta sobre
el test real en esta fase.

## Fase 9 — MLflow

Integrar:

- params;
- metrics;
- artifacts.

Sin autolog como mecanismo principal.

## Fase 10 — CLI

Implementar:

```bash
uv run python -m galaxy3.train
```

## Fase 11 — piloto

Ejecutar la fase piloto de la sección 16.1. Reportar la grilla visual, la decisión sobre
`CenterCrop`, la época del mínimo de `val_loss` y el valor de `epochs` propuesto.
**Esperar decisión** antes de los runs definitivos.

## Fase 12 — notebook

Crear notebook docente sobre el núcleo reusable. Debe incluir, como mínimo:

- grilla de galaxias originales;
- grilla de imágenes transformadas;
- inspección de shapes;
- ejecución de los tres runs;
- curvas de entrenamiento;
- acceso a MLflow;
- feature maps de al menos dos profundidades distintas (por ejemplo, bloque 1 y bloque 3).

## Fase 13 — runs definitivos

Ejecutar A, B y C con las épocas fijadas, compararlos en MLflow, seleccionar con
validation y, solo entonces, ejecutar `evaluate_on_test` una vez sobre la configuración
elegida.

---

# 28. Smoke tests obligatorios

Antes del entrenamiento completo:

## Shape

```python
X = torch.randn(4, 3, 128, 128)
model = GalaxyCNN(...)
logits = model(X)
assert logits.shape == (4, 3)
```

## Loss

```python
y = torch.tensor([0, 1, 2, 1])
loss = nn.CrossEntropyLoss()(logits, y)
assert torch.isfinite(loss)
```

## Backward

```python
loss.backward()
```

Confirmar gradientes en parámetros entrenables.

## Optimizer

```python
optimizer.step()
```

sin error.

## BatchNorm

Verificar que `use_batchnorm=False` no cree `BatchNorm2d`.

Verificar que `use_batchnorm=True` sí.

## Dropout

Verificar `dropout_p=0` y `dropout_p>0`.

## Engine

Ejecutar una época sobre datos sintéticos.

## Checkpoint

Guardar un checkpoint sintético y recargarlo con `torch.load(..., weights_only=True)`.

---

# 29. Criterios de aceptación

La fase se considera lista cuando:

1. `uv sync` reconstruye el ambiente;
2. `import galaxy3` funciona;
3. Ruff pasa;
4. pytest pasa;
5. GalaxyCNN produce `(B, 3)`;
6. CUDA/MPS/CPU están soportados;
7. `num_workers=8` es default configurable;
8. CUDA usa `pin_memory` y `non_blocking` condicionalmente;
9. notebook y CLI comparten `run_experiment`;
10. `train.py` corre con `python -m`;
11. test no aparece dentro de `fit`;
12. el experimento principal usa un número fijo de épocas definido después del piloto;
13. no se implementan scheduler ni early stopping;
14. checkpoint usa validation y se carga con `torch.load(..., weights_only=True)`;
15. MLflow registra params, métricas y artifacts;
16. las configuraciones A/B/C pueden ejecutarse como runs independientes;
17. no se implementó ResNet;
18. no se implementó Docker/API/cloud;
19. no hay training loop duplicado;
20. los contratos están documentados;
21. Galaxy3 usa el ambiente compartido `mlops_unbosque` de `Experimentos_MLOPs`;
22. no existe un `.venv` independiente dentro de `galaxy3/`;
23. `plots.py` genera figuras con Matplotlib sin depender de MLflow;
24. `experiment.py` registra en MLflow métricas y figuras producidas por el núcleo del proyecto;
25. `split_seed` y `train_seed` están separadas; cambiar `train_seed` no cambia el split;
26. el generalization gap se calcula con train evaluado en modo eval;
27. `run_experiment` no evalúa test; `evaluate_on_test` es explícita y se usa una sola vez;
28. el notebook muestra batches desnormalizados y predicciones;
29. la fase piloto está documentada (decisión de crop, `epochs` elegido y justificación);
30. los tres runs principales usan `weight_decay = 0.0`, sin scheduler y sin early stopping;
31. el conteo de parámetros coincide con la sección 6.4 o la diferencia está justificada;
32. el notebook captura y visualiza feature maps sin modificar el `forward` del modelo;
33. los hooks de inspección se eliminan siempre, incluso si ocurre una excepción;
34. los nombres de los bloques convolucionales son estables entre A, B y C.

---

# 30. Entregables esperados de Claude Code

Al finalizar, reportar:

1. árbol final del repositorio;
2. archivos creados/modificados;
3. responsabilidades;
4. comandos exactos para:
   - `uv sync`;
   - tests;
   - Ruff;
   - Jupyter;
   - entrenamiento CLI;
   - MLflow;
5. resultado de smoke tests;
6. resultado de pytest;
7. resultado de Ruff;
8. shapes verificadas;
9. número total y entrenable de parámetros de GalaxyCNN;
10. cualquier desviación necesaria frente a esta especificación;
11. resultados del piloto, cuando corresponda a la fase.

---

# 31. Restricciones finales

No avanzar todavía a ResNet18.

No introducir Docker.

No introducir APIs.

No introducir cloud.

No implementar DDP.

No implementar AMP.

No crear un framework de callbacks sofisticado.

No hacer barridos de hiperparámetros, `train_fraction` ni augmentation en esta fase.

No ocultar la relación entre:

```text
Conv2d
BatchNorm2d
ReLU
MaxPool2d
loss.backward()
optimizer.step()
```

y los conceptos matemáticos correspondientes.

Prioridades:

```text
claridad > abstracción sofisticada
reproducibilidad > conveniencia temporal
contratos claros > código mágico
tests de software != experimentos de ML
```

Para HPC:

```text
correctitud -> profiling -> optimización -> paralelización
```

---

# 32. Resumen ejecutivo

Sistema objetivo:

```text
                           ┌──────────────────┐
                           │ ExperimentConfig │
                           └────────┬─────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │      run_experiment()       │
                     │       experiment.py         │
                     └──────────────┬──────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
       data.py                   models.py                 engine.py
          │                         │                         │
          │                    GalaxyCNN                     │
          │              BN/Dropout configurables            │
          │                         │                         │
          └─────────────────────────┴─────────────────────────┘
                                    │
                                    ▼
                               metrics.py
                                    │
                                    ▼
                                plots.py
                              (Matplotlib)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                      notebook           experiment.py
                                               │
                                               ▼
                                             MLflow
                              (tracking + artifacts + comparison)
```

Dos entradas:

```text
01_galaxy3.ipynb ─────┐
                      ▼
              run_experiment(config)
                      ▲
python -m galaxy3.train
```

HPC:

```text
8 DataLoader workers
        │
        ▼
     batches
        │
        ├── CUDA -> pin_memory + non_blocking
        ├── MPS
        └── CPU
```

Todo debe ejecutarse sobre el ambiente compartido:

```text
Experimentos_MLOPs/mlops_unbosque
```

sin crear un entorno independiente dentro de Galaxy3.

La fase termina cuando la **GalaxyCNN propia desde cero** puede entrenarse, compararse y rastrearse
de forma reproducible tanto desde notebook como desde CLI.

**ResNet18 empieza únicamente después de estabilizar este sistema.**
