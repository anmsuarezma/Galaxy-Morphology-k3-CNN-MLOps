# Galaxy3 — Morfología de galaxias con una CNN propia (MLOps)

Proyecto del curso **Sistemas Inteligentes II + Introducción al Deep Learning**
(Universidad El Bosque).

## Qué es

Galaxy3 es un problema de clasificación de imágenes astronómicas con **3 clases**, tomadas
de [Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html):

| Etiqueta | Clase | Imágenes |
|---|---|---|
| `round_smooth` | Round Smooth | 2645 |
| `barred_spiral` | Barred Spiral | 2043 |
| `edge_on_no_bulge` | Edge-on without Bulge | 1423 |

En total son 6111 imágenes de 256×256, divididas de forma estratificada en
70 % train / 15 % validation / 15 % test (`split_seed = 2026`). Son composiciones en
**falso color** de las bandas g, r, z, no fotografías RGB.

**Experimento 1:** una CNN propia (`GalaxyCNN`) entrenada desde cero. Con la misma
arquitectura y el mismo protocolo se comparan tres runs:

| Run | Nombre | BatchNorm | Dropout |
|---|---|---|---|
| A | `baseline` | no | 0.0 |
| B | `batchnorm` | sí | 0.0 |
| C | `batchnorm_dropout` | sí | 0.3 |

- **Protocolo:** AdamW, `lr = 1e-3`, `weight_decay = 0`, 40 épocas fijas, sin scheduler
  ni early stopping. Se conserva el checkpoint de menor `val_loss`.
- **Selección:** se elige con validation (macro F1). Test se evalúa **una sola vez**, al
  final.
- **Registro:** cada run queda en MLflow (parámetros, métricas por época, figuras y
  checkpoint).

Documentos:

- `docs/SPEC_EXPERIMENTO_1.md`: especificación técnica y pedagógica (el contrato del
  proyecto, escrito cuando vivía dentro del workspace del curso).
- `docs/PILOTO.md`: piloto, por qué 40 épocas, determinismo y `num_workers`.
- `docs/RESULTADOS.md`: resultados oficiales de A, B y C y evaluación final en test.

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) y git. uv
instala Python 3.12 si no lo tienes.

```bash
git clone <url-del-repositorio> Galaxy-Morphology-k3-CNN-MLOps
cd Galaxy-Morphology-k3-CNN-MLOps
uv sync
```

`uv sync` crea el ambiente `.venv/` en la raíz del repositorio. Instala el paquete
`galaxy3` en modo editable y las herramientas de desarrollo (pytest, ruff). Todos los
comandos se ejecutan **desde la raíz del repositorio** con `uv run ...`; no hace falta
activar el ambiente.

| Sistema | PyTorch que se instala |
|---|---|
| Linux y Windows | CUDA 13.2 (índice de PyTorch). Funciona también sin GPU, en CPU. |
| macOS (Apple Silicon) | PyPI: CPU y MPS. |

Comprueba la instalación:

```bash
uv run python -c "import torch, galaxy3; print(torch.__version__, torch.cuda.is_available())"
```

## Datos: descarga y preparación

1. Descarga `Galaxy10_DECals.h5` (≈ 2.7 GB) desde Zenodo y guárdalo en `data/raw/`:

   ```bash
   mkdir -p data/raw
   curl -L -o data/raw/Galaxy10_DECals.h5 \
       https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
   ```

   (o descárgalo con el navegador desde ese enlace).

2. Prepara Galaxy3 (una sola vez, ~1–2 min):

   ```bash
   uv run python scripts/prepare_galaxy3.py
   ```

   El script selecciona las 3 clases, hace el split y exporta las imágenes a
   `data/galaxy3/{train,val,test}/<clase>/` (formato `ImageFolder`). También escribe
   `manifest.csv` y `dataset_stats.json` (conteos, `split_seed` y la media/desviación por
   canal calculadas **solo con train**). Debe reportar 4277 / 917 / 917 imágenes. Para
   regenerar los datos, añade `--overwrite`.

`data/`, `outputs/`, `mlruns/` y `mlflow.db` no se versionan.

## Tests y estilo

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Los tests usan datos sintéticos y un MLflow temporal: no necesitan el dataset ni GPU.

## Entrenar desde la terminal (runs A, B y C)

La CLI y el notebook llaman a la misma función, `run_experiment(config)`. Con el protocolo
oficial:

```bash
uv run python -m galaxy3.train --run-name baseline --epochs 40 --deterministic \
    --mlflow-experiment galaxy3-estudiante
uv run python -m galaxy3.train --run-name batchnorm --use-batchnorm --epochs 40 \
    --deterministic --mlflow-experiment galaxy3-estudiante
uv run python -m galaxy3.train --run-name batchnorm_dropout --use-batchnorm --dropout-p 0.3 \
    --epochs 40 --deterministic --mlflow-experiment galaxy3-estudiante
```

- `uv run python -m galaxy3.train --help` lista todas las opciones.
- Los checkpoints, figuras y métricas van a `outputs/`.
- La CLI **no** evalúa test.
- `--num-workers` (por defecto 8) controla los procesos que cargan imágenes. Con
  `--num-workers 0` se reproduce el protocolo de los runs oficiales; el valor de
  `num_workers` cambia el orden de los batches (ver `docs/PILOTO.md`).
- En CPU cada época es mucho más lenta; para practicar, baja `--epochs`.

## Notebook

```bash
uv run jupyter lab
```

Abre `notebooks/01_galaxy3.ipynb`. El kernel debe ser el de `.venv`; la celda de ambiente
lo comprueba. Solo se edita la **celda de parámetros** (sección 0):

| Parámetro | Por defecto | Para qué |
|---|---|---|
| `TRAIN` | `True` | `True` entrena A, B y C. `False` carga los runs oficiales del docente (no existen en tu máquina). |
| `EXPERIMENT_NAME` | `"galaxy3-estudiante"` | Experimento MLflow de tus runs. |
| `EPOCHS` | `40` | Protocolo oficial. Bájalo para practicar, sobre todo en CPU. |
| `NUM_WORKERS` | `4` | Workers del `DataLoader`. `0` reproduce el protocolo de los runs oficiales. |

Las celdas que leen el piloto (sección 8) y la evaluación oficial en test (sección 15) se
omiten con un mensaje si esos runs no existen; sus resultados están en `docs/PILOTO.md` y
`docs/RESULTADOS.md`.

## MLflow UI

Desde la raíz del repositorio:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

y abre <http://127.0.0.1:5000>. Por defecto los runs se guardan en `mlflow.db` y los
artifacts en `mlruns/`, en la raíz. Si defines `MLFLOW_TRACKING_URI`, se usa ese backend.

## Qué archivo cambiar para cada cosa

| Quiero cambiar… | Archivo |
|---|---|
| Hiperparámetros por defecto (épocas, lr, batch, workers, semillas, rutas) | `src/galaxy3/config.py` (o flags de la CLI / celda de parámetros del notebook) |
| La arquitectura (capas, BatchNorm, Dropout) | `src/galaxy3/models.py` |
| Transformaciones, normalización, `DataLoader` | `src/galaxy3/data.py` |
| Selección de clases, split, exportación del dataset | `src/galaxy3/preparation.py` (+ `scripts/prepare_galaxy3.py`) |
| Loop de entrenamiento y evaluación (`fit`, `evaluate`) | `src/galaxy3/engine.py` |
| Optimizer, loss, qué se registra en MLflow, evaluación en test | `src/galaxy3/experiment.py` |
| Métricas (accuracy, F1, matriz de confusión, gap) | `src/galaxy3/metrics.py` |
| Figuras | `src/galaxy3/plots.py` |
| Feature maps (forward hooks) | `src/galaxy3/inspection.py` |
| Opciones de la CLI | `src/galaxy3/train.py` |
| Device y semillas | `src/galaxy3/utils.py` |
| Dependencias | `pyproject.toml`, y luego `uv lock` y `uv sync` |

Si cambias el código, corre `uv run pytest -q` y `uv run ruff check .`.

## Problemas frecuentes

- **`FileNotFoundError: ... dataset_stats.json`**: falta preparar los datos (sección
  *Datos*).
- **`ValueError: dataset_stats.json no coincide con la configuración`**: cambiaste
  `image_size` o `center_crop`. Regenera los datos con
  `uv run python scripts/prepare_galaxy3.py --overwrite --image-size ... --center-crop ...`.
- **El kernel no usa `.venv`** (falla la celda de ambiente o `import galaxy3`): abre
  Jupyter con `uv run jupyter lab` desde la raíz y elige el kernel *Python 3 (ipykernel)*.
  En VS Code, selecciona el intérprete `.venv`.
- **`uv` crea el ambiente con otro nombre**: tienes definida `UV_PROJECT_ENVIRONMENT` (por
  ejemplo, con direnv). Bórrala (`unset UV_PROJECT_ENVIRONMENT`) para usar `.venv`.
- **`torch.cuda.is_available()` es `False` con GPU NVIDIA**: casi siempre es el driver.
  Las ruedas CUDA 13.x necesitan un driver NVIDIA de la serie 580 o posterior. Revisa la
  versión con `nvidia-smi` (arriba a la derecha aparece la versión máxima de CUDA que
  soporta el driver: debe ser 13.0 o mayor) y actualízalo desde
  <https://www.nvidia.com/Download/index.aspx> o con el gestor de drivers de tu sistema
  (en Ubuntu, `sudo ubuntu-drivers install`, y reinicia). Si `nvidia-smi` no existe, el
  driver no está instalado. Mientras tanto el código corre en CPU.
- **El entrenamiento se queda detenido o hay errores de multiprocessing**: usa
  `NUM_WORKERS = 0` en el notebook o `--num-workers 0` en la CLI.
- **`CUDA out of memory`**: baja `--batch-size` (cambia el protocolo; no lo compares con
  los runs oficiales).
- **Va muy lento**: en CPU es normal; baja `EPOCHS` / `--epochs` para practicar. La
  primera época con workers tarda más porque arrancan los procesos.
- **`mlflow ui` no muestra runs**: ejecútalo desde la raíz con
  `--backend-store-uri sqlite:///mlflow.db`.

## Extensiones

Ideas fuera del Experimento 1 (ver la spec):

- Transfer learning con ResNet18: feature extraction y fine-tuning.
- `ReduceLROnPlateau` y early stopping, ambos monitoreando `val_loss`.
- Data augmentation.
- Repetir A, B y C con `--train-seed` 11, 22 y 33 para ver la variabilidad.
- Completar el diseño factorial: Dropout sin BatchNorm.
- Barridos de `learning_rate` y `dropout_p`; entrenar con una fracción de train.
- `CenterCrop` antes del `Resize` (requiere regenerar los datos).

## Pendientes

- Solo se verificó en Linux (Ubuntu, RTX 3060). Windows y macOS no se han probado.
- macOS Intel (x86_64) no está soportado: no hay ruedas de PyTorch 2.14 para esa
  plataforma. En macOS solo Apple Silicon.
- Las ruedas CUDA 13.2 exigen un driver NVIDIA reciente. No hay alternativa configurada
  para drivers antiguos (por ejemplo, otro índice `cu12x`).
- El notebook se versiona sin salidas. Se verificó completo con `TRAIN = True`,
  `EPOCHS = 1`, `NUM_WORKERS = 0` y MLflow/outputs temporales, solo en GPU (CUDA); no se
  ejecutó completo en CPU ni con las 40 épocas.
- `docs/SPEC_EXPERIMENTO_1.md` todavía describe el workspace original (ambiente compartido
  `mlops_unbosque`, rutas con prefijo `galaxy3/`).
- Los defaults de la CLI no son el protocolo oficial: `epochs = 30` (provisional en la
  spec) y el experimento MLflow `galaxy3-cnn-from-scratch`. Por eso los comandos de arriba
  pasan `--epochs 40` y `--mlflow-experiment`.
- `evaluate_on_test` no tiene comando de CLI; hay que llamarla desde Python.
- Con `num_workers > 0` en Linux, los workers usan `forkserver` para evitar el riesgo de
  deadlock de `fork`. La causa raíz del bloqueo observado no se identificó, y en la CLI la
  primera época tarda ~20 s más porque los workers vuelven a importar el módulo principal.
- No hay archivo de licencia.
