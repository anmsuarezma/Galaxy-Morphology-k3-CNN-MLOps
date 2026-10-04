"""Frontera de MLflow y resolución del tracking URI.

MLflow solo puede usarse en ``experiment.py``: el resto del paquete (en particular
``engine``, ``metrics`` y ``plots``) no lo importa ni directa ni indirectamente.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import galaxy3
from galaxy3.experiment import LOCAL_TRACKING_URI, _artifact_location, resolve_tracking_uri
from galaxy3.utils import PROJECT_ROOT

PACKAGE_DIR = Path(galaxy3.__file__).parent
MLFLOW_MODULE = "experiment.py"
# Entry points que dependen de experiment.py por diseño (y por eso cargan MLflow).
ENTRY_POINTS = {"train.py"}


def _imported_roots(path: Path) -> set[str]:
    """Paquetes raíz importados por un módulo (``import a.b`` y ``from a.b import c`` -> ``a``)."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_only_experiment_imports_mlflow():
    importers = {
        path.name for path in PACKAGE_DIR.glob("*.py") if "mlflow" in _imported_roots(path)
    }
    assert importers == {MLFLOW_MODULE}


def test_core_modules_do_not_load_mlflow_transitively():
    """Importar el núcleo en un proceso limpio no carga MLflow (ni de forma indirecta)."""
    modules = [
        path.stem
        for path in PACKAGE_DIR.glob("*.py")
        if path.name not in {MLFLOW_MODULE, *ENTRY_POINTS}
    ]
    assert {"engine", "metrics", "plots"} <= set(modules)
    imports = "; ".join(f"import galaxy3.{name}" for name in modules)
    code = f"import sys; {imports}; assert 'mlflow' not in sys.modules, 'mlflow cargado'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tracking_uri_comes_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert resolve_tracking_uri() == "http://127.0.0.1:5000"


def test_tracking_uri_falls_back_to_local_sqlite_in_galaxy3(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    expected = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    assert resolve_tracking_uri() == LOCAL_TRACKING_URI == expected


def test_tests_use_temporary_backend(isolated_mlflow_backend):
    uri = resolve_tracking_uri()
    assert uri != LOCAL_TRACKING_URI
    assert uri == f"sqlite:///{isolated_mlflow_backend.as_posix()}"


def test_sqlite_artifacts_live_next_to_database(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    assert _artifact_location(uri) == (tmp_path / "mlruns").resolve().as_uri()


@pytest.mark.parametrize("uri", ["http://127.0.0.1:5000", "databricks"])
def test_remote_backends_keep_server_artifact_root(uri):
    assert _artifact_location(uri) is None
