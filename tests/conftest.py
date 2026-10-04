"""Configuración común de los tests.

Los tests nunca escriben en el backend real de MLflow (``galaxy3/mlflow.db``): durante
toda la sesión ``MLFLOW_TRACKING_URI`` apunta a un SQLite temporal, y los artifacts van a
``mlruns/`` junto a él. Una sola base por sesión evita repetir las migraciones de MLflow
en cada test.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_mlflow_backend(tmp_path_factory):
    """Apunta MLflow a ``sqlite:///<tmp>/mlflow.db`` durante toda la sesión de tests."""
    database = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    patcher = pytest.MonkeyPatch()
    patcher.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    yield database
    patcher.undo()
