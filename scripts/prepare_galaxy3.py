"""Prepara Galaxy3 a partir de Galaxy10 DECaLS (se ejecuta una sola vez).

La lógica vive en ``galaxy3.preparation``; este script solo procesa argumentos y reporta.

Uso, desde la raíz del repositorio::

    uv run python scripts/prepare_galaxy3.py
    uv run python scripts/prepare_galaxy3.py --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from galaxy3.config import ExperimentConfig
from galaxy3.preparation import EXPECTED_COUNTS, prepare
from galaxy3.utils import PROJECT_ROOT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Argumentos CLI; los defaults salen de ``ExperimentConfig``."""
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Prepara Galaxy3 desde Galaxy10 DECaLS.")
    parser.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Galaxy10_DECals.h5"
    )
    parser.add_argument("--output", type=Path, default=defaults.data_root)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--center-crop", type=int, default=defaults.center_crop)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--overwrite", action="store_true", help="Regenera la salida si ya existe.")
    return parser.parse_args(argv)


def print_report(stats: dict[str, Any]) -> None:
    """Imprime conteos por split y clase, y las estadísticas de normalización."""
    print(f"split_seed = {stats['split_seed']}   total = {stats['total']}")
    table = pd.DataFrame(stats["counts"]).T
    print(table.assign(total=table.sum(axis=1)))
    print(f"mean_train = {stats['mean_train']}")
    print(f"std_train  = {stats['std_train']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point CLI."""
    args = parse_args(argv)
    if not args.source.is_file():
        raise SystemExit(f"No existe el archivo fuente: {args.source}")
    try:
        stats = prepare(
            source=args.source,
            output=args.output,
            split_seed=args.split_seed,
            image_size=args.image_size,
            center_crop=args.center_crop,
            overwrite=args.overwrite,
            expected_counts=EXPECTED_COUNTS,
            num_workers=args.num_workers,
        )
    except FileExistsError as err:
        raise SystemExit(f"Error: {err}") from err
    print_report(stats)


if __name__ == "__main__":
    main()
