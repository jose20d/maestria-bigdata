# src/data_loading.py
"""
Módulo de carga de datos.

Se encarga de:
- Crear la estructura de directorios necesaria.
- Descargar el dataset desde Kaggle si no existe copia local.
- Cargar el dataset crudo en un DataFrame de pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import kagglehub
import pandas as pd

from .config import (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    RAW_DATA_FILE,
    KAGGLE_DATASET_SLUG,
    COLUMN_NAMES,
    VERBOSE,
)


def _log(message: str) -> None:
    """
    Escribe mensajes de logging sencillos cuando VERBOSE está activado.
    """
    if VERBOSE:
        print(f"[data_loading] {message}")


def ensure_directories() -> None:
    """
    Crea la estructura de directorios necesaria para datos y resultados.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    _log("Directorios verificados/creados.")


def _is_file_empty(path: Path) -> bool:
    """
    Indica si un archivo existe pero está vacío.
    """
    return path.exists() and path.stat().st_size == 0


def _download_from_kaggle() -> pd.DataFrame:
    """
    Descarga el dataset desde Kaggle utilizando kagglehub y devuelve
    un DataFrame con las columnas definidas en la configuración.
    """
    _log(f"Descargando dataset desde Kaggle: {KAGGLE_DATASET_SLUG}")
    dataset_path_str = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    dataset_path = Path(dataset_path_str)

    csv_files: List[Path] = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"No se encontraron archivos CSV en el dataset descargado: {dataset_path}"
        )

    csv_path = csv_files[0]
    _log(f"Archivo CSV detectado en: {csv_path}")

    df = pd.read_csv(
        csv_path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?", "NA", "NaN"],
    )
    _log(f"Datos descargados desde Kaggle con shape {df.shape}")
    return df


def get_raw_data(force_redownload: bool = False) -> pd.DataFrame:
    """
    Obtiene el DataFrame crudo del dataset.

    Si existe un archivo CSV local no vacío y no se fuerza la redescarga,
    se carga directamente desde data/raw. En caso contrario, se descarga
    desde Kaggle y se guarda en data/raw para usos posteriores.
    """
    ensure_directories()

    has_valid_local_file = RAW_DATA_FILE.exists() and not _is_file_empty(RAW_DATA_FILE)

    if has_valid_local_file and not force_redownload:
        _log(f"Cargando datos crudos desde archivo local: {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
        _log(f"Datos cargados desde archivo local con shape {df.shape}")
        return df

    _log("No existe archivo local válido o se forzó la redescarga. "
         "Iniciando descarga desde Kaggle.")
    df = _download_from_kaggle()

    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    _log(f"Datos crudos guardados en {RAW_DATA_FILE}")

    return df
