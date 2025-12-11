# src/config.py
"""
Archivo de configuración central del proyecto.
Define rutas, nombres de archivo y parámetros globales utilizados por
los distintos módulos del pipeline.
"""

from __future__ import annotations
from pathlib import Path

# -----------------------------
#  RUTAS BASE DEL PROYECTO
# -----------------------------

# Carpeta raíz del proyecto (dos niveles arriba desde src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Carpeta donde se almacenarán todos los datos
DATA_DIR = PROJECT_ROOT / "data"

# Subcarpetas estructuradas
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Carpeta para outputs generados
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# -----------------------------
#  ARCHIVOS DE DATOS
# -----------------------------

# Archivo donde se guardará el dataset crudo descargado/cargado
RAW_DATA_FILE = RAW_DATA_DIR / "heart.csv"

# Archivo donde se guardará el dataset limpio generado por el pipeline
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"

# -----------------------------
#  PARÁMETROS DEL DATASET
# -----------------------------

# Fuente de descarga desde Kaggle
KAGGLE_DATASET_SLUG = "cherngs/heart-disease-cleveland-uci"

# Columnas del dataset original Cleveland (en el orden exacto del CSV)
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# -----------------------------
#  CONFIGURACIÓN GENERAL
# -----------------------------

# Modo verboso global (si se requiere para debug)
VERBOSE = True

# Archivo de configuración de reglas de limpieza
CLEANING_CONFIG_FILE = PROJECT_ROOT / "config" / "cleaning_config.json"
