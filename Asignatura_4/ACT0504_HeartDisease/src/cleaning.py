# src/cleaning.py
"""
Módulo de limpieza y preparación de datos.

Incluye:
- Limpieza configurable por bandas y estrategia (drop, imputación, NA).
- Reglas específicas de outliers acordadas por el grupo.
- Conversión de columnas numéricas a tipos nullable.
- Creación de variables categóricas derivadas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .config import PROCESSED_DATA_FILE, CLEANING_CONFIG_FILE


NUMERIC_COLUMNS = [
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


def _load_cleaning_config(path: Path = CLEANING_CONFIG_FILE) -> Dict[str, Any]:
    """
    Carga el archivo de configuración de limpieza en formato JSON.
    """
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas numéricas a tipo numérico permitiendo valores faltantes.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _apply_band_rules(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Aplica reglas de bandas por columna según el archivo de configuración.
    """
    df = df.copy()
    columns_cfg = config.get("columns", {})

    for col, rules in columns_cfg.items():
        if col not in df.columns:
            continue

        min_val = rules.get("min", None)
        max_val = rules.get("max", None)
        strategy = rules.get("strategy", "drop")

        mask_out = pd.Series(False, index=df.index)

        if min_val is not None:
            mask_out |= df[col] < min_val
        if max_val is not None:
            mask_out |= df[col] > max_val

        if not mask_out.any():
            continue

        if strategy == "drop":
            df = df[~mask_out].copy()

        elif strategy == "set_na":
            df.loc[mask_out, col] = pd.NA

        elif strategy == "impute_mean":
            mean_val = df.loc[~mask_out, col].mean()
            df.loc[mask_out, col] = mean_val

        elif strategy == "impute_median":
            median_val = df.loc[~mask_out, col].median()
            df.loc[mask_out, col] = median_val

        elif strategy == "impute_mode":
            mode_series = df.loc[~mask_out, col].mode()
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                df.loc[mask_out, col] = mode_val
        else:
            raise ValueError(f"Estrategia de limpieza no reconocida: {strategy}")

    return df


def _apply_specific_outlier_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas específicas acordadas para ciertos patrones de outliers.
    """
    df = df.copy()

    if df["target"].nunique() > 2:
        df["condition"] = (df["target"] > 0).astype(int)
    else:
        df["condition"] = df["target"].astype(int)

    mask_chol_564 = ~((df["chol"] == 564) & (df["condition"] == 0))

    mask_oldpeak_extremos = ~(
        df["condition"].eq(0) & df["oldpeak"].isin([3.5, 4.2])
    )

    mask_inconsistente = ~(
        (df["condition"] == 0)
        & (df["thal"] == 2)
        & (df["restecg"] == 2)
        & (df["cp"] != 3)
    )

    mask_total = mask_chol_564 & mask_oldpeak_extremos & mask_inconsistente
    df_clean = df[mask_total].copy()
    return df_clean


def _add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables categóricas derivadas a partir de las variables numéricas.
    """
    df = df.copy()

    df["sexo_categorico"] = df["sex"].map({0: "Femenino", 1: "Masculino"})

    df["dolor_pecho_categorico"] = df["cp"].map(
        {
            0: "Anginal tipico",
            1: "Anginal atipico",
            2: "Dolor no anginal",
            3: "Asintomatico",
        }
    )

    df["glucosa_alta_categorico"] = df["fbs"].map({0: "No", 1: "Si"})

    df["electrocardiograma_categorico"] = df["restecg"].map(
        {
            0: "Normal",
            1: "Anomalia ST-T",
            2: "Hipertrofia ventricular",
        }
    )

    df["angina_ejercicio_categorico"] = df["exang"].map({0: "No", 1: "Si"})

    df["pendiente_categorico"] = df["slope"].map(
        {
            0: "Creciente",
            1: "Neutra",
            2: "Decreciente",
        }
    )

    df["corazon_categorico"] = df["thal"].map(
        {
            0: "Normal",
            1: "Defecto arreglado",
            2: "Defecto arreglable",
        }
    )

    df["condicion_categorico"] = df["condition"].map(
        {
            0: "Saludable",
            1: "Enfermo",
        }
    )

    return df


def _cast_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta los tipos de datos para usar dtypes con soporte de valores faltantes.
    """
    df = df.copy()

    int_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
        "condition",
    ]

    float_columns = [
        "oldpeak",
    ]

    categorical_string_columns = [
        "sexo_categorico",
        "dolor_pecho_categorico",
        "glucosa_alta_categorico",
        "electrocardiograma_categorico",
        "angina_ejercicio_categorico",
        "pendiente_categorico",
        "corazon_categorico",
        "condicion_categorico",
    ]

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype("Float64")

    for col in categorical_string_columns:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline de limpieza completo:
    - eliminación de posibles filas de encabezado mal cargadas,
    - conversión de columnas numéricas,
    - reglas configurables por bandas,
    - reglas específicas de outliers,
    - generación de variables categóricas,
    - ajuste de tipos a dtypes nullable.
    """
    df = df_raw.copy()

    if isinstance(df["age"].iloc[0], str) and df["age"].iloc[0].lower() == "age":
        df = df.iloc[1:].reset_index(drop=True)

    df = _coerce_numeric_columns(df)

    config = _load_cleaning_config()
    df = _apply_band_rules(df, config)

    df = _apply_specific_outlier_rules(df)

    df = _add_categorical_features(df)

    df = _cast_nullable_dtypes(df)

    return df


def save_clean_data(df_clean: pd.DataFrame) -> None:
    """
    Guarda el dataset limpio en la ruta configurada.
    """
    df_clean.to_csv(PROCESSED_DATA_FILE, index=False)
