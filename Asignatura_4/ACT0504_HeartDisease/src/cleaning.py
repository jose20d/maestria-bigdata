# src/cleaning.py
"""
Limpieza y preparación del dataset Heart Disease (Cleveland, UCI).

Pasos principales (alineados con la asignatura):
1. Revisión de calidad del dataset.
2. Revisión y conversión de tipos de datos.
3. Tratamiento de valores atípicos y reglas especiales.
4. Normalización / estandarización (opcional, configurable).
5. Feature engineering (variables categóricas derivadas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json
import pandas as pd

from .config import PROCESSED_DATA_DIR, CLEANING_CONFIG_FILE, VERBOSE

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"


# -----------------------------
#  Carga de configuración
# -----------------------------

def _load_cleaning_config() -> Dict[str, Any]:
    if not CLEANING_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {CLEANING_CONFIG_FILE}. "
            "Crea cleaning_config.json en la carpeta config."
        )
    with open(CLEANING_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
#  1. Revisión de calidad
# -----------------------------

def _drop_header_row_if_duplicated(df: pd.DataFrame) -> pd.DataFrame:
    first_row = df.iloc[0].astype(str).tolist()
    cols = df.columns.astype(str).tolist()
    if first_row == cols:
        if VERBOSE:
            print("[cleaning] Fila duplicada de cabecera detectada y eliminada.")
        df = df.iloc[1:, :].reset_index(drop=True)
    return df


def _drop_duplicates_if_needed(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not cfg.get("general", {}).get("drop_duplicates", True):
        return df
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if VERBOSE and removed > 0:
        print(f"[cleaning] Eliminadas {removed} filas duplicadas.")
    return df


# -----------------------------
#  2. Conversión de tipos
# -----------------------------

def _convert_dtypes(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not cfg.get("general", {}).get("convert_dtypes", True):
        return df

    df = df.copy()

    int_cols = [
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
    ]
    float_cols = ["oldpeak"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    return df


# -----------------------------
#  3. Rangos y valores permitidos
# -----------------------------

def _apply_ranges_and_allowed(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    cols_cfg = cfg.get("columns", {})

    for col, rules in cols_cfg.items():
        if col not in df.columns:
            continue

        strategy = rules.get("strategy", "drop")
        min_val = rules.get("min", None)
        max_val = rules.get("max", None)
        allowed_vals: List[Any] = rules.get("allowed_values", [])

        mask_invalid = pd.Series(False, index=df.index)

        if min_val is not None:
            mask_invalid |= df[col] < min_val
        if max_val is not None:
            mask_invalid |= df[col] > max_val

        if allowed_vals:
            mask_invalid |= (~df[col].isin(allowed_vals)) & df[col].notna()

        if not mask_invalid.any():
            continue

        if strategy == "drop":
            n = mask_invalid.sum()
            if VERBOSE:
                print(f"[cleaning] {col}: {n} filas eliminadas por valores fuera de rango/domino.")
            df = df.loc[~mask_invalid].copy()

        elif strategy == "set_na":
            n = mask_invalid.sum()
            if VERBOSE:
                print(f"[cleaning] {col}: {n} valores marcados como NA por estar fuera de rango/domino.")
            df.loc[mask_invalid, col] = pd.NA

    return df


def _maybe_rename_target(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    target_cfg = cfg.get("columns", {}).get("target", {})
    new_name = target_cfg.get("rename_to")
    if new_name and "target" in df.columns:
        if new_name not in df.columns:
            df[new_name] = df["target"]
            if VERBOSE:
                print(f"[cleaning] target renombrada internamente a '{new_name}'.")
    return df


# -----------------------------
#  4. Tratamiento de atípicos especiales
# -----------------------------

def _apply_special_rules(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    rules = cfg.get("special_rules", [])

    for rule in rules:
        name = rule.get("rule_name", "regla_sin_nombre")
        action = rule.get("action", "drop")

        mask = pd.Series(False, index=df.index)

        if "condition" in rule:
            cond = rule["condition"]
            mask = pd.Series(True, index=df.index)
            for col, val in cond.items():
                if col in df.columns:
                    mask &= df[col] == val
                else:
                    mask &= False

        elif "condition_or" in rule:
            for cond in rule["condition_or"]:
                local_mask = pd.Series(True, index=df.index)
                for col, val in cond.items():
                    if col in df.columns:
                        local_mask &= df[col] == val
                    else:
                        local_mask &= False
                mask |= local_mask

        elif "condition_and" in rule:
            cond_and = rule["condition_and"]
            mask = pd.Series(True, index=df.index)
            for col, val in cond_and.items():
                if col.endswith("_not"):
                    real_col = col[:-4]
                    if real_col in df.columns:
                        mask &= df[real_col] != val
                    else:
                        mask &= False
                else:
                    if col in df.columns:
                        mask &= df[col] == val
                    else:
                        mask &= False

        if not mask.any():
            continue

        if action == "drop":
            n = mask.sum()
            if VERBOSE:
                print(f"[cleaning] Regla especial '{name}': {n} fila(s) eliminadas.")
            df = df.loc[~mask].copy()

    return df


# -----------------------------
#  5. Normalización (opcional)
# -----------------------------

def _apply_normalization(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    norm_cfg = cfg.get("normalization", {})
    if not norm_cfg.get("apply", False):
        if VERBOSE:
            print("[cleaning] Normalización desactivada en la configuración.")
        return df

    df = df.copy()
    method = norm_cfg.get("method", "zscore")
    cols = [c for c in norm_cfg.get("columns", []) if c in df.columns]

    if method == "zscore":
        for col in cols:
            series = df[col].astype("Float64")
            mean = series.mean()
            std = series.std()
            if pd.isna(std) or std == 0:
                continue
            df[f"{col}_zscore"] = (series - mean) / std
            if VERBOSE:
                print(f"[cleaning] Columna {col}_zscore creada (z-score).")
    else:
        if VERBOSE:
            print(f"[cleaning] Método de normalización '{method}' no implementado.")

    return df


# -----------------------------
#  6. Feature engineering
# -----------------------------

def _add_feature_engineering(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    if "sex" in df.columns:
        df["sex_cat"] = df["sex"].map({0: "Femenino", 1: "Masculino"}).astype("string")

    if "cp" in df.columns:
        cp_map = {
            0: "Angina típica",
            1: "Angina atípica",
            2: "Dolor no anginal",
            3: "Asintomático",
        }
        df["cp_cat"] = df["cp"].map(cp_map).astype("string")

    if "condition" in df.columns:
        df["condition_cat"] = df["condition"].map(
            {0: "Saludable", 1: "Enfermedad cardiaca"}
        ).astype("string")

    fe_cfg = cfg.get("feature_engineering", {})
    age_group_cfg = fe_cfg.get("age_group", {})
    if "age" in df.columns and age_group_cfg:
        bins = age_group_cfg.get("bins")
        labels = age_group_cfg.get("labels")
        if bins and labels and len(bins) - 1 == len(labels):
            df["age_group"] = pd.cut(
                df["age"].astype("Float64"),
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=True,
            ).astype("string")

    return df


# -----------------------------
#  7. Ajuste final de dtypes
# -----------------------------

def _cast_nullable_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")
    return df


# -----------------------------
#  API principal
# -----------------------------

def clean_data(df_raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    global VERBOSE
    VERBOSE = verbose

    cfg = _load_cleaning_config()

    if VERBOSE:
        print(f"[cleaning] Inicio de limpieza. Shape inicial: {df_raw.shape}")

    df = df_raw.copy()

    df = _drop_header_row_if_duplicated(df)
    df = _drop_duplicates_if_needed(df, cfg)
    df = _convert_dtypes(df, cfg)

    if VERBOSE:
        total_nulls = df.isna().sum().sum()
        print(f"[cleaning] Valores nulos tras conversión de tipos: {total_nulls}")

    df = _apply_ranges_and_allowed(df, cfg)
    df = _maybe_rename_target(df, cfg)
    df = _apply_special_rules(df, cfg)
    df = _apply_normalization(df, cfg)
    df = _add_feature_engineering(df, cfg)
    df = _cast_nullable_categoricals(df)

    if VERBOSE:
        print(f"[cleaning] Shape final tras limpieza: {df.shape}")
        print(f"[cleaning] Nulos totales tras limpieza: {df.isna().sum().sum()}")

    return df


def save_clean_data(df_clean: pd.DataFrame, path: Path | None = None) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = path or PROCESSED_DATA_FILE
    df_clean.to_csv(output_path, index=False, na_rep="NA")
    if VERBOSE:
        print(f"[cleaning] Dataset limpio guardado en: {output_path.resolve()}")
    return output_path
