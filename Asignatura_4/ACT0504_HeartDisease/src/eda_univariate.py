# src/eda_univariate.py
"""
EDA univariante:
- Tabla de resumen numérico (tendencia central + dispersión + asimetría + conteo de atípicos)
- Tabla de frecuencias categóricas (conteo + porcentaje)
- Gráficos univariantes guardados en output/plots/

Salidas:
- output/reports/univariate_numeric_summary.csv
- output/reports/univariate_categorical_frequencies.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    OUTPUT_DIR,
)
from .plotting import (
    save_histogram,
    save_boxplot,
    save_density,
    save_bar_counts,
)


# Listas de variables fijas (acordadas)
NUMERIC_VARS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_VARS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "condition"]


# Rutas de salida (se mantienen aquí para evitar desalineación; pueden moverse a config.py luego)
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

UNIV_NUM_CSV = REPORTS_DIR / "univariate_numeric_summary.csv"
UNIV_CAT_CSV = REPORTS_DIR / "univariate_categorical_frequencies.csv"

UNIV_NUM_PLOTS_DIR = PLOTS_DIR / "univariate" / "numeric"
UNIV_CAT_PLOTS_DIR = PLOTS_DIR / "univariate" / "categorical"


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    UNIV_NUM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    UNIV_CAT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _iqr_outlier_count(s: pd.Series) -> int:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return 0
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return 0
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((x < low) | (x > high)).sum())


def _mode_value(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return pd.NA
    m = x.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA


def _skewness(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 3:
        return pd.NA
    return x.skew()


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in NUMERIC_VARS:
        if var not in df.columns:
            continue

        x = pd.to_numeric(df[var], errors="coerce")
        n_total = len(x)
        n_missing = int(x.isna().sum())
        x_non = x.dropna()

        if x_non.empty:
            rows.append(
                {
                    "variable": var,
                    "n": n_total,
                    "missing": n_missing,
                    "mean": pd.NA,
                    "median": pd.NA,
                    "mode": pd.NA,
                    "variance": pd.NA,
                    "std": pd.NA,
                    "iqr": pd.NA,
                    "q1": pd.NA,
                    "q3": pd.NA,
                    "min": pd.NA,
                    "max": pd.NA,
                    "skewness": pd.NA,
                    "outliers_iqr_count": 0,
                }
            )
            continue

        q1 = x_non.quantile(0.25)
        q3 = x_non.quantile(0.75)
        iqr = q3 - q1

        rows.append(
            {
                "variable": var,
                "n": n_total,
                "missing": n_missing,
                "mean": float(x_non.mean()),
                "median": float(x_non.median()),
                "mode": _mode_value(df[var]),
                "variance": float(x_non.var(ddof=1)) if len(x_non) > 1 else 0.0,
                "std": float(x_non.std(ddof=1)) if len(x_non) > 1 else 0.0,
                "iqr": float(iqr),
                "q1": float(q1),
                "q3": float(q3),
                "min": float(x_non.min()),
                "max": float(x_non.max()),
                "skewness": _skewness(df[var]),
                "outliers_iqr_count": _iqr_outlier_count(df[var]),
            }
        )

    return pd.DataFrame(rows)


def build_categorical_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for var in CATEGORICAL_VARS:
        if var not in df.columns:
            continue

        s = df[var].astype("object")
        counts = s.value_counts(dropna=False)
        total = counts.sum() if counts.sum() != 0 else 1

        for cat, cnt in counts.items():
            out_rows.append(
                {
                    "variable": var,
                    "category": "NA" if pd.isna(cat) else str(cat),
                    "count": int(cnt),
                    "percent": float(cnt) / float(total) * 100.0,
                }
            )

    return pd.DataFrame(out_rows)


def generate_univariate_plots(df: pd.DataFrame) -> None:
    # Gráficos numéricos
    for var in NUMERIC_VARS:
        if var not in df.columns:
            continue
        series = df[var]
        save_histogram(series, UNIV_NUM_PLOTS_DIR / f"{var}_hist.png")
        save_boxplot(series, UNIV_NUM_PLOTS_DIR / f"{var}_box.png")
        save_density(series, UNIV_NUM_PLOTS_DIR / f"{var}_density.png")

    # Gráficos categóricos (conteos)
    for var in CATEGORICAL_VARS:
        if var not in df.columns:
            continue
        series = df[var]
        save_bar_counts(series, UNIV_CAT_PLOTS_DIR / f"{var}_bar.png", normalize=False)


def run_univariate_eda(df: pd.DataFrame) -> Dict[str, str]:
    """
    Ejecuta el EDA univariante completo (tablas + gráficos). Devuelve un dict con rutas de salida.
    """
    _ensure_dirs()

    num_summary = build_numeric_summary(df)
    cat_freq = build_categorical_frequencies(df)

    num_summary.to_csv(UNIV_NUM_CSV, index=False)
    cat_freq.to_csv(UNIV_CAT_CSV, index=False)

    generate_univariate_plots(df)

    return {
        "univariate_numeric_summary_csv": str(UNIV_NUM_CSV),
        "univariate_categorical_frequencies_csv": str(UNIV_CAT_CSV),
        "univariate_numeric_plots_dir": str(UNIV_NUM_PLOTS_DIR),
        "univariate_categorical_plots_dir": str(UNIV_CAT_PLOTS_DIR),
    }
