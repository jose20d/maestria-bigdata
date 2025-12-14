# src/eda_bivariate.py
"""
EDA bivariante:
- Numérico–Numérico: correlaciones de Pearson y Spearman + gráficos de dispersión
- Categórico–Categórico: tablas de contingencia + porcentajes por fila + barras apiladas + resumen chi2 (nivel EDA)
- Numérico–Categórico: estadísticas descriptivas por grupo + boxplots agrupados

Salidas:
- output/reports/bivariate_numeric_numeric_correlations.csv
- output/reports/bivariate_cat_cat_summary.csv
- output/reports/bivariate_contingency/{pair}_contingency.csv
- output/reports/bivariate_contingency/{pair}_crosstab_percent.csv
- output/reports/bivariate_numeric_categorical_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from .config import OUTPUT_DIR
from .plotting import save_scatter, save_stacked_bar, save_grouped_boxplot


# Pares fijos (acordados)
NUM_NUM_PAIRS: List[Tuple[str, str]] = [
    ("age", "thalach"),
    ("trestbps", "chol"),
]

CAT_CAT_PAIRS: List[Tuple[str, str]] = [
    ("sex", "condition"),
    ("cp", "condition"),
]

NUM_CAT_PAIRS: List[Tuple[str, str]] = [
    ("thalach", "condition"),
    ("chol", "condition"),
]


REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

BIV_NUM_NUM_CSV = REPORTS_DIR / "bivariate_numeric_numeric_correlations.csv"
BIV_CAT_CAT_CSV = REPORTS_DIR / "bivariate_cat_cat_summary.csv"
BIV_NUM_CAT_CSV = REPORTS_DIR / "bivariate_numeric_categorical_summary.csv"

CONTINGENCY_DIR = REPORTS_DIR / "bivariate_contingency"

PLOTS_NUM_NUM_DIR = PLOTS_DIR / "bivariate" / "num_num"
PLOTS_CAT_CAT_DIR = PLOTS_DIR / "bivariate" / "cat_cat"
PLOTS_NUM_CAT_DIR = PLOTS_DIR / "bivariate" / "num_cat"


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONTINGENCY_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_NUM_NUM_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_CAT_CAT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_NUM_CAT_DIR.mkdir(parents=True, exist_ok=True)


def _pair_name(x: str, y: str) -> str:
    return f"{x}_vs_{y}"


# -----------------------------
# Numérico–Numérico
# -----------------------------

def build_num_num_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, y in NUM_NUM_PAIRS:
        if x not in df.columns or y not in df.columns:
            continue

        xs = pd.to_numeric(df[x], errors="coerce")
        ys = pd.to_numeric(df[y], errors="coerce")
        mask = xs.notna() & ys.notna()

        n = int(mask.sum())
        pair = _pair_name(x, y)

        if n < 3:
            rows.append(
                {
                    "pair": pair,
                    "x": x,
                    "y": y,
                    "pearson_r": pd.NA,
                    "pearson_p": pd.NA,
                    "spearman_r": pd.NA,
                    "spearman_p": pd.NA,
                    "n": n,
                }
            )
            continue

        pr, pp = pearsonr(xs[mask], ys[mask])
        sr, sp = spearmanr(xs[mask], ys[mask])

        rows.append(
            {
                "pair": pair,
                "x": x,
                "y": y,
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "n": n,
            }
        )

    return pd.DataFrame(rows)


def generate_num_num_plots(df: pd.DataFrame) -> None:
    for x, y in NUM_NUM_PAIRS:
        if x not in df.columns or y not in df.columns:
            continue
        pair = _pair_name(x, y)
        save_scatter(df, x, y, PLOTS_NUM_NUM_DIR / f"{pair}_scatter.png")


# -----------------------------
# Categórico–Categórico
# -----------------------------

def _crosstab_with_na(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    a = df[x].astype("object")
    b = df[y].astype("object")
    return pd.crosstab(a, b, dropna=False)


def _row_percent(ct: pd.DataFrame) -> pd.DataFrame:
    denom = ct.sum(axis=1).replace(0, pd.NA)
    return ct.div(denom, axis=0) * 100.0


def build_cat_cat_summary_and_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, pd.DataFrame, pd.DataFrame]]]:
    """
    Devuelve:
      - dataframe de resumen (chi2/p/grados_de_libertad/n por par)
      - lista de (nombre_del_par, tabla_de_contingencia, tabla_de_porcentaje_por_fila)
    """
    summary_rows = []
    tables = []

    for x, y in CAT_CAT_PAIRS:
        if x not in df.columns or y not in df.columns:
            continue

        pair = _pair_name(x, y)
        ct = _crosstab_with_na(df, x, y)

        n = int(ct.values.sum())
        if ct.shape[0] == 0 or ct.shape[1] == 0 or n == 0:
            chi2 = pd.NA
            p = pd.NA
            dof = pd.NA
        else:
            chi2_val, p_val, dof_val, _ = chi2_contingency(ct.fillna(0))
            chi2, p, dof = float(chi2_val), float(p_val), int(dof_val)

        pct = _row_percent(ct)

        summary_rows.append(
            {
                "pair": pair,
                "x": x,
                "y": y,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "n": n,
            }
        )

        tables.append((pair, ct, pct))

    return pd.DataFrame(summary_rows), tables


def save_cat_cat_tables(tables: List[Tuple[str, pd.DataFrame, pd.DataFrame]]) -> None:
    for pair, ct, pct in tables:
        ct_path = CONTINGENCY_DIR / f"{pair}_contingency.csv"
        pct_path = CONTINGENCY_DIR / f"{pair}_crosstab_percent.csv"
        ct.to_csv(ct_path)
        pct.to_csv(pct_path)


def generate_cat_cat_plots(df: pd.DataFrame) -> None:
    for x, y in CAT_CAT_PAIRS:
        if x not in df.columns or y not in df.columns:
            continue
        pair = _pair_name(x, y)
        save_stacked_bar(df, x, y, PLOTS_CAT_CAT_DIR / f"{pair}_stacked_bar.png", percent=True)


# -----------------------------
# Numérico–Categórico
# -----------------------------

def build_num_cat_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for num, grp in NUM_CAT_PAIRS:
        if num not in df.columns or grp not in df.columns:
            continue

        x = pd.to_numeric(df[num], errors="coerce")
        g = df[grp].astype("object")
        tmp = pd.DataFrame({num: x, grp: g}).dropna()

        pair = f"{num}_by_{grp}"

        for group_value, sub in tmp.groupby(grp):
            vals = sub[num].dropna()
            rows.append(
                {
                    "pair": pair,
                    "numeric_var": num,
                    "group_var": grp,
                    "group": "NA" if pd.isna(group_value) else str(group_value),
                    "n": int(vals.shape[0]),
                    "mean": float(vals.mean()) if len(vals) else pd.NA,
                    "median": float(vals.median()) if len(vals) else pd.NA,
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
            )

    return pd.DataFrame(rows)


def generate_num_cat_plots(df: pd.DataFrame) -> None:
    for num, grp in NUM_CAT_PAIRS:
        if num not in df.columns or grp not in df.columns:
            continue
        pair = f"{num}_by_{grp}"
        save_grouped_boxplot(df, num, grp, PLOTS_NUM_CAT_DIR / f"{pair}_box.png")


# -----------------------------
# Ejecución
# -----------------------------

def run_bivariate_eda(df: pd.DataFrame) -> Dict[str, str]:
    """
    Ejecuta el EDA bivariante completo (tablas + gráficos). Devuelve un dict con rutas de salida.
    """
    _ensure_dirs()

    # numérico–numérico
    num_num = build_num_num_correlations(df)
    num_num.to_csv(BIV_NUM_NUM_CSV, index=False)
    generate_num_num_plots(df)

    # categórico–categórico
    cat_cat_summary, cat_tables = build_cat_cat_summary_and_tables(df)
    cat_cat_summary.to_csv(BIV_CAT_CAT_CSV, index=False)
    save_cat_cat_tables(cat_tables)
    generate_cat_cat_plots(df)

    # numérico–categórico
    num_cat = build_num_cat_summary(df)
    num_cat.to_csv(BIV_NUM_CAT_CSV, index=False)
    generate_num_cat_plots(df)

    return {
        "bivariate_numeric_numeric_correlations_csv": str(BIV_NUM_NUM_CSV),
        "bivariate_cat_cat_summary_csv": str(BIV_CAT_CAT_CSV),
        "bivariate_numeric_categorical_summary_csv": str(BIV_NUM_CAT_CSV),
        "bivariate_contingency_dir": str(CONTINGENCY_DIR),
        "bivariate_num_num_plots_dir": str(PLOTS_NUM_NUM_DIR),
        "bivariate_cat_cat_plots_dir": str(PLOTS_CAT_CAT_DIR),
        "bivariate_num_cat_plots_dir": str(PLOTS_NUM_CAT_DIR),
    }
