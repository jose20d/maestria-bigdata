# src/hypothesis_tests.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import REPORTS_DIR


def run_hypothesis_tests(df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, str]:
    """
    Ejecuta:
    - Chi-cuadrado: independencia entre dos variables categóricas.
    - t-test o ANOVA: comparación de medias de una variable numérica entre grupos (condition).

    Genera: output/reports/hypothesis_tests.md
    Devuelve: dict con rutas de salida para el pipeline/metadata.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = REPORTS_DIR / "hypothesis_tests.md"
    # CSVs requeridos por el PDF ejecutivo (report_builder.py)
    out_chi_csv = REPORTS_DIR / "chi_square_cp_vs_condition_contingency.csv"
    out_ttest_csv = REPORTS_DIR / "ttest_chol_by_condition_summary.csv"
    out_results_json = REPORTS_DIR / "hypothesis_results.json"

    lines: list[str] = []
    lines.append("# Contraste de hipótesis")
    lines.append("")
    lines.append(f"Nivel de significación (α): **{alpha:.2f}**")
    lines.append("")

    # ============
    # 1) Chi-cuadrado (categórica vs categórica)
    # Elegimos: cp vs condition (alineado con el informe ejecutivo)
    # ============
    chi_var_a = "cp"
    chi_var_b = "condition" if "condition" in df.columns else "target"

    if chi_var_a in df.columns and chi_var_b in df.columns:
        # CSV: tabla de contingencia (para PDF)
        _write_chi_square_contingency_csv(df, chi_var_a, chi_var_b, out_chi_csv)
        chi_stats = _compute_chi_square_stats(df, chi_var_a, chi_var_b)
        lines.extend(_chi_square_section(df, chi_var_a, chi_var_b, alpha))
    else:
        chi_stats = None
        lines.append("## 1) Prueba Chi-cuadrado (independencia)")
        lines.append("")
        lines.append(f"No se pudo ejecutar: faltan columnas `{chi_var_a}` y/o `{chi_var_b}`.")
        lines.append("")

    # ============
    # 2) t-test o ANOVA (numérica vs categórica)
    # Elegimos: chol por condition (alineado con el informe ejecutivo)
    # ============
    group_col = chi_var_b
    num_col = "chol" if "chol" in df.columns else None

    if group_col in df.columns and num_col is not None:
        # CSV: resumen por grupo (para PDF)
        _write_ttest_group_summary_csv(df, numeric_col=num_col, group_col=group_col, out_csv=out_ttest_csv)
        mean_stats = _compute_mean_comparison_stats(df, numeric_col=num_col, group_col=group_col)
        lines.extend(_mean_comparison_section(df, num_col, group_col, alpha))
    else:
        mean_stats = None
        lines.append("## 2) Comparación de medias (t-test / ANOVA)")
        lines.append("")
        lines.append("No se pudo ejecutar por falta de columnas necesarias.")
        lines.append("")

    results_payload: Dict[str, Any] = {
        "alpha": float(alpha),
        "chi_square": chi_stats,
        "mean_comparison": mean_stats,
    }
    out_results_json.write_text(
        json.dumps(results_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    outputs: Dict[str, str] = {"hypothesis_md": str(out_md)}
    if out_chi_csv.exists():
        outputs["chi_square_cp_vs_condition_contingency_csv"] = str(out_chi_csv)
    if out_ttest_csv.exists():
        outputs["ttest_chol_by_condition_summary_csv"] = str(out_ttest_csv)
    outputs["hypothesis_results_json"] = str(out_results_json)
    return outputs


def _write_chi_square_contingency_csv(df: pd.DataFrame, a: str, b: str, out_csv: Path) -> None:
    """
    Escribe una tabla de contingencia para el PDF (CSV plano, sin index "Unnamed: 0").
    """
    work = df[[a, b]].dropna().copy()
    # Forzamos categorías discretas (por si vienen como float/int nullable)
    work[a] = work[a].astype("Int64")
    work[b] = work[b].astype("Int64")

    table = pd.crosstab(work[a], work[b]).reset_index()
    table.to_csv(out_csv, index=False)


def _write_ttest_group_summary_csv(df: pd.DataFrame, numeric_col: str, group_col: str, out_csv: Path) -> None:
    """
    Escribe un resumen por grupo (n, media, mediana, std, min, max) para el PDF.
    Nota: el PDF muestra esta tabla bajo "t-test / ANOVA", aunque aquí solo guardamos el resumen.
    """
    work = df[[numeric_col, group_col]].copy()
    work[numeric_col] = pd.to_numeric(work[numeric_col], errors="coerce")
    work[group_col] = pd.to_numeric(work[group_col], errors="coerce")
    work = work.dropna(subset=[numeric_col, group_col])

    # grupos discretos
    work[group_col] = work[group_col].astype("Int64")

    summary = (
        work.groupby(group_col)[numeric_col]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    summary.to_csv(out_csv, index=False)


def _decision(p_value: float, alpha: float) -> str:
    if np.isnan(p_value):
        return "No aplicable"
    return "Rechazar H0" if p_value < alpha else "No rechazar H0"


def _compute_chi_square_stats(df: pd.DataFrame, a: str, b: str) -> Dict[str, Any]:
    """
    Calcula estadísticos de Chi-cuadrado para dos variables categóricas.
    """
    work = df[[a, b]].dropna().copy()
    work[a] = work[a].astype("Int64").astype("string")
    work[b] = work[b].astype("Int64").astype("string")
    table = pd.crosstab(work[a], work[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "variables": {"a": a, "b": b},
            "chi2": float("nan"),
            "p_value": float("nan"),
            "dof": 0,
            "n": int(table.to_numpy().sum()),
        }
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "variables": {"a": a, "b": b},
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "n": int(table.to_numpy().sum()),
    }


def _compute_mean_comparison_stats(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
    """
    Calcula t-test (Welch) si hay 2 grupos; ANOVA si hay 3+.
    """
    alpha_default = 0.05
    work = df[[numeric_col, group_col]].dropna().copy()
    work[numeric_col] = pd.to_numeric(work[numeric_col], errors="coerce")
    work[group_col] = pd.to_numeric(work[group_col], errors="coerce")
    work = work.dropna(subset=[numeric_col, group_col])

    groups = sorted(work[group_col].astype("Int64").dropna().unique().tolist())
    samples = [work.loc[work[group_col].astype("Int64") == g, numeric_col].to_numpy() for g in groups]
    samples = [s[np.isfinite(s)] for s in samples]

    payload: Dict[str, Any] = {
        "numeric_col": numeric_col,
        "group_col": group_col,
        "groups": [int(g) for g in groups],
    }

    if len(groups) < 2:
        payload.update({"test": "none", "statistic": float("nan"), "p_value": float("nan")})
        return payload

    if len(groups) == 2:
        t_stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit")
        payload.update(
            {
                "test": "welch_ttest",
                "statistic": float(t_stat),
                "p_value": float(p),
                "decision_alpha_0_05": _decision(float(p), alpha_default),
            }
        )
        return payload

    f_stat, p = stats.f_oneway(*samples)
    payload.update(
        {
            "test": "anova_one_way",
            "statistic": float(f_stat),
            "p_value": float(p),
            "decision_alpha_0_05": _decision(float(p), alpha_default),
        }
    )
    return payload


def _chi_square_section(df: pd.DataFrame, a: str, b: str, alpha: float) -> list[str]:
    out: list[str] = []
    out.append("## 1) Prueba Chi-cuadrado (independencia)")
    out.append("")
    out.append(f"**Variables:** `{a}` (categórica) vs `{b}` (categórica)")
    out.append("")
    out.append("**Hipótesis**")
    out.append(f"- H0: `{a}` y `{b}` son independientes.")
    out.append(f"- H1: existe asociación entre `{a}` y `{b}`.")
    out.append("")

    work = df[[a, b]].dropna().copy()
    # Normalizamos a categorías discretas por si vienen como float/int nullable
    work[a] = work[a].astype("Int64").astype("string")
    work[b] = work[b].astype("Int64").astype("string")

    table = pd.crosstab(work[a], work[b])

    if table.shape[0] < 2 or table.shape[1] < 2:
        out.append("No se pudo ejecutar: la tabla de contingencia no tiene suficientes categorías.")
        out.append("")
        return out

    chi2, p, dof, expected = stats.chi2_contingency(table)

    decision = "RECHAZAR H0" if p < alpha else "NO rechazar H0"

    out.append("**Tabla de contingencia**")
    out.append("")
    out.append(table.to_markdown())
    out.append("")
    out.append("**Resultados**")
    out.append(f"- χ² = {chi2:.4f}")
    out.append(f"- gl = {int(dof)}")
    out.append(f"- p-valor = {p:.6f}")
    out.append(f"- Decisión (α={alpha:.2f}): **{decision}**")
    out.append("")
    out.append(
        "Interpretación: si p < α, hay evidencia estadística de asociación entre las variables."
    )
    out.append("")
    return out


def _mean_comparison_section(df: pd.DataFrame, numeric_col: str, group_col: str, alpha: float) -> list[str]:
    out: list[str] = []
    out.append("## 2) Comparación de medias (t-test / ANOVA)")
    out.append("")
    out.append(f"**Variable numérica:** `{numeric_col}`")
    out.append(f"**Variable de agrupación:** `{group_col}`")
    out.append("")

    work = df[[numeric_col, group_col]].dropna().copy()
    # Forzamos numérico
    work[numeric_col] = pd.to_numeric(work[numeric_col], errors="coerce")
    work = work.dropna(subset=[numeric_col, group_col])

    # Convertimos grupos a categorías discretas
    # (condition suele ser 0/1)
    groups = sorted(work[group_col].astype("Int64").dropna().unique().tolist())

    if len(groups) < 2:
        out.append("No se pudo ejecutar: no hay al menos 2 grupos.")
        out.append("")
        return out

    # Extraemos arrays por grupo
    samples = [work.loc[work[group_col].astype("Int64") == g, numeric_col].to_numpy() for g in groups]
    samples = [s[np.isfinite(s)] for s in samples]

    # Resumen por grupo
    out.append("**Resumen por grupo**")
    out.append("")
    summary_rows = []
    for g, s in zip(groups, samples):
        summary_rows.append(
            {
                "grupo": int(g),
                "n": int(len(s)),
                "media": float(np.mean(s)) if len(s) else np.nan,
                "desv_std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
            }
        )
    out.append(pd.DataFrame(summary_rows).to_markdown(index=False))
    out.append("")

    # 2 grupos -> t-test, 3+ -> ANOVA
    if len(groups) == 2:
        out.append("**Prueba aplicada:** t de Student (Welch, varianzas no asumidas iguales)")
        out.append("")
        out.append("**Hipótesis**")
        out.append(f"- H0: la media de `{numeric_col}` es igual entre los dos grupos.")
        out.append(f"- H1: la media de `{numeric_col}` difiere entre grupos.")
        out.append("")

        t_stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit")
        decision = "RECHAZAR H0" if p < alpha else "NO rechazar H0"

        out.append("**Resultados**")
        out.append(f"- t = {float(t_stat):.4f}")
        out.append(f"- p-valor = {float(p):.6f}")
        out.append(f"- Decisión (α={alpha:.2f}): **{decision}**")
        out.append("")
        out.append("Interpretación: si p < α, hay evidencia de diferencia de medias entre grupos.")
        out.append("")
        return out

    out.append("**Prueba aplicada:** ANOVA de un factor")
    out.append("")
    out.append("**Hipótesis**")
    out.append(f"- H0: todas las medias de `{numeric_col}` son iguales entre grupos.")
    out.append(f"- H1: al menos una media difiere.")
    out.append("")

    f_stat, p = stats.f_oneway(*samples)
    decision = "RECHAZAR H0" if p < alpha else "NO rechazar H0"

    out.append("**Resultados**")
    out.append(f"- F = {float(f_stat):.4f}")
    out.append(f"- p-valor = {float(p):.6f}")
    out.append(f"- Decisión (α={alpha:.2f}): **{decision}**")
    out.append("")
    out.append("Interpretación: si p < α, hay evidencia de diferencias de medias entre grupos.")
    out.append("")
    return out
