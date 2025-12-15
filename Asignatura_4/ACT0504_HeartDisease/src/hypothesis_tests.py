# src/hypothesis_tests.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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

    lines: list[str] = []
    lines.append("# Contraste de hipótesis")
    lines.append("")
    lines.append(f"Nivel de significación (α): **{alpha:.2f}**")
    lines.append("")

    # ============
    # 1) Chi-cuadrado (categórica vs categórica)
    # Elegimos: sex vs condition (sencillo y estándar)
    # ============
    chi_var_a = "sex"
    chi_var_b = "condition" if "condition" in df.columns else "target"

    if chi_var_a in df.columns and chi_var_b in df.columns:
        lines.extend(_chi_square_section(df, chi_var_a, chi_var_b, alpha))
    else:
        lines.append("## 1) Prueba Chi-cuadrado (independencia)")
        lines.append("")
        lines.append(f"No se pudo ejecutar: faltan columnas `{chi_var_a}` y/o `{chi_var_b}`.")
        lines.append("")

    # ============
    # 2) t-test o ANOVA (numérica vs categórica)
    # Elegimos: chol por condition (si no, thalach por condition)
    # ============
    group_col = chi_var_b
    numeric_candidates = ["chol", "thalach", "age", "trestbps", "oldpeak"]
    num_col = next((c for c in numeric_candidates if c in df.columns), None)

    if group_col in df.columns and num_col is not None:
        lines.extend(_mean_comparison_section(df, num_col, group_col, alpha))
    else:
        lines.append("## 2) Comparación de medias (t-test / ANOVA)")
        lines.append("")
        lines.append("No se pudo ejecutar por falta de columnas necesarias.")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"hypothesis_md": str(out_md)}


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
