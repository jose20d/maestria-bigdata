# src/stats_tests.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import REPORTS_DIR


@dataclass
class TestResult:
    nombre: str
    estadistico: float
    pvalor: float
    gl: int | None = None
    efecto: float | None = None
    efecto_nombre: str | None = None
    notas: list[str] | None = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _cramers_v(contingency: pd.DataFrame) -> float:
    """Cramér's V para tablas RxC."""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    if n == 0:
        return float("nan")
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1) if min(r - 1, k - 1) > 0 else 1)))


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (pooled SD) para dos grupos."""
    x = x.astype(float)
    y = y.astype(float)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return float("nan")
    sx = x.var(ddof=1)
    sy = y.var(ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if s_pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / s_pooled)


def _eta_squared_from_anova(f_stat: float, df_between: int, df_within: int) -> float:
    """Eta squared: tamaño de efecto para ANOVA."""
    if df_between <= 0 or df_within <= 0:
        return float("nan")
    return float((f_stat * df_between) / (f_stat * df_between + df_within))


def chi_square_cp_vs_condition(df: pd.DataFrame) -> tuple[TestResult, pd.DataFrame, pd.DataFrame]:
    """
    Chi-cuadrado: cp (categórica) vs condition (categórica 0/1).
    Devuelve: resultado, tabla de contingencia, porcentajes por fila.
    """
    notas: list[str] = []
    cols = ["cp", "condition"]
    data = df[cols].dropna()

    contingency = pd.crosstab(data["cp"], data["condition"])
    row_pct = contingency.div(contingency.sum(axis=1), axis=0).mul(100).round(2)

    chi2, p, dof, expected = stats.chi2_contingency(contingency, correction=False)

    # Regla práctica: esperado >= 5 en la mayoría de celdas
    expected_min = float(np.min(expected)) if expected.size else float("nan")
    if expected.size and np.any(expected < 5):
        notas.append(
            "Advertencia: existen frecuencias esperadas < 5 en alguna celda; interpretar con cautela."
        )
        notas.append(f"Mínimo esperado observado: {expected_min:.2f}")

    v = _cramers_v(contingency)

    res = TestResult(
        nombre="Chi-cuadrado: cp vs condition",
        estadistico=float(chi2),
        pvalor=float(p),
        gl=int(dof),
        efecto=v,
        efecto_nombre="Cramér's V",
        notas=notas,
    )
    return res, contingency, row_pct


def ttest_thalach_by_condition(df: pd.DataFrame) -> tuple[TestResult, pd.DataFrame]:
    """
    t-test (Welch): thalach (numérica) por condition (0/1).
    Devuelve: resultado y resumen por grupo.
    """
    notas: list[str] = []
    data = df[["thalach", "condition"]].dropna()

    # Asegurar numérico
    thalach = pd.to_numeric(data["thalach"], errors="coerce")
    cond = data["condition"]
    data = pd.DataFrame({"thalach": thalach, "condition": cond}).dropna()

    g0 = data.loc[data["condition"] == 0, "thalach"].to_numpy()
    g1 = data.loc[data["condition"] == 1, "thalach"].to_numpy()

    if g0.size < 2 or g1.size < 2:
        res = TestResult(
            nombre="t-test (Welch): thalach por condition",
            estadistico=float("nan"),
            pvalor=float("nan"),
            efecto=float("nan"),
            efecto_nombre="Cohen's d",
            notas=["No hay suficientes datos en ambos grupos para aplicar t-test."],
        )
        summary = _group_summary(data, group_col="condition", value_col="thalach")
        return res, summary

    t_stat, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")

    d = _cohens_d(g0, g1)

    summary = _group_summary(data, group_col="condition", value_col="thalach")

    res = TestResult(
        nombre="t-test (Welch): thalach por condition",
        estadistico=float(t_stat),
        pvalor=float(p),
        gl=None,
        efecto=d,
        efecto_nombre="Cohen's d",
        notas=notas,
    )
    return res, summary


def anova_age_by_cp(df: pd.DataFrame) -> tuple[TestResult, pd.DataFrame]:
    """
    ANOVA: age (numérica) por cp (0-3).
    Devuelve: resultado y resumen por grupo.
    """
    notas: list[str] = []
    data = df[["age", "cp"]].dropna()

    age = pd.to_numeric(data["age"], errors="coerce")
    cp = data["cp"]
    data = pd.DataFrame({"age": age, "cp": cp}).dropna()

    groups = []
    group_labels = []
    for k, g in data.groupby("cp")["age"]:
        arr = g.to_numpy()
        if arr.size > 0:
            groups.append(arr)
            group_labels.append(k)

    if len(groups) < 2:
        res = TestResult(
            nombre="ANOVA: age por cp",
            estadistico=float("nan"),
            pvalor=float("nan"),
            efecto=float("nan"),
            efecto_nombre="Eta squared",
            notas=["No hay suficientes grupos con datos para aplicar ANOVA."],
        )
        summary = _group_summary(data, group_col="cp", value_col="age")
        return res, summary

    f_stat, p = stats.f_oneway(*groups)

    # grados de libertad: k-1, N-k
    k = len(groups)
    n_total = sum(g.size for g in groups)
    df_between = k - 1
    df_within = n_total - k
    eta2 = _eta_squared_from_anova(float(f_stat), df_between, df_within)

    summary = _group_summary(data, group_col="cp", value_col="age")

    res = TestResult(
        nombre="ANOVA: age por cp",
        estadistico=float(f_stat),
        pvalor=float(p),
        gl=None,
        efecto=eta2,
        efecto_nombre="Eta squared (η²)",
        notas=notas,
    )
    return res, summary


def _group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)[value_col]
        .agg(n="count", media="mean", mediana="median", std="std", min="min", max="max")
        .round(4)
        .reset_index()
    )


def run_hypothesis_tests(df_clean: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """
    Ejecuta todas las pruebas y guarda artefactos en output/reports/.
    Retorna un dict con resultados por si quieres imprimirlos en consola.
    """
    _ensure_dir(REPORTS_DIR)

    results: dict[str, Any] = {}

    # 1) Chi-cuadrado
    chi_res, contingency, row_pct = chi_square_cp_vs_condition(df_clean)
    contingency_path = REPORTS_DIR / "chi_square_cp_vs_condition_contingency.csv"
    rowpct_path = REPORTS_DIR / "chi_square_cp_vs_condition_rowpct.csv"
    contingency.to_csv(contingency_path, index=True)
    row_pct.to_csv(rowpct_path, index=True)

    results["chi_square"] = {
        "result": chi_res,
        "contingency_csv": str(contingency_path),
        "rowpct_csv": str(rowpct_path),
    }

    # 2) t-test
    t_res, t_summary = ttest_thalach_by_condition(df_clean)
    t_summary_path = REPORTS_DIR / "ttest_thalach_by_condition_summary.csv"
    t_summary.to_csv(t_summary_path, index=False)
    results["ttest"] = {
        "result": t_res,
        "summary_csv": str(t_summary_path),
    }

    # 3) ANOVA
    a_res, a_summary = anova_age_by_cp(df_clean)
    a_summary_path = REPORTS_DIR / "anova_age_by_cp_summary.csv"
    a_summary.to_csv(a_summary_path, index=False)
    results["anova"] = {
        "result": a_res,
        "summary_csv": str(a_summary_path),
    }

    # 4) Reporte MD (español, listo para pegar)
    report_path = REPORTS_DIR / "hypothesis_tests.md"
    report_path.write_text(_build_md_report(chi_res, t_res, a_res, alpha), encoding="utf-8")
    results["report_md"] = str(report_path)

    return results


def _decision(p: float, alpha: float) -> str:
    if np.isnan(p):
        return "No aplicable"
    return "Rechazar H₀" if p < alpha else "No rechazar H₀"


def _build_md_report(chi: TestResult, tt: TestResult, an: TestResult, alpha: float) -> str:
    def fmt(x: float) -> str:
        if x is None or np.isnan(x):
            return "NA"
        return f"{x:.6g}"

    lines: list[str] = []
    lines.append("# Contraste de hipótesis (Punto 5)")
    lines.append("")
    lines.append(f"Nivel de significación: **α = {alpha}**")
    lines.append("")

    # Chi-square
    lines.append("## 1) Prueba Chi-cuadrado (cp vs condition)")
    lines.append("**H₀:** cp y condition son independientes.")
    lines.append("**H₁:** cp y condition no son independientes.")
    lines.append("")
    lines.append(f"- Estadístico χ²: **{fmt(chi.estadistico)}**")
    lines.append(f"- p-valor: **{fmt(chi.pvalor)}**")
    if chi.gl is not None:
        lines.append(f"- Grados de libertad: **{chi.gl}**")
    if chi.efecto is not None:
        lines.append(f"- Tamaño de efecto ({chi.efecto_nombre}): **{fmt(chi.efecto)}**")
    lines.append(f"- Decisión (α={alpha}): **{_decision(chi.pvalor, alpha)}**")
    if chi.notas:
        lines.append("")
        lines.append("**Notas:**")
        for n in chi.notas:
            lines.append(f"- {n}")
    lines.append("")

    # t-test
    lines.append("## 2) t-test (Welch) (thalach por condition)")
    lines.append("**H₀:** la media de thalach es igual entre condition=0 y condition=1.")
    lines.append("**H₁:** las medias de thalach difieren entre condition=0 y condition=1.")
    lines.append("")
    lines.append(f"- Estadístico t: **{fmt(tt.estadistico)}**")
    lines.append(f"- p-valor: **{fmt(tt.pvalor)}**")
    if tt.efecto is not None:
        lines.append(f"- Tamaño de efecto ({tt.efecto_nombre}): **{fmt(tt.efecto)}**")
    lines.append(f"- Decisión (α={alpha}): **{_decision(tt.pvalor, alpha)}**")
    if tt.notas:
        lines.append("")
        lines.append("**Notas:**")
        for n in tt.notas:
            lines.append(f"- {n}")
    lines.append("")

    # ANOVA
    lines.append("## 3) ANOVA (age por cp)")
    lines.append("**H₀:** todas las medias de age son iguales entre los grupos de cp.")
    lines.append("**H₁:** al menos una media de age difiere entre grupos de cp.")
    lines.append("")
    lines.append(f"- Estadístico F: **{fmt(an.estadistico)}**")
    lines.append(f"- p-valor: **{fmt(an.pvalor)}**")
    if an.efecto is not None:
        lines.append(f"- Tamaño de efecto ({an.efecto_nombre}): **{fmt(an.efecto)}**")
    lines.append(f"- Decisión (α={alpha}): **{_decision(an.pvalor, alpha)}**")
    if an.notas:
        lines.append("")
        lines.append("**Notas:**")
        for n in an.notas:
            lines.append(f"- {n}")
    lines.append("")

    lines.append("### Archivos generados")
    lines.append("- `chi_square_cp_vs_condition_contingency.csv` (tabla de contingencia)")
    lines.append("- `chi_square_cp_vs_condition_rowpct.csv` (porcentajes por fila)")
    lines.append("- `ttest_thalach_by_condition_summary.csv` (resumen por grupo)")
    lines.append("- `anova_age_by_cp_summary.csv` (resumen por grupo)")
    lines.append("- `hypothesis_tests.md` (este reporte)")
    lines.append("")

    return "\n".join(lines)
