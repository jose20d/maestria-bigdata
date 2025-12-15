# src/plotting.py
"""
Utilidades de visualización (solo Matplotlib).
Genera y guarda los gráficos requeridos para el EDA (univariante y bivariante).

Reglas:
- El código permanece en inglés (estándar profesional).
- Los textos visibles (título, ejes, leyendas) se muestran en español.
- Todas las funciones guardan PNG en disco (no muestran en pantalla).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_name(name: str) -> str:
    """Make a filesystem-safe filename stem."""
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-\.]+", "", name)
    return name or "plot"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _wrap_title(title: str, width: int = 78) -> str:
    """Wrap long titles to avoid clipping."""
    return "\n".join(textwrap.wrap(title, width=width))


def _save(fig: plt.Figure, outpath: Path) -> None:
    """
    Save figure without clipping legends/titles.
    bbox_inches='tight' is key to avoid cut-offs.
    """
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# Etiquetas “bonitas” en español (para presentación)
LABELS_ES = {
    "age": "Edad (años)",
    "trestbps": "Presión arterial en reposo (mmHg)",
    "chol": "Colesterol total (mg/dL)",
    "thalach": "Frecuencia cardiaca máxima (lpm)",
    "oldpeak": "Depresión del ST (oldpeak)",
    "sex": "Sexo",
    "cp": "Tipo de dolor torácico",
    "fbs": "Glucosa en ayunas > 120 mg/dL",
    "restecg": "ECG en reposo",
    "exang": "Angina inducida por ejercicio",
    "slope": "Pendiente del ST",
    "ca": "Vasos principales (0–4)",
    "thal": "Prueba de talio (thal)",
    "target": "Objetivo (target)",
    "condition": "Condición (0=No, 1=Sí)",
}


def _label(var: str) -> str:
    return LABELS_ES.get(var, var)


# -----------------------------
# Univariate plots
# -----------------------------

def save_histogram(series: pd.Series, outpath: Path, bins: int = 20) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(s, bins=bins)
    ax.set_title(_wrap_title(f"Histograma: {_label(series.name)}"))
    ax.set_xlabel(_label(series.name))
    ax.set_ylabel("Frecuencia")
    _save(fig, outpath)


def save_boxplot(series: pd.Series, outpath: Path) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.boxplot(s, vert=True)
    ax.set_title(_wrap_title(f"Diagrama de caja: {_label(series.name)}"))
    ax.set_ylabel(_label(series.name))
    _save(fig, outpath)


def save_density(series: pd.Series, outpath: Path) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(constrained_layout=True)
    if len(s) >= 2:
        s.plot(kind="density", ax=ax)
    ax.set_title(_wrap_title(f"Gráfico de densidad: {_label(series.name)}"))
    ax.set_xlabel(_label(series.name))
    ax.set_ylabel("Densidad")
    _save(fig, outpath)


def save_bar_counts(series: pd.Series, outpath: Path, normalize: bool = False) -> None:
    s = series.astype("object")
    counts = s.value_counts(dropna=False, normalize=normalize).sort_index()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(counts.index.astype(str), counts.values)

    titulo = "Gráfico de barras"
    if normalize:
        titulo += " (porcentaje)"
        ax.set_ylabel("Porcentaje")
    else:
        titulo += " (frecuencia)"
        ax.set_ylabel("Frecuencia")

    ax.set_title(_wrap_title(f"{titulo}: {_label(series.name)}"))
    ax.set_xlabel(_label(series.name))
    ax.tick_params(axis="x", labelrotation=45)
    _save(fig, outpath)


# -----------------------------
# Bivariate plots
# -----------------------------

def save_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path) -> None:
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    mask = xs.notna() & ys.notna()

    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(xs[mask], ys[mask])
    ax.set_title(_wrap_title(f"Dispersión: {_label(x)} vs {_label(y)}"))
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    _save(fig, outpath)


def save_grouped_boxplot(df: pd.DataFrame, numeric_var: str, group_var: str, outpath: Path) -> None:
    x = pd.to_numeric(df[numeric_var], errors="coerce")
    g = df[group_var].astype("object")

    tmp = pd.DataFrame({numeric_var: x, group_var: g}).dropna()
    groups = sorted(tmp[group_var].unique(), key=lambda v: str(v))
    data = [tmp.loc[tmp[group_var] == grp, numeric_var].values for grp in groups]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.boxplot(data, labels=[str(grp) for grp in groups])
    ax.set_title(_wrap_title(f"Diagrama de caja: {_label(numeric_var)} por {_label(group_var)}"))
    ax.set_xlabel(_label(group_var))
    ax.set_ylabel(_label(numeric_var))
    _save(fig, outpath)


def save_stacked_bar(
    df: pd.DataFrame,
    x_cat: str,
    y_cat: str,
    outpath: Path,
    percent: bool = True,
) -> None:
    x = df[x_cat].astype("object")
    y = df[y_cat].astype("object")
    ct = pd.crosstab(x, y, dropna=False)

    if percent:
        ct_plot = ct.div(ct.sum(axis=1).replace(0, pd.NA), axis=0) * 100
        ylabel = "Porcentaje"
        title = f"Barras apiladas (%): {_label(x_cat)} vs {_label(y_cat)}"
    else:
        ct_plot = ct
        ylabel = "Frecuencia"
        title = f"Barras apiladas (frecuencia): {_label(x_cat)} vs {_label(y_cat)}"

    fig, ax = plt.subplots(constrained_layout=True)
    bottom = None

    for col in ct_plot.columns:
        vals = ct_plot[col].fillna(0).values
        label = str(col)
        # Si es la variable condition, ya la estamos describiendo en el título/legend-title,
        # y el valor 0/1 es suficiente.
        if y_cat == "condition":
            label = str(col)
        if bottom is None:
            ax.bar(ct_plot.index.astype(str), vals, label=label)
            bottom = vals
        else:
            ax.bar(ct_plot.index.astype(str), vals, bottom=bottom, label=label)
            bottom = bottom + vals

    # Título con espacio inferior para no chocar con la leyenda
    ax.set_title(_wrap_title(title), pad=18)

    ax.set_xlabel(_label(x_cat))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)

    # Leyenda: arriba y fuera del área del gráfico
    legend_title = _label(y_cat)
    ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=min(3, len(ct_plot.columns)), frameon=True)

    _save(fig, outpath)
