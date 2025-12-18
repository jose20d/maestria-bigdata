# src/report_builder.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.config import OUTPUT_DIR


# -----------------------------
#  Inputs
# -----------------------------
@dataclass(frozen=True)
class ExecutiveReportInputs:
    dataset_name: str
    generated_at: datetime

    # Tablas/insumos
    univariate_numeric_csv: Path
    chi_square_contingency_csv: Path
    ttest_or_anova_summary_csv: Path
    hypothesis_results_json: Path

    # Gráficas (carpeta base)
    plots_dir: Path

    # Salida
    output_pdf: Path


# -----------------------------
#  Estilos
# -----------------------------
def _styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "TitleES",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1ES",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2ES",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyES",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "SmallES",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        spaceAfter=4,
    )
    caption = ParagraphStyle(
        "CaptionES",
        parent=base["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=11,
        spaceBefore=4,
        spaceAfter=10,
        textColor=colors.HexColor("#333333"),
    )

    return title, h1, h2, body, small, caption


# -----------------------------
#  Helpers
# -----------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    return pd.read_csv(path)


def _round_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _available_width(page_width: float, left_margin: float, right_margin: float) -> float:
    return page_width - left_margin - right_margin


def _table_from_df(
    df: pd.DataFrame,
    max_width: float,
    header_bg=colors.HexColor("#D9D9D9"),
    font_size: int = 8,
    leading: int = 9,
) -> Table:
    """
    Crea tabla compacta y evita traslapes:
    - Redondea valores
    - Fuente pequeña
    - ColWidths repartidos (con primera columna un poco más ancha)
    - WordWrap activado
    """
    df = _round_df(df, decimals=3)
    data = [list(df.columns)] + df.astype(str).values.tolist()

    ncols = len(df.columns)
    if ncols == 0:
        return Table([["(sin datos)"]])

    # Columna 0 un poco más ancha (suele ser "variable" o "grupo")
    base_w = max_width / ncols
    col_widths = [base_w] * ncols
    if ncols >= 2:
        col_widths[0] = base_w * 1.2
        # reajuste para conservar ancho total
        rest = max_width - col_widths[0]
        for i in range(1, ncols):
            col_widths[i] = rest / (ncols - 1)

    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), leading),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def _pairwise(items: List[Path]) -> Iterable[Tuple[Path, Optional[Path]]]:
    it = iter(items)
    for a in it:
        b = next(it, None)
        yield a, b


def _plot_caption_es(plot_path: Path) -> str:
    """
    Convierte nombres de archivo a captions en español (sin mezclar idiomas).
    Ajustá este mapeo si cambian nombres.
    """
    stem = plot_path.stem.lower()

    var_labels = {
        "age": "Edad",
        "trestbps": "Presión arterial en reposo",
        "chol": "Colesterol total",
        "thalach": "Frecuencia cardiaca máxima",
        "oldpeak": "Oldpeak",
        "sex": "Sexo",
        "cp": "Tipo de dolor torácico",
        "fbs": "Glucosa en ayunas > 120 mg/dL",
        "restecg": "ECG en reposo",
        "exang": "Angina inducida por ejercicio",
        "slope": "Pendiente del ST",
        "ca": "Vasos principales (0–4)",
        "thal": "Prueba de talio (thal)",
        "condition": "Condición (0=No, 1=Sí)",
        "target": "Objetivo (target)",
    }

    # Casos bivariantes con nombre explícito
    explicit = {
        "cp_vs_condition_stacked_bar": "Dolor torácico (cp) vs condición (barras apiladas %)",
        "sex_vs_condition_stacked_bar": "Sexo vs condición (barras apiladas %)",
        "chol_by_condition_box": "Colesterol (chol) por condición (boxplot)",
        "thalach_by_condition_box": "Frecuencia cardiaca máxima (thalach) por condición (boxplot)",
        "age_vs_thalach_scatter": "Edad vs frecuencia cardiaca máxima (diagrama de dispersión)",
        "trestbps_vs_chol_scatter": "Presión en reposo (trestbps) vs colesterol (diagrama de dispersión)",
    }
    if stem in explicit:
        return explicit[stem]

    # Patrones univariantes: {var}_{tipo}
    parts = stem.split("_")
    if len(parts) >= 2:
        var = parts[0]
        kind = parts[-1]
        label = var_labels.get(var, var)

        if kind == "hist":
            return f"Histograma: {label}"
        if kind == "box":
            return f"Diagrama de caja: {label}"
        if kind == "density":
            return f"Gráfico de densidad: {label}"
        if kind == "bar":
            if var == "condition":
                return "Distribución de la condición (0=No, 1=Sí)"
            return f"Gráfico de barras: {label}"

    return stem.replace("_", " ").strip().capitalize()


def _load_curated_plots(inputs: ExecutiveReportInputs) -> List[Tuple[str, List[Path]]]:
    """
    Selección “curada” de gráficas para el PDF:
    - Incluye un set amplio (univariante + bivariante)
    - Mantiene un orden estable
    - Omite faltantes sin romper
    """
    base = inputs.plots_dir

    numeric_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_vars = ["condition", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    uni_num: List[Path] = []
    for v in numeric_vars:
        # Reducimos un poco: histograma + boxplot (omitimos densidad para acortar el PDF)
        for kind in ["hist", "box"]:
            uni_num.append(base / "univariate" / "numeric" / f"{v}_{kind}.png")

    uni_cat: List[Path] = []
    for v in categorical_vars:
        uni_cat.append(base / "univariate" / "categorical" / f"{v}_bar.png")

    bi_cat_cat: List[Path] = [
        base / "bivariate" / "cat_cat" / "sex_vs_condition_stacked_bar.png",
        base / "bivariate" / "cat_cat" / "cp_vs_condition_stacked_bar.png",
    ]
    bi_num_cat: List[Path] = [
        base / "bivariate" / "num_cat" / "chol_by_condition_box.png",
        base / "bivariate" / "num_cat" / "thalach_by_condition_box.png",
    ]
    bi_num_num: List[Path] = [
        base / "bivariate" / "num_num" / "age_vs_thalach_scatter.png",
        base / "bivariate" / "num_num" / "trestbps_vs_chol_scatter.png",
    ]

    def existing(paths: List[Path]) -> List[Path]:
        seen: set[str] = set()
        out: List[Path] = []
        for p in paths:
            if not p.exists():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    sections: List[Tuple[str, List[Path]]] = [
        ("Univariante (numéricas) – distribución y outliers", existing(uni_num)),
        ("Univariante (categóricas) – frecuencias", existing(uni_cat)),
        ("Bivariante (categóricas) – asociaciones", existing(bi_cat_cat)),
        ("Bivariante (numérica vs condición) – comparación por grupos", existing(bi_num_cat)),
        ("Bivariante (numéricas) – dispersión", existing(bi_num_num)),
    ]

    return [(title, plots) for title, plots in sections if plots]


def _img_flowable(path: Path, max_width: float, max_height: float) -> Image:
    img = Image(str(path))
    img._restrictSize(max_width, max_height)
    img.hAlign = "CENTER"
    return img


# -----------------------------
#  Builder principal
# -----------------------------
def build_executive_report_elements(
    inputs: ExecutiveReportInputs,
    alpha: float = 0.05,
) -> Tuple[Path, List]:
    """
    Devuelve (output_pdf, elements) para que pdf_report.py construya el PDF.
    """
    title, h1, h2, body, small, caption = _styles()

    # Márgenes (deben coincidir con pdf_report.py)
    page_w, page_h = A4
    left_margin = 40
    right_margin = 40
    top_margin = 40
    bottom_margin = 40
    usable_w = _available_width(page_w, left_margin, right_margin)

    # Validaciones mínimas
    _ = _safe_read_csv(inputs.univariate_numeric_csv)
    _ = _safe_read_csv(inputs.chi_square_contingency_csv)
    _ = _safe_read_csv(inputs.ttest_or_anova_summary_csv)
    if not inputs.hypothesis_results_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {inputs.hypothesis_results_json}")

    elements: List = []

    # -------------------------
    # Portada / Resumen
    # -------------------------
    elements.append(Paragraph("Informe Ejecutivo – Resultados del Análisis", title))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Dataset:</b> {inputs.dataset_name}", body))
    elements.append(Paragraph(f"<b>Generado:</b> {inputs.generated_at.strftime('%Y-%m-%d %H:%M')}", body))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Resumen", h1))
    elements.append(
        Paragraph(
            "Este documento resume los resultados principales del análisis exploratorio (EDA) y del contraste de hipótesis. "
            "Incluye visualizaciones clave y métricas para una lectura rápida orientada a toma de decisiones.",
            body,
        )
    )
    elements.append(Spacer(1, 6))

    # -------------------------
    # Tabla univariante (compacta en 2 partes)
    # -------------------------
    elements.append(Paragraph("EDA univariante – resumen numérico", h1))
    elements.append(Paragraph("Extracto de medidas descriptivas principales.", body))
    elements.append(Spacer(1, 6))

    uni = _safe_read_csv(inputs.univariate_numeric_csv)

    # Parte 1: tendencia central
    cols1 = ["variable", "n", "missing", "mean", "median", "mode", "variance"]
    # Parte 2: dispersión/forma
    cols2 = ["std", "iqr", "q1", "q3", "min", "max", "skewness", "outliers_iqr_count"]

    # Normaliza nombres por si cambian (p.ej. asimetría vs skewness)
    renames = {
        "asimetría": "skewness",
        "outliers_iqr": "outliers_iqr_count",
        "outliers_iqr_count": "outliers_iqr_count",
    }
    for old, new in renames.items():
        if old in uni.columns and new not in uni.columns:
            uni = uni.rename(columns={old: new})

    part1 = uni[[c for c in cols1 if c in uni.columns]].copy()
    part2 = uni[[c for c in cols2 if c in uni.columns]].copy()

    # Traducción header “skewness/outliers...”
    part2 = part2.rename(columns={"skewness": "asimetría", "outliers_iqr_count": "outliers_iqr"})

    elements.append(Paragraph("<i>Parte 1</i>", small))
    elements.append(_table_from_df(part1, max_width=usable_w, font_size=8, leading=9))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<i>Parte 2</i>", small))
    elements.append(_table_from_df(part2, max_width=usable_w, font_size=8, leading=9))
    elements.append(PageBreak())

    # -------------------------
    # Gráficas (curadas) – 2 por página (sin páginas “vacías” de solo títulos)
    # -------------------------
    sections = _load_curated_plots(inputs)
    if not sections:
        elements.append(Paragraph("No se encontraron gráficas en output/plots/.", body))
        elements.append(PageBreak())
    else:
        # 2 imágenes por página (vertical)
        max_img_w = usable_w
        usable_h = page_h - top_margin - bottom_margin
        # Dejamos un poco más de margen para que el bloque completo (títulos + 2 imágenes)
        # siempre quepa y no deje títulos “colgados” en una página.
        max_img_h = (usable_h * 0.38)

        first_page_overall = True
        for section_idx, (section_title, plots) in enumerate(sections):
            for page_idx, (p1, p2) in enumerate(_pairwise(plots)):
                if not first_page_overall:
                    elements.append(PageBreak())
                first_page_overall = False

                # Importante: ponemos encabezados dentro del bloque para evitar páginas con solo títulos
                block: List = []
                if section_idx == 0 and page_idx == 0:
                    block.append(Paragraph("Gráficas (EDA)", h1))
                    block.append(Spacer(1, 6))

                block.append(Paragraph(section_title, h1))
                block.append(Spacer(1, 6))

                block.append(Paragraph(_plot_caption_es(p1), h2))
                block.append(_img_flowable(p1, max_img_w, max_img_h))

                if p2 is not None:
                    block.append(Spacer(1, 6))
                    block.append(Paragraph(_plot_caption_es(p2), h2))
                    block.append(_img_flowable(p2, max_img_w, max_img_h))

                elements.append(KeepTogether(block))

        elements.append(PageBreak())

    # -------------------------
    # Contraste de hipótesis (resumen corto)
    # -------------------------
    elements.append(Paragraph("Contraste de hipótesis (Punto 5)", h1))
    elements.append(Paragraph(f"Nivel de significación utilizado: α = {alpha:.2f}.", body))
    elements.append(Spacer(1, 8))

    # Cargamos resultados para mostrar estadísticos (evita “markdown tirado”)
    results = json.loads(inputs.hypothesis_results_json.read_text(encoding="utf-8"))

    # Chi-cuadrado: tabla de contingencia (resumen)
    elements.append(Paragraph("1) Prueba Chi-cuadrado (independencia)", h2))
    elements.append(Paragraph("<b>Variables:</b> <code>cp</code> (categórica) vs <code>condition</code> (categórica)", body))
    elements.append(
        Paragraph(
            "<b>Hipótesis</b><br/>"
            "• H0: <code>cp</code> y <code>condition</code> son independientes.<br/>"
            "• H1: existe asociación entre <code>cp</code> y <code>condition</code>.",
            body,
        )
    )
    chi = _safe_read_csv(inputs.chi_square_contingency_csv)
    # Tabla compacta (si viene con columnas tipo sex,0,1 o similar)
    elements.append(Paragraph("<b>Tabla de contingencia</b>", body))
    elements.append(_table_from_df(chi, max_width=usable_w, font_size=8, leading=9))
    chi_stats = (results.get("chi_square") or {})
    if chi_stats:
        chi2 = chi_stats.get("chi2")
        p = chi_stats.get("p_value")
        dof = chi_stats.get("dof")
        decision = "Rechazar H0" if (isinstance(p, (int, float)) and p < alpha) else "No rechazar H0"
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("<b>Resultados</b>", body))
        elements.append(
            Paragraph(
                f"χ² = <b>{chi2:.4f}</b> &nbsp;&nbsp; gl = <b>{int(dof)}</b> &nbsp;&nbsp; p-valor = <b>{p:.6f}</b><br/>"
                f"Decisión (α={alpha:.2f}): <b>{decision}</b>",
                body,
            )
        )
    elements.append(Spacer(1, 10))

    # t-test / ANOVA resumen
    elements.append(Paragraph("2) Comparación de medias (t-test / ANOVA)", h2))
    comp = _safe_read_csv(inputs.ttest_or_anova_summary_csv)
    elements.append(Paragraph("<b>Resumen por grupo</b>", body))
    elements.append(_table_from_df(comp, max_width=usable_w, font_size=8, leading=9))
    mean_stats = (results.get("mean_comparison") or {})
    if mean_stats:
        test = mean_stats.get("test")
        stat = mean_stats.get("statistic")
        p = mean_stats.get("p_value")
        if test == "welch_ttest":
            test_name = "t de Student (Welch)"
            stat_name = "t"
        elif test == "anova_one_way":
            test_name = "ANOVA de un factor"
            stat_name = "F"
        else:
            test_name = "No aplicable"
            stat_name = "estadístico"

        if isinstance(p, (int, float)):
            decision = "Rechazar H0" if p < alpha else "No rechazar H0"
        else:
            decision = "No aplicable"

        elements.append(Spacer(1, 6))
        elements.append(Paragraph("<b>Resultados</b>", body))
        if isinstance(stat, (int, float)) and isinstance(p, (int, float)):
            elements.append(
                Paragraph(
                    f"Prueba aplicada: <b>{test_name}</b><br/>"
                    f"{stat_name} = <b>{stat:.4f}</b> &nbsp;&nbsp; p-valor = <b>{p:.6f}</b><br/>"
                    f"Decisión (α={alpha:.2f}): <b>{decision}</b>",
                    body,
                )
            )
        else:
            elements.append(Paragraph("No se pudieron calcular estadísticos para esta comparación.", body))
    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            "<b>Trazabilidad (archivos de salida)</b><br/>"
            "Las salidas completas (tablas y detalles del EDA) se encuentran en <code>output/reports/</code>, "
            "y las gráficas en <code>output/plots/</code>.",
            body,
        )
    )

    return inputs.output_pdf, elements


# -----------------------------
#  Factory (opcional) para construir inputs por defecto
# -----------------------------
def default_executive_report_inputs() -> ExecutiveReportInputs:
    """
    Si querés simplificar main/pdf_report: genera inputs con rutas estándar.
    """
    reports_dir = OUTPUT_DIR / "reports"
    plots_dir = OUTPUT_DIR / "plots"
    return ExecutiveReportInputs(
        dataset_name="Heart Disease (Cleveland, UCI)",
        generated_at=datetime.now(),
        univariate_numeric_csv=reports_dir / "univariate_numeric_summary.csv",
        chi_square_contingency_csv=reports_dir / "chi_square_cp_vs_condition_contingency.csv",
        ttest_or_anova_summary_csv=reports_dir / "ttest_chol_by_condition_summary.csv",
        hypothesis_results_json=reports_dir / "hypothesis_results.json",
        plots_dir=plots_dir,
        output_pdf=reports_dir / "INFORME_EJECUTIVO_ACT0504.pdf",
    )
