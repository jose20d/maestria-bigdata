# src/report_builder.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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

from .config import OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR


# =========================
#  Labels / captions (ES)
# =========================
_PLOT_LABELS_ES = {
    "condition_bar": "Barras: Condición",
    "sex_bar": "Barras: Sexo",
    "cp_bar": "Barras: Tipo de dolor torácico",
    "fbs_bar": "Barras: Glucosa en ayunas (fbs)",
    "restecg_bar": "Barras: ECG en reposo (restecg)",
    "exang_bar": "Barras: Angina inducida por ejercicio (exang)",
    "slope_bar": "Barras: Pendiente ST (slope)",
    "thal_bar": "Barras: Thal",
    "ca_bar": "Barras: Vasos coloreados (ca)",
    "age_hist": "Histograma: Edad",
    "age_box": "Diagrama de caja: Edad",
    "age_density": "Densidad: Edad",
    "chol_hist": "Histograma: Colesterol",
    "chol_box": "Diagrama de caja: Colesterol",
    "chol_density": "Densidad: Colesterol",
    "thalach_hist": "Histograma: Frecuencia cardiaca máxima (thalach)",
    "thalach_box": "Diagrama de caja: Frecuencia cardiaca máxima (thalach)",
    "thalach_density": "Densidad: Frecuencia cardiaca máxima (thalach)",
    "trestbps_hist": "Histograma: Presión en reposo (trestbps)",
    "trestbps_box": "Diagrama de caja: Presión en reposo (trestbps)",
    "trestbps_density": "Densidad: Presión en reposo (trestbps)",
    "oldpeak_hist": "Histograma: Oldpeak",
    "oldpeak_box": "Diagrama de caja: Oldpeak",
    "oldpeak_density": "Densidad: Oldpeak",
    "cp_vs_condition_stacked_bar": "Barras apiladas (%): Dolor torácico vs Condición",
    "sex_vs_condition_stacked_bar": "Barras apiladas (%): Sexo vs Condición",
    "chol_by_condition_box": "Caja: Colesterol por Condición",
    "thalach_by_condition_box": "Caja: Thalach por Condición",
    "age_vs_thalach_scatter": "Dispersión: Edad vs Thalach",
    "trestbps_vs_chol_scatter": "Dispersión: Trestbps vs Colesterol",
}


def plot_caption_es(path: Path) -> str:
    key = path.stem.lower()
    return _PLOT_LABELS_ES.get(key, key.replace("_", " ").title())


# =========================
#  Inputs
# =========================
@dataclass(frozen=True)
class ReportInputs:
    # CSVs
    univariate_numeric_csv: Path
    hypothesis_md: Path

    # plots (ordenados)
    plots: List[Path]


def _discover_inputs() -> ReportInputs:
    # Reportes esperados (ajustá si cambian nombres)
    uni_num = REPORTS_DIR / "univariate_numeric_summary.csv"
    hypo_md = REPORTS_DIR / "hypothesis_tests.md"

    # Gráficas: tomamos todas las PNG bajo output/plots/
    all_plots = sorted([p for p in PLOTS_DIR.rglob("*.png")])

    return ReportInputs(
        univariate_numeric_csv=uni_num,
        hypothesis_md=hypo_md,
        plots=all_plots,
    )


# =========================
#  Styles
# =========================
def _styles():
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "TitleES",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1ES",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2ES",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyES",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "SmallES",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        spaceAfter=6,
        textColor=colors.HexColor("#333333"),
    )
    mono = ParagraphStyle(
        "MonoES",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        spaceAfter=6,
    )
    return title, h1, h2, body, small, mono


# =========================
#  Helpers (tables/images)
# =========================
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _format_df_for_table(df: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    for col in df2.columns:
        if pd.api.types.is_float_dtype(df2[col]):
            df2[col] = df2[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return df2


def _make_table(
    df: pd.DataFrame,
    col_widths: List[float],
    header_bg=colors.HexColor("#DDDDDD"),
) -> Table:
    data = [list(df.columns)] + df.values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)

    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]
        )
    )
    return t


def _image_block(img_path: Path, caption: str, max_width: float, max_height: float) -> KeepTogether:
    # Ajuste proporcional al espacio disponible
    img = Image(str(img_path))
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        iw, ih = 800, 600

    scale = min(max_width / iw, max_height / ih, 1.0)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale

    # Evita que “título en una página / imagen en otra”
    return KeepTogether(
        [
            Paragraph(caption, getSampleStyleSheet()["Heading3"]),
            Spacer(1, 0.15 * cm),
            img,
            Spacer(1, 0.35 * cm),
        ]
    )


# =========================
#  Main builder
# =========================
def build_executive_report_elements(
    inputs: Optional[ReportInputs] = None,
    alpha: float = 0.05,
) -> Tuple[Path, List]:
    """
    Devuelve: (ruta_pdf_salida, lista_de_elementos_platypus)
    - inputs: si es None, autodetecta desde output/plots y output/reports
    """
    if inputs is None:
        inputs = _discover_inputs()

    # salida
    pdf_name = "INFORME_EJECUTIVO_ACT0504.pdf"
    output_pdf = REPORTS_DIR / pdf_name
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # estilos
    title, h1, h2, body, small, mono = _styles()

    # layout: A4 con márgenes como en pdf_report.py (40pt aprox)
    page_w, page_h = A4
    left_margin = right_margin = 40
    usable_w = page_w - left_margin - right_margin

    elements: List = []

    # =========================
    # Portada / encabezado
    # =========================
    elements.append(Paragraph("Informe Ejecutivo – Resultados del Análisis", title))
    elements.append(Paragraph("Dataset: Heart Disease (Cleveland, UCI)", body))
    elements.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    elements.append(Spacer(1, 0.4 * cm))

    # =========================
    # Resumen
    # =========================
    elements.append(Paragraph("Resumen", h1))
    elements.append(
        Paragraph(
            "Este documento resume los resultados principales del análisis exploratorio (EDA) y del contraste de hipótesis. "
            "Incluye visualizaciones clave y métricas para una lectura rápida orientada a toma de decisiones.",
            body,
        )
    )

    # =========================
    # EDA univariante (tabla numérica)
    # =========================
    elements.append(Paragraph("EDA univariante – resumen numérico", h1))
    elements.append(Paragraph("Extracto de medidas descriptivas principales.", small))

    uni_df = _safe_read_csv(inputs.univariate_numeric_csv)
    if uni_df is None or uni_df.empty:
        elements.append(
            Paragraph(
                f"No se encontró el archivo esperado: {inputs.univariate_numeric_csv}. "
                "Ejecute el pipeline para generar el resumen univariante.",
                small,
            )
        )
    else:
        uni_df = _format_df_for_table(uni_df, max_rows=10)

        # Partimos en 2 tablas para que “outliers_iqr_count” no se salga
        cols_part1 = ["variable", "n", "missing", "mean", "median", "mode", "variance"]
        cols_part2 = ["std", "iqr", "q1", "q3", "min", "max", "skewness", "outliers_iqr_count"]

        part1 = uni_df[cols_part1].copy() if all(c in uni_df.columns for c in cols_part1) else uni_df.copy()
        part2 = uni_df[cols_part2].copy() if all(c in uni_df.columns for c in cols_part2) else pd.DataFrame()

        # Anchos: distribuimos para que quepa en A4
        # (si querés más fino, ajustamos por columna)
        w1 = [usable_w * x for x in [0.14, 0.06, 0.08, 0.12, 0.12, 0.10, 0.14]]
        w2 = [usable_w * x for x in [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.18]]

        elements.append(Paragraph("Parte 1", small))
        elements.append(_make_table(part1, w1))
        elements.append(Spacer(1, 0.35 * cm))

        if not part2.empty:
            elements.append(Paragraph("Parte 2", small))
            elements.append(_make_table(part2, w2))
            elements.append(Spacer(1, 0.35 * cm))

    # =========================
    # Gráficas clave
    # =========================
    elements.append(PageBreak())
    elements.append(Paragraph("Gráficas clave (EDA)", h1))
    elements.append(
        Paragraph(
            "A continuación se incluyen visualizaciones seleccionadas del análisis univariante y bivariante.",
            small,
        )
    )
    elements.append(Spacer(1, 0.2 * cm))

    # Imágenes: limitamos alto para que cada bloque sea “bonito”
    max_img_w = usable_w
    max_img_h = page_h * 0.55  # evita que una imagen empuje el caption a otra página

    for p in inputs.plots:
        caption = plot_caption_es(p)
        elements.append(_image_block(p, caption, max_img_w, max_img_h))

    # =========================
    # Contraste de hipótesis (Punto 5)
    # =========================
    elements.append(PageBreak())
    elements.append(Paragraph("Contraste de hipótesis (Punto 5)", h1))
    elements.append(Paragraph(f"Nivel de significación utilizado: α = {alpha:.2f}.", body))

    # En vez de “H□” y cosas raras, usamos texto plano (H0/H1)
    if inputs.hypothesis_md.exists():
        # Mostramos un extracto simple del .md (sin markdown avanzado)
        try:
            txt = inputs.hypothesis_md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        if txt.strip():
            # Sanitizamos a algo corto (evita PDF larguísimo si el md crece)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            excerpt = "\n".join(lines[:80])
            for block in excerpt.split("\n"):
                elements.append(Paragraph(block.replace("H₀", "H0").replace("H₁", "H1"), body))
        else:
            elements.append(Paragraph("No hay contenido disponible en hypothesis_tests.md.", small))
    else:
        elements.append(
            Paragraph(
                f"No se encontró hypothesis_tests.md en {inputs.hypothesis_md}.",
                small,
            )
        )

    # =========================
    # Trazabilidad
    # =========================
    elements.append(Spacer(1, 0.35 * cm))
    elements.append(Paragraph("Trazabilidad (archivos de salida)", h2))
    elements.append(
        Paragraph(
            f"- Gráficas: {PLOTS_DIR}\n"
            f"- Reportes: {REPORTS_DIR}\n"
            f"- Output raíz: {OUTPUT_DIR}",
            mono,
        )
    )

    return output_pdf, elements
