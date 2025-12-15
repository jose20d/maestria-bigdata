# src/report_builder.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

from .config import PROJECT_ROOT, OUTPUT_DIR

# ---------------------------------------------------------------------
# Entradas del reporte
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutiveReportInputs:
    # CSVs (salidas) que ya genera el pipeline
    univariate_numeric_csv: Path
    hypothesis_md: Path | None = None

    # Carpeta de plots generados
    plots_dir: Path = OUTPUT_DIR / "plots"

    # PDF destino
    output_pdf: Path = OUTPUT_DIR / "reports" / "INFORME_EJECUTIVO_ACT0504.pdf"

    # Metadata
    dataset_name: str = "Heart Disease (Cleveland, UCI)"


# ---------------------------------------------------------------------
# Estilos básicos
# ---------------------------------------------------------------------
_STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_STYLES["Heading1"], fontSize=22, leading=26, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=_STYLES["Heading2"], fontSize=14, leading=18, spaceAfter=8)
P  = ParagraphStyle("P", parent=_STYLES["BodyText"], fontSize=11, leading=14, spaceAfter=8)
CAP = ParagraphStyle("CAP", parent=_STYLES["BodyText"], fontSize=10, leading=12, textColor=colors.grey)

# ---------------------------------------------------------------------
# Traducción “bonita” para captions (evita Age box, Condition Bar, etc.)
# ---------------------------------------------------------------------
CAPTION_ES = {
    "age": "Edad (años)",
    "trestbps": "Presión arterial en reposo",
    "chol": "Colesterol",
    "thalach": "Frecuencia cardiaca máxima",
    "oldpeak": "Depresión ST (oldpeak)",

    "sex": "Sexo",
    "cp": "Tipo de dolor torácico",
    "fbs": "Glucosa en ayunas (fbs)",
    "restecg": "ECG en reposo (restecg)",
    "exang": "Angina por ejercicio (exang)",
    "slope": "Pendiente ST (slope)",
    "thal": "Thal",
    "ca": "Vasos principales (ca)",
    "condition": "Condición (0=No, 1=Sí)",
}

PLOT_NAME_ES = {
    # univariate numeric
    "age_box": "Diagrama de caja: Edad",
    "age_hist": "Histograma: Edad",
    "age_density": "Densidad: Edad",

    "trestbps_box": "Diagrama de caja: Presión",
    "trestbps_hist": "Histograma: Presión",
    "trestbps_density": "Densidad: Presión",

    "chol_box": "Diagrama de caja: Colesterol",
    "chol_hist": "Histograma: Colesterol",
    "chol_density": "Densidad: Colesterol",

    "thalach_box": "Diagrama de caja: Frecuencia cardiaca máxima",
    "thalach_hist": "Histograma: Frecuencia cardiaca máxima",
    "thalach_density": "Densidad: Frecuencia cardiaca máxima",

    "oldpeak_box": "Diagrama de caja: Oldpeak",
    "oldpeak_hist": "Histograma: Oldpeak",
    "oldpeak_density": "Densidad: Oldpeak",

    # bivariate
    "cp_vs_condition_stacked_bar": "Barras apiladas (%): Dolor torácico vs Condición",
    "sex_vs_condition_stacked_bar": "Barras apiladas (%): Sexo vs Condición",
    "chol_by_condition_box": "Boxplot: Colesterol por condición",
    "thalach_by_condition_box": "Boxplot: Thalach por condición",
    "age_vs_thalach_scatter": "Dispersión: Edad vs Thalach",
    "trestbps_vs_chol_scatter": "Dispersión: Presión vs Colesterol",

    # categorical bars
    "condition_bar": "Gráfico de barras: Condición",
    "sex_bar": "Gráfico de barras: Sexo",
    "cp_bar": "Gráfico de barras: Dolor torácico",
    "fbs_bar": "Gráfico de barras: FBS",
    "restecg_bar": "Gráfico de barras: Restecg",
    "exang_bar": "Gráfico de barras: Exang",
    "slope_bar": "Gráfico de barras: Slope",
    "thal_bar": "Gráfico de barras: Thal",
    "ca_bar": "Gráfico de barras: Ca",
}


# ---------------------------------------------------------------------
# Helpers: tablas y grids de imágenes (clave para bajar páginas)
# ---------------------------------------------------------------------
def _safe_exists(p: Path) -> bool:
    return p is not None and Path(p).exists()


def _wrap(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _sp(h: float) -> Spacer:
    return Spacer(1, h)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _shorten_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Para evitar que "outliers_iqr_count" reviente el ancho
    mapping = {
        "outliers_iqr_count": "outliers_iqr",
        "skewness": "asimetría",
    }
    cols = [mapping.get(c, c) for c in df.columns]
    df = df.copy()
    df.columns = cols
    return df


def _df_to_table(df: pd.DataFrame, max_width: float) -> Table:
    # Convierte DF en tabla con estilo y ajuste de ancho
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, repeatRows=1)

    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )

    # Ajuste simple de anchos: reparte el ancho total
    ncols = len(df.columns)
    col_width = max_width / max(ncols, 1)
    t._argW = [col_width] * ncols
    return t


def _plot_caption_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    # "age_box" => busca directo, si no usa fallback
    if stem in PLOT_NAME_ES:
        return PLOT_NAME_ES[stem]
    # fallback: capitaliza el stem
    return stem.replace("_", " ").strip().capitalize()


def _image_cell(img_path: Path, cell_w: float, cell_h: float) -> list:
    # Imagen + caption debajo (todo en español)
    img = Image(str(img_path))
    img._restrictSize(cell_w, cell_h)
    cap = _wrap(_plot_caption_from_filename(img_path.name), CAP)
    return [img, _sp(2), cap]


def _images_grid(
    title: str,
    image_paths: list[Path],
    ncols: int = 2,
    cell_h: float = 8.0 * cm,
) -> KeepTogether:
    """
    Crea un bloque compacto: título + tabla de imágenes (ncols) con captions.
    Evita que título y gráficos se separen en páginas.
    """
    # Filtra solo existentes
    imgs = [p for p in image_paths if _safe_exists(p)]
    if not imgs:
        return KeepTogether([_wrap(title, H2), _wrap("No se encontraron gráficos para esta sección.", P)])

    page_w, page_h = A4
    usable_w = page_w - (40 + 40)  # márgenes (coincide con SimpleDocTemplate)
    cell_w = usable_w / ncols - 6

    rows: list[list] = []
    row: list = []

    # arma celdas
    for p in imgs:
        row.append(_image_cell(p, cell_w=cell_w, cell_h=cell_h))
        if len(row) == ncols:
            rows.append(row)
            row = []
    if row:
        # rellena la fila final
        while len(row) < ncols:
            row.append("")
        rows.append(row)

    grid = Table(rows, colWidths=[usable_w / ncols] * ncols)
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return KeepTogether([_wrap(title, H2), grid, _sp(10)])


# ---------------------------------------------------------------------
# Construcción del reporte (elementos ReportLab)
# ---------------------------------------------------------------------
def build_executive_report_elements(inputs: ExecutiveReportInputs, alpha: float = 0.05):
    """
    Retorna (output_pdf, elements) para que pdf_report.py lo construya.
    """
    out_pdf = inputs.output_pdf
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements: list = []

    # Portada / encabezado
    elements.append(_wrap("Informe Ejecutivo – Resultados del Análisis", H1))
    elements.append(_wrap(f"<b>Dataset:</b> {inputs.dataset_name}", P))
    elements.append(_wrap(f"<b>Generado:</b> {now}", P))
    elements.append(_sp(12))

    elements.append(_wrap("Resumen", H2))
    elements.append(
        _wrap(
            "Este documento resume los resultados principales del análisis exploratorio (EDA) y del "
            "contraste de hipótesis. Incluye visualizaciones clave y métricas para una lectura rápida "
            "orientada a toma de decisiones.",
            P,
        )
    )
    elements.append(_sp(6))

    # ----------------------------
    # Tabla univariante numérica (partida en 2 y headers acortados)
    # ----------------------------
    elements.append(_wrap("EDA univariante – resumen numérico", H2))
    elements.append(_wrap("Extracto de medidas descriptivas principales.", P))

    df_uni = _read_csv(inputs.univariate_numeric_csv)
    df_uni = _shorten_headers(df_uni)

    # Partimos tabla para que no explote el ancho
    # Parte 1: columnas básicas
    cols_1 = ["variable", "n", "missing", "mean", "median", "mode", "variance"]
    cols_1 = [c for c in cols_1 if c in df_uni.columns]
    part1 = df_uni[cols_1].copy()

    # Parte 2: el resto
    cols_2 = [c for c in df_uni.columns if c not in cols_1]
    part2 = df_uni[cols_2].copy() if cols_2 else pd.DataFrame()

    page_w, _ = A4
    usable_w = page_w - (40 + 40)

    elements.append(_wrap("<i>Parte 1</i>", CAP))
    elements.append(_df_to_table(part1, max_width=usable_w))
    elements.append(_sp(12))

    if not part2.empty:
        elements.append(_wrap("<i>Parte 2</i>", CAP))
        elements.append(_df_to_table(part2, max_width=usable_w))
        elements.append(_sp(8))

    elements.append(PageBreak())

    # ----------------------------
    # Gráficas clave: AGRUPADAS (baja páginas)
    # ----------------------------
    elements.append(_wrap("Gráficas clave (EDA)", H2))

    plots = inputs.plots_dir

    # Univariante numérico (2x2 por página aprox)
    uni_num = [
        plots / "univariate" / "numeric" / "age_box.png",
        plots / "univariate" / "numeric" / "age_density.png",
        plots / "univariate" / "numeric" / "chol_box.png",
        plots / "univariate" / "numeric" / "chol_density.png",
    ]
    elements.append(_images_grid("Univariante (numéricas) – distribución y outliers", uni_num, ncols=2, cell_h=7.5 * cm))

    # Univariante categórico
    uni_cat = [
        plots / "univariate" / "categorical" / "condition_bar.png",
        plots / "univariate" / "categorical" / "sex_bar.png",
        plots / "univariate" / "categorical" / "cp_bar.png",
        plots / "univariate" / "categorical" / "exang_bar.png",
    ]
    elements.append(_images_grid("Univariante (categóricas) – frecuencias", uni_cat, ncols=2, cell_h=7.0 * cm))

    # Bivariante (cat-cat y num-cat)
    bi_group1 = [
        plots / "bivariate" / "cat_cat" / "cp_vs_condition_stacked_bar.png",
        plots / "bivariate" / "cat_cat" / "sex_vs_condition_stacked_bar.png",
        plots / "bivariate" / "num_cat" / "chol_by_condition_box.png",
        plots / "bivariate" / "num_cat" / "thalach_by_condition_box.png",
    ]
    elements.append(_images_grid("Bivariante – asociaciones y comparaciones por condición", bi_group1, ncols=2, cell_h=7.0 * cm))

    # Bivariante num-num (scatter)
    bi_group2 = [
        plots / "bivariate" / "num_num" / "age_vs_thalach_scatter.png",
        plots / "bivariate" / "num_num" / "trestbps_vs_chol_scatter.png",
    ]
    elements.append(_images_grid("Bivariante (numéricas) – dispersión", bi_group2, ncols=2, cell_h=7.5 * cm))

    elements.append(PageBreak())

    # ----------------------------
    # Hipótesis (si existe md generado)
    # ----------------------------
    if inputs.hypothesis_md and _safe_exists(inputs.hypothesis_md):
        elements.append(_wrap("Contraste de hipótesis (Punto 5)", H2))
        elements.append(_wrap(f"Nivel de significación utilizado: α = {alpha:.2f}.", P))

        # MD simple: tratamos líneas como párrafos (sin “H□” ni símbolos raros)
        text = inputs.hypothesis_md.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                elements.append(_sp(6))
                continue

            # Limpia bullets markdown
            line = line.replace("**", "")
            if line.startswith("#"):
                line = line.lstrip("#").strip()
                elements.append(_wrap(line, H2))
            else:
                # reemplazo seguro de H0/H1 con subíndices usando HTML de ReportLab
                line = (
                    line.replace("H0", "H<sub>0</sub>")
                        .replace("H1", "H<sub>1</sub>")
                        .replace("H₀", "H<sub>0</sub>")
                        .replace("H₁", "H<sub>1</sub>")
                )
                elements.append(_wrap(line, P))

    # Cierre
    elements.append(_sp(10))
    elements.append(_wrap("Fin del informe.", CAP))

    return out_pdf, elements
