# src/pdf_report.py
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from .config import OUTPUT_DIR
from .report_builder import ExecutiveReportInputs, build_executive_report_elements


def generate_executive_pdf(alpha: float = 0.05) -> Path:
    inputs = ExecutiveReportInputs(
        univariate_numeric_csv=OUTPUT_DIR / "reports" / "univariate_numeric_summary.csv",
        hypothesis_md=OUTPUT_DIR / "reports" / "hypothesis_tests.md",
        plots_dir=OUTPUT_DIR / "plots",
        output_pdf=OUTPUT_DIR / "reports" / "INFORME_EJECUTIVO_ACT0504.pdf",
        dataset_name="Heart Disease (Cleveland, UCI)",
    )

    output_pdf, elements = build_executive_report_elements(inputs=inputs, alpha=alpha)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Informe Ejecutivo – Resultados del Análisis",
    )
    doc.build(elements)
    return output_pdf
