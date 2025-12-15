# src/pdf_report.py
from __future__ import annotations

from pathlib import Path
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4

from src.report_builder import ExecutiveReportInputs, build_executive_report_elements, default_executive_report_inputs


def generate_executive_pdf(alpha: float = 0.05, inputs: ExecutiveReportInputs | None = None) -> Path:
    if inputs is None:
        inputs = default_executive_report_inputs()

    output_pdf, elements = build_executive_report_elements(inputs=inputs, alpha=alpha)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    doc.build(elements)
    return output_pdf
