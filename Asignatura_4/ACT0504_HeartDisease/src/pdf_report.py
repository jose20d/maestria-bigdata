# src/pdf_report.py
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from src.report_builder import build_executive_report_elements


def generate_executive_pdf(alpha: float = 0.05) -> Path:
    output_pdf, elements = build_executive_report_elements(alpha=alpha)

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
