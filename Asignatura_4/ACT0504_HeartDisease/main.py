# main.py
from __future__ import annotations

from src.data_loading import get_raw_data
from src.cleaning import clean_data, save_clean_data
from src.eda_univariate import run_univariate_eda
from src.eda_bivariate import run_bivariate_eda
from src.pdf_report import generate_executive_pdf
from src.validation import require_columns
from src.run_metadata import write_run_metadata
from src.config import COLUMN_NAMES
from src.hypothesis_tests import run_hypothesis_tests

def main() -> None:
    alpha = 0.05

    # 1. Cargar datos crudos
    df_raw = get_raw_data()
    require_columns(df_raw, COLUMN_NAMES, where="main/get_raw_data")
    print("Shape del dataset crudo:", df_raw.shape)

    # 2. Limpiar datos
    df_clean = clean_data(df_raw)
    print("Shape del dataset limpio:", df_clean.shape)

    # 3. Guardar dataset limpio
    save_clean_data(df_clean)

    # 4. EDA univariante
    print("\nEjecutando análisis univariante...")
    uni_outputs = run_univariate_eda(df_clean)

    # 5. EDA bivariante
    print("\nEjecutando análisis bivariante...")
    bi_outputs = run_bivariate_eda(df_clean)

    # 6. Hipótesis
    print("\nEjecutando contraste de hipótesis...")
    hypo_outputs = run_hypothesis_tests(df_clean, alpha=alpha)

    # 7. Metadata de ejecución
    outputs = {**uni_outputs, **bi_outputs, **hypo_outputs}
    write_run_metadata(df_raw, df_clean, outputs=outputs, alpha=alpha)
    outputs = {
    **uni_outputs,
    **bi_outputs,
    **hypo_outputs,
    }

    # 8. PDF ejecutivo
    print("\nGenerando informe ejecutivo (PDF)...")
    pdf_path = generate_executive_pdf(alpha=alpha)
    print("PDF generado:", pdf_path)

    print("\nPipeline completado correctamente.")

if __name__ == "__main__":
    main()
