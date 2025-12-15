# main.py
from __future__ import annotations

from src.data_loading import get_raw_data
from src.cleaning import clean_data, save_clean_data
from src.eda_univariate import run_univariate_eda
from src.eda_bivariate import run_bivariate_eda
from src.stats_tests import run_hypothesis_tests
from src.pdf_report import generate_executive_pdf

def main() -> None:
    # 1. Cargar datos crudos
    df_raw = get_raw_data()
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

    # 6. Contraste de hipótesis (extra)
    print("\nEjecutando contraste de hipótesis...")
    ht_outputs = run_hypothesis_tests(df_clean, alpha=0.05)

    # 7. Información final
    print("\nPipeline completado correctamente.")
    print("\nOutputs generados:")
    for k, v in {**uni_outputs, **bi_outputs, **ht_outputs}.items():
        print(f"- {k}: {v}")

    # 8. Informe ejecutivo (PDF)
    print("\nGenerando informe ejecutivo (PDF)...")
    pdf_path = generate_executive_pdf(alpha=0.05)
    print("Informe generado en:", pdf_path)

if __name__ == "__main__":
    main()
