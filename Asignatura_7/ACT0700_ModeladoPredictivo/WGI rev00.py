# ============================================================
# PREPROCESAMIENTO DEL DATASET WGI (WORLDWIDE GOVERNANCE INDICATORS)
# Opción A: Government Effectiveness (GE)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# ------------------------------------------------------------
# 1. Cargar solo la pestaña de Government Effectiveness (ge)
# ------------------------------------------------------------
wgi_ge = pd.read_excel(
    raw_file("wgidataset-2025.xlsx"),
    sheet_name="ge"   # <-- ESTA ES LA PESTAÑA CORRECTA
)

print("Columnas detectadas:")
print(wgi_ge.columns)


# ------------------------------------------------------------
# 2. Filtrar año 2023
# ------------------------------------------------------------
wgi_ge_2023 = wgi_ge[wgi_ge["Year"] == 2023].copy()

print("\nFilas GE del año 2023 (primeras 10):")
print(wgi_ge_2023.head(10))


# ------------------------------------------------------------
# 3. Seleccionar columnas relevantes
# ------------------------------------------------------------
wgi_ge_clean = wgi_ge_2023[[
    "Economy (code)",   # ISO3
    "Governance estimate (approx. -2.5 to +2.5)"  # Valor GE
]].copy()

# Renombrar columnas
wgi_ge_clean.rename(columns={
    "Economy (code)": "iso3",
    "Governance estimate (approx. -2.5 to +2.5)": "ge_2023"
}, inplace=True)


# ------------------------------------------------------------
# 4. Limpiar dataset final
# ------------------------------------------------------------
wgi_final = wgi_ge_clean.dropna(subset=["ge_2023"])

print("\nDataset final listo para unir:")
print(wgi_final.head())
wgi_final.to_csv(
    processed_file("wgi_ge_final.csv"),
    index=False
)