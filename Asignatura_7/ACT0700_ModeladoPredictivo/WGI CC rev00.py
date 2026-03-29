# ============================================================
# PREPROCESAMIENTO DEL DATASET WGI (WORLDWIDE GOVERNANCE INDICATORS)
# Opción B: Control of Corruption (CC)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# ------------------------------------------------------------
# 1. Cargar solo la pestaña de Control of Corruption (cc)
# ------------------------------------------------------------
wgi_cc = pd.read_excel(
    raw_file("wgidataset-2025.xlsx"),
    sheet_name="cc"   # <-- ESTA ES LA PESTAÑA CORRECTA
)

print("Columnas detectadas:")
print(wgi_cc.columns)


# ------------------------------------------------------------
# 2. Filtrar año 2023
# ------------------------------------------------------------
wgi_cc_2023 = wgi_cc[wgi_cc["Year"] == 2023].copy()

print("\nFilas CC del año 2023 (primeras 10):")
print(wgi_cc_2023.head(10))


# ------------------------------------------------------------
# 3. Seleccionar columnas relevantes
# ------------------------------------------------------------
wgi_cc_clean = wgi_cc_2023[[
    "Economy (code)",   # ISO3
    "Governance estimate (approx. -2.5 to +2.5)"  # Valor CC
]].copy()

# Renombrar columnas
wgi_cc_clean.rename(columns={
    "Economy (code)": "iso3",
    "Governance estimate (approx. -2.5 to +2.5)": "cc_2023"
}, inplace=True)


# ------------------------------------------------------------
# 4. Limpiar dataset final
# ------------------------------------------------------------
wgi_cc_final = wgi_cc_clean.dropna(subset=["cc_2023"])

print("\nDataset final listo para unir:")
print(wgi_cc_final.head())
wgi_cc_final.to_csv(
    processed_file("wgi_cc_final.csv"),
    index=False
)