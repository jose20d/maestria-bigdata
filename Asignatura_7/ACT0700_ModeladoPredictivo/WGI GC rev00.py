# ============================================================
# PREPROCESAMIENTO DEL DATASET WGI (WORLDWIDE GOVERNANCE INDICATORS)
# Opción C: Índice compuesto de gobernanza (promedio de las 6 dimensiones)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# Ruta del archivo
FILE_PATH = raw_file("wgidataset-2025.xlsx")

# Las 6 pestañas del WGI
DIMENSIONS = {
    "va": "va_2023",
    "pv": "pv_2023",
    "ge": "ge_2023",
    "rq": "rq_2023",
    "rl": "rl_2023",
    "cc": "cc_2023"
}

# Diccionario para almacenar los dataframes procesados
dfs = {}

# ------------------------------------------------------------
# 1. Cargar cada pestaña, filtrar 2023 y extraer ISO3 + estimate
# ------------------------------------------------------------
for sheet, colname in DIMENSIONS.items():
    df = pd.read_excel(FILE_PATH, sheet_name=sheet)

    # Filtrar año 2023
    df_2023 = df[df["Year"] == 2023].copy()

    # Extraer ISO3 + estimate
    df_clean = df_2023[[
        "Economy (code)",
        "Governance estimate (approx. -2.5 to +2.5)"
    ]].copy()

    # Renombrar columnas
    df_clean.rename(columns={
        "Economy (code)": "iso3",
        "Governance estimate (approx. -2.5 to +2.5)": colname
    }, inplace=True)

    # Guardar
    dfs[sheet] = df_clean

# ------------------------------------------------------------
# 2. Unir las 6 dimensiones por ISO3
# ------------------------------------------------------------
wgi_merged = dfs["va"]

for sheet in ["pv", "ge", "rq", "rl", "cc"]:
    wgi_merged = wgi_merged.merge(dfs[sheet], on="iso3", how="outer")

# ------------------------------------------------------------
# 3. Calcular el índice compuesto
# ------------------------------------------------------------
wgi_merged["governance_composite"] = wgi_merged[
    ["va_2023", "pv_2023", "ge_2023", "rq_2023", "rl_2023", "cc_2023"]
].mean(axis=1)

# ------------------------------------------------------------
# 4. Limpiar dataset final
# ------------------------------------------------------------
wgi_final = wgi_merged.dropna(subset=["governance_composite"])

print("\nDataset final con índice compuesto (primeras filas):")
print(wgi_final.head())
wgi_final.to_csv(
    processed_file("wgi_gc_final.csv"),
    index=False
)