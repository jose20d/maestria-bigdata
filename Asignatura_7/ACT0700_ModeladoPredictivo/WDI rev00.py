# ============================================================
# PREPROCESAMIENTO DEL DATASET WDI (WORLD DEVELOPMENT INDICATORS)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# ------------------------------------------------------------
# 1. Cargar archivo WDI (formato Banco Mundial)
# ------------------------------------------------------------
wdi = pd.read_csv(
    raw_file("wdi.csv"),
    header=0
)

print("Columnas detectadas:")
print(wdi.columns)

# ------------------------------------------------------------
# 2. Lista de indicadores seleccionados (Opción A)
# ------------------------------------------------------------
selected_indicators = {
    "SE.SEC.ENRR": "education_secondary",
    "SE.TER.ENRR": "education_tertiary",
    "SP.DYN.LE00.IN": "life_expectancy",
    "SH.XPD.CHEX.GD.ZS": "health_expenditure",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SL.UEM.TOTL.ZS": "unemployment",
    "EG.ELC.ACCS.ZS": "electricity_access",
    "IT.NET.USER.ZS": "internet_users"
}

# Filtrar solo los indicadores seleccionados
wdi_filtered = wdi[wdi["Indicator Code"].isin(selected_indicators.keys())].copy()

print("\nFilas filtradas (primeras 10):")
print(wdi_filtered.head(10))

# ------------------------------------------------------------
# 3. Seleccionar el año 2023
# ------------------------------------------------------------
year = "2023"
wdi_filtered["value_2023"] = wdi_filtered[str(year)]

# Mantener solo columnas relevantes
wdi_reduced = wdi_filtered[[
    "Country Code",
    "Indicator Code",
    "value_2023"
]].copy()

# ------------------------------------------------------------
# 4. Pivotar para tener un dataset por país
# ------------------------------------------------------------
wdi_pivot = wdi_reduced.pivot_table(
    index="Country Code",
    columns="Indicator Code",
    values="value_2023"
).reset_index()

# Renombrar columnas a nombres amigables
wdi_pivot.rename(columns=selected_indicators, inplace=True)

# Renombrar ISO3
wdi_pivot.rename(columns={"Country Code": "iso3"}, inplace=True)

# ------------------------------------------------------------
# 5. Dataset final limpio
# ------------------------------------------------------------
wdi_final = wdi_pivot.dropna(subset=["iso3"])

print("\nDataset final listo para unir:")
print(wdi_final.head())

# ------------------------------------------------------------
# 6. GUARDAR CSV FINAL
# ------------------------------------------------------------
wdi_final.to_csv(
    processed_file("wdi_final.csv"),
    index=False
)

print("\nArchivo guardado como wdi_final.csv")