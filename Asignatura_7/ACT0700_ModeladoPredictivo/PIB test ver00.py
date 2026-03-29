# ============================================================
# PREPROCESAMIENTO DEL DATASET DE PIB (TU ARCHIVO)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# 1. Cargar el archivo (encabezados están en la fila 0)
gdp = pd.read_excel(
    raw_file("worldbank_gdp.xlsx"),
    header=0
)

print("Columnas detectadas:")
print(gdp.columns)

# 2. Renombrar columnas a nombres estándar
gdp.rename(columns={
    "Pais": "country",
    "Codigo_ISO": "iso3",
    "Anio": "year",
    "PIB_USD": "gdp"
}, inplace=True)

# 3. Filtrar el año 2023
gdp_2023 = gdp[gdp["year"] == 2023].copy()

print("\nFilas para el año 2023:")
print(gdp_2023.head())

# 4. Eliminar filas sin ISO3 o sin PIB
gdp_clean = gdp_2023.dropna(subset=["iso3", "gdp"])

# 5. Dataset final listo para unir
gdp_final = gdp_clean[["iso3", "gdp"]]

print("\nDataset final listo para unir:")
print(gdp_final.head())

# 6. GUARDAR CSV FINAL
gdp_final.to_csv(
    processed_file("gdp_final.csv"),
    index=False
)

print("\nArchivo guardado como gdp_final.csv")