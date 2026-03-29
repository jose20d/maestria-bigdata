# ============================================================
# PREPROCESAMIENTO DEL DATASET FSI (CORREGIDO FINAL)
# ============================================================

import pandas as pd
import pycountry
from path_utils import raw_file, processed_file

# 1. Cargar archivo con encabezados reales
fsi = pd.read_excel(
    raw_file("fsi.xlsx"),
    header=0
)

print("Columnas detectadas:")
print(fsi.columns)

# 2. Identificar columnas relevantes
column_country = "Country"
column_year = "Year"
column_score = "Total"   # Puntaje total del FSI

# 3. Filtrar año 2023
fsi_2023 = fsi[fsi[column_year] == 2023].copy()

print("\nFilas del año 2023:")
print(fsi_2023.head())

# 4. Convertir nombres de países a ISO3
def country_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

fsi_2023["iso3"] = fsi_2023[column_country].apply(country_to_iso3)

# 5. Eliminar filas sin ISO3 o sin puntaje
fsi_clean = fsi_2023.dropna(subset=["iso3", column_score])

# 6. Dataset final
fsi_final = fsi_clean[["iso3", column_score]].copy()
fsi_final.rename(columns={column_score: "fsi_2023"}, inplace=True)

print("\nDataset final listo para unir:")
print(fsi_final.head())

# 7. GUARDAR CSV FINAL
fsi_final.to_csv(
    processed_file("fsi_final.csv"),
    index=False
)

print("\nArchivo guardado como fsi_final.csv")