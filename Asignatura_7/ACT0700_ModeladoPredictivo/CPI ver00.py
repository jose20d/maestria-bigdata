# =========================================================================================
# PREPROCESAMIENTO DEL DATASET CPI 2023 PARA CREAR UN SISTEMA DE PREDICCION TIPO REGRESION
# =========================================================================================

# ------------------------------------------------------------
# 1. Cargar el archivo CPI 2023
# ------------------------------------------------------------
# Ajusta el nombre del archivo según corresponda
# cpi = pd.read_excel(raw_file("CPI2023_Global_Results__Trends.xlsx"))

# ============================================================
# PREPROCESAMIENTO DEL DATASET CPI 2023 (CORREGIDO FINAL)
# ============================================================

import pandas as pd
from path_utils import raw_file, processed_file

# 1. Cargar archivo usando la fila correcta como encabezado
cpi = pd.read_excel(
    raw_file("CPI2023_Global_Results__Trends.xlsx"),
    header=3
)

print("Columnas detectadas:")
print(cpi.columns)

# 2. Seleccionar columnas relevantes
column_country = "Country / Territory"
column_iso3 = "ISO3"
column_cpi2023 = "CPI score 2023"   # ← NOMBRE CORRECTO

# Verificación
print("\nVerificando columna CPI:")
print(column_cpi2023 in cpi.columns)

# Seleccionar columnas
cpi_2023 = cpi[[column_country, column_iso3, column_cpi2023]].copy()

# 3. Renombrar columnas
cpi_2023.rename(columns={
    column_country: "country",
    column_iso3: "iso3",
    column_cpi2023: "cpi_2023"
}, inplace=True)

# 4. Limpiar filas vacías
cpi_2023_clean = cpi_2023.dropna(subset=["iso3", "cpi_2023"])

print("\nDataset limpio:")
print(cpi_2023_clean.head())

# 5. Dataset final
cpi_2023_final = cpi_2023_clean[["iso3", "cpi_2023"]]

print("\nDataset final listo para unir:")
print(cpi_2023_final.head())

# 6. GUARDAR CSV FINAL
cpi_2023_final.to_csv(
    processed_file("cpi_final.csv"),
    index=False
)

print("\nArchivo guardado como cpi_final.csv")