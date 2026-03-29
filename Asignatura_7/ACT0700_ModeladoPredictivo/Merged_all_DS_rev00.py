# ============================================================
# UNION DE TODOS LOS DATASETS PARA EL MODELO (VERSIÓN CORREGIDA)
# ============================================================

import pandas as pd
from path_utils import processed_file

# ------------------------------------------------------------
# 0. Ruta base donde están todos los CSV finales
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. Cargar datasets individuales
# ------------------------------------------------------------
# Cada uno de estos CSV ya fue preprocesado en su script correspondiente.
# Todos tienen una columna clave común: "iso3".

wdi = pd.read_csv(processed_file("wdi_final.csv"))          # Indicadores WDI (socioeconómicos)
wgi_ge = pd.read_csv(processed_file("wgi_ge_final.csv"))    # Government Effectiveness (GE)
wgi_cc = pd.read_csv(processed_file("wgi_cc_final.csv"))    # Control of Corruption (CC)
wgi_gc = pd.read_csv(processed_file("wgi_gc_final.csv"))    # Índice compuesto WGI (6 dimensiones)
cpi = pd.read_csv(processed_file("cpi_final.csv"))          # CPI 2023
gdp = pd.read_csv(processed_file("gdp_final.csv"))          # PIB total
fsi = pd.read_csv(processed_file("fsi_final.csv"))          # Fragile States Index (FSI)

print("Tamaños iniciales:")
print("WDI:", wdi.shape)
print("WGI_GE:", wgi_ge.shape)
print("WGI_CC:", wgi_cc.shape)
print("WGI_GC:", wgi_gc.shape)
print("CPI:", cpi.shape)
print("GDP:", gdp.shape)
print("FSI:", fsi.shape)

# ------------------------------------------------------------
# 2. Empezar base con WDI
# ------------------------------------------------------------
# Usamos WDI como base porque contiene la mayor cantidad de variables explicativas.
master = wdi.copy()

# ------------------------------------------------------------
# 3. Unir WGI (GE, CC, Composite)
# ------------------------------------------------------------
# IMPORTANTE:
# - wgi_ge tiene columna "ge_2023"
# - wgi_cc tiene columna "cc_2023"
# - wgi_gc tiene columnas "va_2023", "pv_2023", "ge_2023", "rq_2023", "rl_2023", "cc_2023", "governance_composite"
# Al hacer merges sucesivos, Pandas creará columnas duplicadas con sufijos _x y _y.

# Merge con Government Effectiveness (GE)
master = master.merge(wgi_ge, on="iso3", how="inner")   # añade ge_2023

# Merge con Control of Corruption (CC)
master = master.merge(wgi_cc, on="iso3", how="inner")   # añade cc_2023

# Merge con índice compuesto WGI (6 dimensiones + governance_composite)
master = master.merge(wgi_gc, on="iso3", how="inner")   # añade va_2023, pv_2023, ge_2023, rq_2023, rl_2023, cc_2023, governance_composite

# ------------------------------------------------------------
# 4. Renombrar columnas duplicadas de WGI
# ------------------------------------------------------------
# Después de los merges anteriores, las columnas ge_2023 y cc_2023 aparecen dos veces:
# - ge_2023_x: viene de wgi_ge_final (GE "oficial")
# - ge_2023_y: viene de wgi_gc_final (la misma dimensión usada dentro del índice compuesto)
# - cc_2023_x: viene de wgi_cc_final (CC "oficial")
# - cc_2023_y: viene de wgi_gc_final (la misma dimensión usada dentro del índice compuesto)
#
# Para evitar el KeyError y tener nombres claros:
# - ge_2023  → la usaremos como target principal de GE
# - cc_2023  → la usaremos como target principal de CC
# - ge_2023_composite y cc_2023_composite → versiones internas usadas en el índice compuesto

master.rename(columns={
    "ge_2023_x": "ge_2023",                 # Government Effectiveness "oficial"
    "ge_2023_y": "ge_2023_composite",       # GE dentro del índice compuesto
    "cc_2023_x": "cc_2023",                 # Control of Corruption "oficial"
    "cc_2023_y": "cc_2023_composite"        # CC dentro del índice compuesto
}, inplace=True)

# (Opcional pero útil para depuración: ver columnas después del renombrado)
# print("\nColumnas después de renombrar duplicadas WGI:")
# print(master.columns.tolist())

# ------------------------------------------------------------
# 5. Unir CPI, GDP, FSI
# ------------------------------------------------------------
# Estos datasets no generan colisiones de nombres con las columnas anteriores.
master = master.merge(cpi, on="iso3", how="inner")   # añade cpi_2023
master = master.merge(gdp, on="iso3", how="inner")   # añade gdp
master = master.merge(fsi, on="iso3", how="inner")   # añade fsi_2023

print("\nTamaño después de todos los merges:", master.shape)

print("\nPrimeras filas del dataset maestro:")
print(master.head())

# ------------------------------------------------------------
# 6. Limpiar filas con NaNs en los targets del modelo
# ------------------------------------------------------------
# Targets principales para el modelo:
# - ge_2023: Government Effectiveness (indicador WGI)
# - cc_2023: Control of Corruption (indicador WGI)
# - governance_composite: índice compuesto de gobernanza (promedio de las 6 dimensiones)
#
# Es fundamental que estas columnas EXISTAN (ya existen tras el renombrado)
# y que no tengan NaNs en las filas que usaremos para entrenar el modelo.

targets = ["ge_2023", "cc_2023", "governance_composite"]

# Si alguna de estas columnas no existiera, aquí daría KeyError.
# Como ya renombramos correctamente, esta línea debe ejecutarse sin errores.
master_clean = master.dropna(subset=targets)

print("\nTamaño después de limpiar NaNs en targets:", master_clean.shape)

# ------------------------------------------------------------
# 7. Guardar dataset maestro final
# ------------------------------------------------------------
# Este es el dataset consolidado y listo para:
# - análisis exploratorio
# - selección de variables
# - entrenamiento de modelos de regresión / clasificación
# - visualizaciones para el TFM

output_path = processed_file("dataset_maestro_2023.csv")
master_clean.to_csv(output_path, index=False)

print(f"\nDataset maestro guardado como '{output_path}'")