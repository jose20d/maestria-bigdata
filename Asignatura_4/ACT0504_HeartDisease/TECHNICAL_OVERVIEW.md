# TECHNICAL_OVERVIEW.md  
# (ES / EN)

# 1. Objetivo técnico / Technical Objective

Este documento describe en detalle el pipeline técnico utilizado para analizar el dataset de Heart Disease.  
This document describes in detail the technical pipeline used to analyze the Heart Disease dataset.

---

# 2. Fuentes de datos / Data Sources

- Kaggle dataset: cherngs/heart-disease-cleveland-uci  
- Archivo procesado en `data/raw/heart.csv`  

---

# 3. Preprocesamiento / Preprocessing

Incluye / Includes:

### • Lectura y validación del dataset  
### • Conversión de tipos (Int64, Float64, string)  
### • Limpieza mediante reglas configurables en `cleaning_config.json`  
### • Tratamiento de outliers  
### • Normalización opcional  
### • Feature engineering  

---

# 4. Archivo de configuración de limpieza / Cleaning Configuration File

El archivo `cleaning_config.json` permite ajustar:  
- Límites min/max por variable  
- Estrategias de limpieza (drop, set_na, imputación)  
- Conversión de tipos  
- Reglas específicas de negocio  

The `cleaning_config.json` file allows adjusting:  
- Min/max limits  
- Cleaning strategies  
- Type conversion  
- Domain‑specific rules  

---

# 5. Flujo del pipeline / Pipeline Flow

1. `data_loading.py` descarga y valida los datos  
2. `cleaning.py` aplica las reglas del JSON  
3. Se generan columnas derivadas  
4. Se exporta el dataset limpio  
5. `eda.py` y `plotting.py` generan análisis y gráficos  

---

# 6. Output final / Final Output

El dataset limpio se guarda en:  
`data/processed/heart_disease_clean.csv`

A clean dataset is saved to:  
`data/processed/heart_disease_clean.csv`
