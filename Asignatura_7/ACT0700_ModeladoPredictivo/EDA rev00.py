# ============================================================
# EDA DEL DATASET MAESTRO 2023 — GRÁFICOS MEJORADOS
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from path_utils import processed_file

sns.set(style="whitegrid")

# ------------------------------------------------------------
# 1. Cargar dataset maestro
# ------------------------------------------------------------
df = pd.read_csv(
    processed_file("dataset_maestro_2023.csv")
)

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

# ------------------------------------------------------------
# 2. Histogramas individuales (claros y legibles)
# ------------------------------------------------------------
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=25, color="steelblue")
    plt.title(f"Distribución de {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 3. Boxplots por grupos para evitar distorsión
# ------------------------------------------------------------

# Grupo 1 — Variables socioeconómicas
socio = [
    "electricity_access", "internet_users", "gdp_per_capita",
    "education_secondary", "education_tertiary",
    "health_expenditure", "unemployment", "life_expectancy"
]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[socio])
plt.xticks(rotation=45)
plt.title("Boxplots — Variables socioeconómicas")
plt.tight_layout()
plt.show()

# Grupo 2 — Indicadores WGI (misma escala)
wgi_vars = [
    "ge_2023", "cc_2023", "va_2023", "pv_2023",
    "rq_2023", "rl_2023", "governance_composite"
]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[wgi_vars])
plt.xticks(rotation=45)
plt.title("Boxplots — Indicadores WGI")
plt.tight_layout()
plt.show()

# Grupo 3 — Riesgo y corrupción
risk = ["cpi_2023", "fsi_2023"]

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[risk])
plt.xticks(rotation=45)
plt.title("Boxplots — CPI y FSI")
plt.tight_layout()
plt.show()

# Grupo 4 — GDP (log-transform para visualizar)
plt.figure(figsize=(8, 5))
sns.boxplot(x=np.log10(df["gdp"]))
plt.title("Boxplot — GDP (log10 transform)")
plt.xlabel("log10(GDP)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. Heatmap mejorado (más legible y elegante)
# ------------------------------------------------------------
corr = df[numeric_cols].corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=.5,
    cbar_kws={"shrink": .8}
)

plt.title("Matriz de correlación (mejorada)", fontsize=16)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Scatterplots clave con regresión
# ------------------------------------------------------------
key_vars = [
    "gdp_per_capita", "education_secondary", "education_tertiary",
    "internet_users", "electricity_access", "cpi_2023", "fsi_2023"
]

for var in key_vars:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=var, y="ge_2023")
    sns.regplot(data=df, x=var, y="ge_2023", scatter=False, color="red")
    plt.title(f"{var} vs GE")
    plt.tight_layout()
    plt.show()

for var in key_vars:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=var, y="cc_2023")
    sns.regplot(data=df, x=var, y="cc_2023", scatter=False, color="red")
    plt.title(f"{var} vs CC")
    plt.tight_layout()
    plt.show()