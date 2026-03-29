# ============================================================
# MODELO 2 — Predicción de Control of Corruption (CC)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from path_utils import processed_file

# ------------------------------------------------------------
# 1. Cargar dataset
# ------------------------------------------------------------
df = pd.read_csv(
    processed_file("dataset_maestro_2023.csv")
)

# ------------------------------------------------------------
# 2. Selección de variables
# ------------------------------------------------------------
features_cc = [
    "gdp_per_capita", "education_secondary", "education_tertiary",
    "internet_users", "electricity_access", "health_expenditure",
    "unemployment", "life_expectancy", "cpi_2023", "fsi_2023"
]

target_cc = "cc_2023"

X = df[features_cc]
y = df[target_cc]

# ------------------------------------------------------------
# 3. Imputación de faltantes
# ------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# ------------------------------------------------------------
# 4. Escalado
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ------------------------------------------------------------
# 5. Train/Test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 6. Modelos
# ------------------------------------------------------------

# Modelo A: Regresión Lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Modelo B: Random Forest
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ------------------------------------------------------------
# 7. Evaluación
# ------------------------------------------------------------
def evaluar(y_true, y_pred, nombre):
    print(f"\n--- {nombre} ---")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))

evaluar(y_test, y_pred_lr, "Regresión Lineal")
evaluar(y_test, y_pred_rf, "Random Forest")

# ------------------------------------------------------------
# 8. Importancia de variables (Random Forest)
# ------------------------------------------------------------
importancias = pd.DataFrame({
    "variable": features_cc,
    "importancia": rf.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\nImportancia de variables (Random Forest):")
print(importancias)