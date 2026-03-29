# ============================================================
# SCRIPT MAESTRO — MODELO 1 (GE) Y MODELO 2 (CC)
# ============================================================
# Incluye:
# - Entrenamiento de modelos (Regresión Lineal y Random Forest)
# - Gráficos de importancia de variables (GE y CC)
# - Validación cruzada (K-Fold) para GE y CC
# - Optimización de hiperparámetros con GridSearchCV
# - Tablas de resumen de modelos (GE y CC)
# - Tabla comparativa de importancia de variables (GE vs CC)
# - Gráficos de predicción vs valor real (GE y CC)
# ============================================================

# -----------------------------
# 0. Importación de librerías
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from path_utils import processed_file

sns.set(style="whitegrid")

# -----------------------------
# 1. Carga de datos y definición de features
# -----------------------------
# Cargar dataset maestro
df = pd.read_csv(
    processed_file("dataset_maestro_2023.csv")
)

# Definir conjunto común de variables explicativas (sin otros WGI)
features = [
    "gdp_per_capita",      # Desarrollo económico
    "education_secondary", # Educación secundaria
    "education_tertiary",  # Educación terciaria
    "internet_users",      # Usuarios de internet
    "electricity_access",  # Acceso a electricidad
    "health_expenditure",  # Gasto en salud
    "unemployment",        # Desempleo
    "life_expectancy",     # Esperanza de vida
    "cpi_2023",            # Índice de percepción de corrupción
    "fsi_2023"             # Fragile States Index
]

# -----------------------------
# 2. Función genérica de entrenamiento (GE o CC)
# -----------------------------
def entrenar_modelo(target_name):
    """
    Entrena dos modelos (Regresión Lineal y Random Forest) para un target dado.
    Devuelve métricas y el modelo Random Forest entrenado.
    """
    # Matriz de características y target
    X = df[features]
    y = df[target_name]

    # 2.1. Imputación de valores faltantes (mediana)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # 2.2. Escalado de variables (media 0, desviación 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # 2.3. División en train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 2.4. Modelo A: Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 2.5. Modelo B: Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 2.6. Función interna para calcular métricas
    def metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }

    return {
        "lr_metrics": metrics(y_test, y_pred_lr),
        "rf_metrics": metrics(y_test, y_pred_rf),
        "rf_model": rf
    }

# -----------------------------
# 3. Entrenamiento de Modelo 1 (GE) y Modelo 2 (CC)
# -----------------------------
res_ge = entrenar_modelo("ge_2023")   # Modelo 1: Government Effectiveness
res_cc = entrenar_modelo("cc_2023")   # Modelo 2: Control of Corruption

print("\n=== MÉTRICAS MODELO 1 — GE ===")
print("Regresión Lineal:", res_ge["lr_metrics"])
print("Random Forest:", res_ge["rf_metrics"])

print("\n=== MÉTRICAS MODELO 2 — CC ===")
print("Regresión Lineal:", res_cc["lr_metrics"])
print("Random Forest:", res_cc["rf_metrics"])

# -----------------------------
# 4. Gráficos de importancia de variables (GE y CC)
# -----------------------------
def plot_importance(rf_model, title):
    """
    Genera un gráfico de barras con la importancia de variables
    de un modelo Random Forest dado.
    """
    importancias = pd.DataFrame({
        "variable": features,
        "importancia": rf_model.feature_importances_
    }).sort_values(by="importancia", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=importancias,
        x="importancia",
        y="variable",
        palette="viridis"
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Importancia relativa")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

    return importancias

imp_ge = plot_importance(
    res_ge["rf_model"],
    "Importancia de variables — Modelo GE (Random Forest)"
)

imp_cc = plot_importance(
    res_cc["rf_model"],
    "Importancia de variables — Modelo CC (Random Forest)"
)

# -----------------------------
# 5. Validación cruzada (K-Fold) para GE y CC
# -----------------------------
def kfold_rf(target_name, n_splits=5):
    """
    Aplica validación cruzada K-Fold a un Random Forest
    para un target concreto y muestra los R² por fold.
    """
    X = df[features]
    y = df[target_name]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    rf = RandomForestRegressor(n_estimators=500, random_state=42)

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        rf,
        X_scaled,
        y,
        cv=kf,
        scoring="r2"
    )

    print(f"\nValidación cruzada ({target_name}) — R² por fold:", scores)
    print("Media R²:", scores.mean())
    print("Desviación estándar R²:", scores.std())

kfold_rf("ge_2023")
kfold_rf("cc_2023")

# -----------------------------
# 6. Optimización de hiperparámetros con GridSearchCV
# -----------------------------
def gridsearch_rf(target_name):
    """
    Realiza GridSearchCV para encontrar los mejores hiperparámetros
    de un Random Forest para un target dado.
    """
    X = df[features]
    y = df[target_name]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [200, 500, 800],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_scaled, y)

    print(f"\nMejores parámetros para {target_name}:")
    print(grid.best_params_)
    print("Mejor R² medio (CV):", grid.best_score_)

    return grid.best_estimator_

best_rf_ge = gridsearch_rf("ge_2023")
best_rf_cc = gridsearch_rf("cc_2023")

# -----------------------------
# 7. Tablas de resumen de modelos (GE y CC)
# -----------------------------
resumen = pd.DataFrame([
    ["Modelo 1", "GE", "Regresión Lineal",
     res_ge["lr_metrics"]["MAE"], res_ge["lr_metrics"]["RMSE"], res_ge["lr_metrics"]["R2"]],
    ["Modelo 1", "GE", "Random Forest",
     res_ge["rf_metrics"]["MAE"], res_ge["rf_metrics"]["RMSE"], res_ge["rf_metrics"]["R2"]],
    ["Modelo 2", "CC", "Regresión Lineal",
     res_cc["lr_metrics"]["MAE"], res_cc["lr_metrics"]["RMSE"], res_cc["lr_metrics"]["R2"]],
    ["Modelo 2", "CC", "Random Forest",
     res_cc["rf_metrics"]["MAE"], res_cc["rf_metrics"]["RMSE"], res_cc["rf_metrics"]["R2"]],
],
    columns=["Modelo", "Target", "Algoritmo", "MAE", "RMSE", "R2"]
)

print("\nResumen de desempeño de modelos:")
print(resumen)

# -----------------------------
# 8. Tabla comparativa de importancia de variables (GE vs CC)
# -----------------------------
tabla_imp = imp_ge.merge(
    imp_cc,
    on="variable",
    suffixes=("_GE", "_CC")
)

print("\nImportancia comparada de variables (GE vs CC):")
print(tabla_imp)

# -----------------------------
# 9. Gráficos de predicción vs valor real (GE y CC)
# -----------------------------
def plot_pred_vs_real(model, target_name, title):
    """
    Dibuja un scatterplot de valor real vs predicción
    para un modelo dado y un target concreto.
    """
    X = df[features]
    y = df[target_name]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    y_pred = model.predict(X_scaled)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_pred_vs_real(best_rf_ge, "ge_2023", "Predicción vs valor real — GE (Random Forest optimizado)")
plot_pred_vs_real(best_rf_cc, "cc_2023", "Predicción vs valor real — CC (Random Forest optimizado)")