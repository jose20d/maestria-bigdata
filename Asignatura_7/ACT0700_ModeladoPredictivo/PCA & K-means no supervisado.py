# ============================================================
# ANÁLISIS NO SUPERVISADO — PCA + K-MEANS
# ============================================================
# Objetivo:
# - Usar variables estructurales (económicas, sociales, institucionales)
#   para descubrir grupos de países con perfiles similares.
# - Reducir dimensionalidad con PCA para:
#     * Entender ejes latentes (desarrollo, fragilidad, etc.)
#     * Visualizar países en 2D
# - Aplicar K-Means sobre los datos estandarizados (o sobre PCA)
#   para obtener clusters interpretables.
# ============================================================

# -----------------------------
# 0. Importación de librerías
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
from path_utils import processed_file

sns.set(style="whitegrid")

# -----------------------------
# 1. Carga de datos
# -----------------------------
# Usamos el mismo dataset maestro que en los modelos supervisados
df = pd.read_csv(
    processed_file("dataset_maestro_2023.csv")
)

# -----------------------------
# 2. Selección de variables para el análisis no supervisado
# -----------------------------
# Usamos las mismas variables estructurales que en los modelos GE/CC.
features_unsup = [
    "gdp_per_capita",      # Desarrollo económico
    "education_secondary", # Educación secundaria
    "education_tertiary",  # Educación terciaria
    "internet_users",      # Usuarios de internet
    "electricity_access",  # Acceso a electricidad
    "health_expenditure",  # Gasto en salud
    "unemployment",        # Desempleo
    "life_expectancy",     # Esperanza de vida
    "cpi_2023",            # Percepción de corrupción
    "fsi_2023"             # Fragilidad estatal
]

X = df[features_unsup]

# (Opcional) Guardamos el nombre del país si existe una columna así
country_col = None
for col in df.columns:
    if col.lower() in ["country", "pais", "country_name"]:
        country_col = col
        break

if country_col is not None:
    countries = df[country_col]
else:
    countries = pd.Series(range(len(df)), name="id_pais")

# -----------------------------
# 3. Imputación de valores faltantes
# -----------------------------
# Paso clave: el modelo no supervisado no admite NaNs.
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# -----------------------------
# 4. Escalado de variables
# -----------------------------
# PCA y K-Means son sensibles a la escala de las variables.
# Estandarizamos para que todas tengan media 0 y desviación 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------------
# 5. PCA — Reducción de dimensionalidad
# -----------------------------
# Objetivo:
# - Reducir de 10 variables a 2 o 3 componentes principales.
# - Capturar la mayor parte de la varianza.
# - Facilitar la visualización y la interpretación.
pca = PCA(n_components=2)  # 2 componentes para visualización 2D
X_pca = pca.fit_transform(X_scaled)

# Creamos un DataFrame con los componentes principales
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

# Añadimos el país (si existe)
pca_df["pais"] = countries.values

# Mostramos la varianza explicada por cada componente
print("\nVarianza explicada por cada componente (PCA):")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.3f}")

print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum())

# -----------------------------
# 6. Interpretación de componentes (cargas)
# -----------------------------
# Las "cargas" indican cuánto contribuye cada variable a cada componente.
cargas = pd.DataFrame(
    pca.components_,
    columns=features_unsup,
    index=["PC1", "PC2"]
)

print("\nCargas de las variables en los componentes principales:")
print(cargas)

# -----------------------------
# 7. Elección del número de clusters (K) para K-Means
# -----------------------------
# Usamos dos criterios:
# - Método del codo (inercia)
# - Silhouette score
inertias = []
silhouettes = []
K_range = range(2, 8)  # probamos entre 2 y 7 clusters

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)  # clustering sobre datos escalados
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels_temp))

# Gráfico del codo
plt.figure(figsize=(6, 4))
plt.plot(list(K_range), inertias, marker="o")
plt.title("Método del codo (K-Means)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (suma de distancias al centro)")
plt.tight_layout()
plt.show()

# Gráfico de silhouette
plt.figure(figsize=(6, 4))
plt.plot(list(K_range), silhouettes, marker="o", color="green")
plt.title("Silhouette score por número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette score medio")
plt.tight_layout()
plt.show()

print("\nSilhouette scores por k:")
for k, s in zip(K_range, silhouettes):
    print(f"k={k}: silhouette={s:.3f}")

# -----------------------------
# 8. Entrenamiento final de K-Means
# -----------------------------
# Aquí eliges k basándote en los gráficos anteriores.
# Por ejemplo, supongamos que k=3 es un buen compromiso.
k_optimo = 3

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Añadimos los clusters al DataFrame PCA
pca_df["cluster"] = cluster_labels

# -----------------------------
# 9. Visualización PCA + Clusters
# -----------------------------
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set2",
    s=70
)
plt.title("Países en el espacio PCA con clusters (K-Means)")
plt.xlabel("Componente principal 1 (PC1)")
plt.ylabel("Componente principal 2 (PC2)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Resumen de clusters
# -----------------------------
# Calculamos la media de cada variable por cluster para interpretar perfiles.
X_cluster = pd.DataFrame(X_imputed, columns=features_unsup)
X_cluster["cluster"] = cluster_labels

cluster_profiles = X_cluster.groupby("cluster").mean()

print("\nPerfiles medios por cluster (en escala original):")
print(cluster_profiles)

# (Opcional) Añadir GE y CC para ver cómo se distribuyen por cluster
if "ge_2023" in df.columns and "cc_2023" in df.columns:
    temp = df[["ge_2023", "cc_2023"]].copy()
    temp["cluster"] = cluster_labels
    print("\nGE y CC medios por cluster:")
    print(temp.groupby("cluster").mean())

# -----------------------------
# 11. Exportar resultados (opcional)
# -----------------------------
# Guardar asignación de clusters por país
resultado_clusters = pca_df[["pais", "PC1", "PC2", "cluster"]]
resultado_clusters.to_csv(
    processed_file("resultado_clusters_pca_kmeans.csv"),
    index=False,
    encoding="utf-8"
)

print("\nArchivo 'resultado_clusters_pca_kmeans.csv' exportado con éxito.")