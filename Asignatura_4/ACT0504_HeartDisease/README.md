# ACT0504 – Análisis Estadístico del Dataset Heart Disease (UCI)  
# ACT0504 – Statistical Analysis of the Heart Disease Dataset (UCI)

**Autor principal / Main author:** Jose Daniel Alfaro  
**Colaboradores / Collaborators:** *(to be added)*  
**Programa / Program:** Máster en Big Data & Business Intelligence – NEXT Educación  
**Asignatura / Course:** Fundamentos de Análisis y Preprocesamiento de Datos  
**Año académico / Academic year:** 2025  

---

## 1. Aviso importante sobre el uso de este repositorio  
## 1. Important Notice Regarding the Use of this Repository

Este repositorio es público únicamente para permitir su revisión académica por parte del profesorado y de la institución.  
No constituye un proyecto de código abierto ni un recurso reutilizable para terceros.

This repository is public solely to allow academic review by the instructor and the institution.  
It is not an open-source project, nor a reusable resource for third parties.

**Queda estrictamente prohibido / Strictly prohibited:**

- Copiar, reproducir o reutilizar total o parcialmente el código, análisis o contenido aquí publicado para trabajos académicos propios o ajenos.  
- Presentar este material en cualquier institución educativa como si fuera de autoría propia.  
- Redistribuir, clonar o derivar proyectos basados en este repositorio sin autorización expresa.

Copying, reproducing, or reusing any portion of the code, analyses, or content for personal or third-party academic work is strictly forbidden.  
Submitting this material to any educational institution as original work is prohibited.  
Redistributing, cloning, or deriving projects from this repository without explicit permission is not allowed.

**Se permite / Allowed:**

- La revisión académica por parte del profesor y la universidad.  
- La referencia conceptual sin copiar contenido literal.

- Academic review by the instructor and institution.  
- Conceptual referencing without literal reproduction of code or text.

---

## 2. Descripción general del proyecto  
## 2. Project Overview

Este proyecto implementa un análisis estadístico completo del dataset *Heart Disease (Cleveland, UCI)*.  
Incluye procesos de limpieza, preprocesamiento, estadística descriptiva, visualización y pruebas de hipótesis, siguiendo los requerimientos de la asignatura.

This project implements a complete statistical analysis of the *Heart Disease (Cleveland, UCI)* dataset.  
It includes data cleaning, preprocessing, descriptive statistics, visualization, and hypothesis testing according to course requirements.

**Componentes principales / Main components:**

- Descarga y carga automatizada del dataset  
- Limpieza de datos (imputación, tratamiento de tipos, duplicados, outliers)  
- Estadística descriptiva y tablas de frecuencias  
- Visualizaciones (histogramas, boxplots, correlaciones)  
- Pruebas de hipótesis (Chi-cuadrado, t-test)  
- Pipeline modular implementado en Python

- Automated dataset acquisition and loading  
- Data cleaning (imputation, type handling, duplicates, outliers)  
- Descriptive statistics and frequency tables  
- Visualizations (histograms, boxplots, correlation heatmaps)  
- Hypothesis testing (Chi-square, t-test)  
- Modular processing pipeline implemented in Python

---

## 3. Estructura del proyecto  
## 3. Project Structure

ACT0504_HeartDisease/
│
├── data/ # Dataset (descargado automáticamente / automatically downloaded)
├── output/plots/ # Imágenes generadas / generated plots
├── src/
│ ├── config.py # Configuración de rutas / path configuration
│ ├── data_loading.py # Descarga y carga / dataset loading
│ ├── cleaning.py # Procesos de limpieza / cleaning pipeline
│ ├── eda.py # Estadística descriptiva / descriptive statistics
│ ├── plotting.py # Visualización / visualization
│ ├── stats_tests.py # Pruebas de hipótesis / hypothesis tests
│ └── utils.py # Utilidades / utilities
│
├── main.py # Ejecución principal / main execution script
├── requirements.txt # Dependencias / dependencies
└── README.md # Este documento / this document