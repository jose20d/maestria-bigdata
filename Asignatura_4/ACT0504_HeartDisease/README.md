# ACT0504 – Análisis Estadístico del Dataset Heart Disease (UCI)
# ACT0504 – Statistical Analysis of the Heart Disease Dataset (UCI)

Autores / Authors:
- Jose Daniel Alfaro
- Alejandro Abarca Méndez
- Dafnee Yesenia Alcantara Villalobos
- Juan Luis Chávez Mejía
- Marilyn Melissa Cerdas Brenes

Programa / Program: Máster en Big Data & Business Intelligence – NEXT Educación  
Asignatura / Course: Fundamentos de Análisis y Preprocesamiento de Datos  
Año académico / Academic Year: 2025

---

## 1. Aviso importante sobre el uso de este repositorio  
## 1. Important Notice Regarding the Use of This Repository

Este repositorio es público exclusivamente para fines de revisión académica.  
This repository is public solely for academic review purposes.

No constituye un proyecto de código abierto ni un recurso reutilizable por terceros.  
It is not an open‑source project nor a reusable resource.

### Prohibido / Strictly prohibited:
- Copiar o reutilizar código o análisis.
- Presentar este material como propio en cualquier institución.
- Clonar o derivar proyectos basados en este repositorio.

### Permitido / Allowed:
- Revisión académica.
- Referencias conceptuales sin copiar contenido literal.

---

## 2. Descripción general del proyecto  
## 2. Project Overview

Este proyecto implementa un pipeline modular, reproducible y automatizado para analizar el dataset *Heart Disease (Cleveland, UCI)*.  
This project implements a modular, reproducible, and automated pipeline to analyze the *Heart Disease (Cleveland, UCI)* dataset.

Incluye / It includes:
- Descarga automática desde Kaggle / Automatic download from Kaggle  
- Limpieza configurable mediante JSON / Configurable cleaning rules via JSON  
- Conversión de tipos / Type conversion  
- Feature engineering  
- Estadística descriptiva / Descriptive statistics  
- Visualizaciones / Visualizations  
- Pruebas estadísticas / Statistical tests  

---

## 3. Pipeline del proyecto  
## 3. Project Pipeline

1. Carga del dataset / Data loading  
2. Limpieza configurable / Configurable cleaning  
3. Feature engineering  
4. Exportación del dataset limpio / Export of cleaned dataset  
5. Estadística descriptiva y pruebas / EDA & statistical tests  

---

## 4. Estructura del proyecto  
## 4. Project Structure

ACT0504_HeartDisease/  
├── data/  
│   ├── raw/  
│   └── processed/  
├── output/  
│   ├── plots/  
│   └── reports/  
├── src/  
│   ├── config.py  
│   ├── data_loading.py  
│   ├── cleaning.py  
│   ├── eda.py  
│   ├── plotting.py  
│   ├── stats_tests.py  
│   └── utils.py  
├── cleaning_config.json  
├── main.py  
├── requirements.txt  
└── README.md  

---

## 5. Ejecución del proyecto  
## 5. Running the Project

```
python main.py
```
