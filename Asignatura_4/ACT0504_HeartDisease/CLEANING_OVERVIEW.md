# CLEANING_OVERVIEW – Reglas de limpieza del dataset Heart Disease (Cleveland, UCI)  
# CLEANING_OVERVIEW – Data cleaning rules for the Heart Disease dataset (Cleveland, UCI)

---

## 1. Objetivo del documento  
## 1. Purpose of this document

Este documento describe, en lenguaje no técnico, **todas las decisiones de limpieza y preparación de datos** aplicadas al dataset *Heart Disease (Cleveland, UCI)* en este proyecto.  

Las reglas se implementan en código, pero aquí se explican de forma que cualquier miembro del equipo (aunque no programe) pueda entender:

- **Qué se hace** con los datos  
- **Por qué se hace** (criterio médico / estadístico / enunciado del trabajo)  
- **Dónde se aplica** dentro del pipeline  

---

This document explains, in non‑technical language, **all data cleaning and preparation decisions** applied to the *Heart Disease (Cleveland, UCI)* dataset in this project.  

The rules are implemented in code, but here they are described so that any team member (even without coding experience) can understand:

- **What** is done to the data  
- **Why** it is done (medical / statistical / assignment reasons)  
- **Where** it happens inside the pipeline  

---

## 2. Relación con el código y con cleaning_config.json  
## 2. Relation to the code and to cleaning_config.json

En el código, la limpieza se controla mediante un archivo de configuración llamado `cleaning_config.json`.  
Ese archivo contiene, en formato de texto estructurado, las reglas descritas en este documento:

- Rangos válidos para cada variable  
- Valores permitidos para variables categóricas  
- Casos especiales que deben eliminarse  
- Opciones adicionales como normalización o creación de nuevas variables  

El archivo JSON **no es necesario entenderlo para seguir este documento**: aquí se resumen todas las decisiones de forma descriptiva.

---

In the code, cleaning behaviour is controlled through a configuration file named `cleaning_config.json`.  
That file contains, in structured text form, the same rules described here:

- Valid ranges for each variable  
- Allowed values for categorical variables  
- Special cases that must be removed  
- Extra options such as normalisation or new derived variables  

Understanding JSON is **not required to read this document**: all decisions are summarised descriptively here.

---

## 3. Vista general del pipeline de limpieza  
## 3. High‑level view of the cleaning pipeline

El pipeline de limpieza aplicado en `cleaning.py` sigue exactamente los pasos que aparecen en la teoría de la asignatura y en el enunciado del trabajo:

1. **Revisión de la calidad del dataset**  
2. **Revisión y conversión de tipos de datos**  
3. **Tratamiento de valores atípicos y errores**  
4. **Normalización / estandarización (opcional)**  
5. **Feature engineering (nuevas variables categóricas)**  

---

The cleaning pipeline implemented in `cleaning.py` follows the same steps as in the course theory and in the assignment statement:

1. **Dataset quality review**  
2. **Data type review and conversion**  
3. **Treatment of outliers and errors**  
4. **Normalisation / standardisation (optional)**  
5. **Feature engineering (new categorical variables)**  

Cada sección siguiente explica estos pasos en detalle.  
Each of the following sections details these steps.

---

## 4. Revisión de la calidad del dataset  
## 4. Dataset quality review

### 4.1. Detección de cabecera duplicada  
En algunos archivos CSV, la primera fila puede repetir los nombres de las columnas.  
El pipeline comprueba si la primera fila es idéntica a los nombres de columna; si lo es, **se elimina esa fila** para evitar contarla como una observación adicional.

---

In some CSV files, the first row may accidentally duplicate the column headers.  
The pipeline checks whether the first row matches the column names; if it does, **that row is removed** so it is not treated as an extra record.

---

### 4.2. Eliminación de duplicados  
Después de cargar el dataset, se eliminan **filas completamente duplicadas**.  
Esto evita que una observación repetida influya dos veces en medias, proporciones o pruebas estadísticas.

---

After loading the dataset, **fully duplicated rows** are removed.  
This avoids counting the same observation twice in means, proportions or statistical tests.

---

### 4.3. Valores fuera de rango y dominios válidos  
Para cada variable se definen **rangos fisiológicos o dominios lógicos**.  
Cuando un valor cae fuera de esos rangos, se considera que el dato es poco fiable.

En general, la estrategia aplicada es:

- Si el valor está fuera del rango permitido o no pertenece a la lista de valores válidos →  
  - En la mayoría de variables se **elimina la fila completa** (estrategia “drop”).  
  - En la variable `ca` (número de vasos) se marca como **valor ausente (NA)**, ya que el propio dataset original permite NA.

---

For each variable, **physiological ranges or logical domains** are defined.  
When a value falls outside these limits, the measurement is considered unreliable.

In general, the strategy is:

- If a value is outside the allowed range or not in the valid list →  
  - For most variables the **entire row is removed** (“drop” strategy).  
  - For `ca` (number of vessels) the value is set to **missing (NA)**, because the original dataset allows NA for this variable.

---

### 4.4. Resumen de rangos y valores permitidos  
### 4.4. Summary of ranges and allowed values

A continuación se muestra un resumen de las reglas principales por variable.  
The table below summarises the main rules per variable.

- **`age`**  
  - Rango aceptado: 0–120 años  
  - Fuera de este rango: se elimina la fila  
  - Motivación: límites fisiológicamente razonables  

- **`sex`**  
  - Valores válidos: 0 (mujer), 1 (hombre)  
  - Cualquier otro valor: se elimina la fila  

- **`cp`** (tipo de dolor de pecho)  
  - Valores válidos: 0, 1, 2, 3  
  - Cualquier otro valor: se elimina la fila  

- **`trestbps`** (presión arterial en reposo)  
  - Rango aceptado: 70–180 mmHg  
  - Fuera de este rango: se elimina la fila  
  - Justificación: valores extremos se consideran errores de registro para este estudio descriptivo.  

- **`chol`** (colesterol sérico)  
  - Rango aceptado general: 100–600 mg/dl  
  - Fuera de rango: se elimina la fila, salvo reglas especiales descritas más abajo.  

- **`fbs`** (glucemia en ayunas > 120 mg/dl)  
  - Valores válidos: 0 (no), 1 (sí)  
  - Cualquier otro valor: se elimina la fila  

- **`restecg`** (electrocardiograma en reposo)  
  - Valores válidos: 0, 1, 2  
  - Cualquier otro valor: se elimina la fila  

- **`thalach`** (frecuencia cardiaca máxima alcanzada)  
  - Rango aceptado: 60–250 latidos por minuto  
  - Fuera de rango: se elimina la fila  

- **`exang`** (angina inducida por ejercicio)  
  - Valores válidos: 0 (no), 1 (sí)  
  - Cualquier otro valor: se elimina la fila  

- **`oldpeak`** (depresión del ST)  
  - Rango aceptado: 0.0–6.5  
  - Fuera de rango: se elimina la fila  

- **`slope`** (pendiente del segmento ST)  
  - Valores válidos: 0, 1, 2  
  - Cualquier otro valor: se elimina la fila  

- **`ca`** (número de vasos principales coloreados por fluoroscopia)  
  - Rango aceptado: 0–4  
  - Fuera de este rango: el valor se marca como NA, no se elimina la fila  
  - Motivo: en el dataset original hay NA explícitos; se prefiere no perder toda la observación.  

- **`thal`** (resultado del test de thalium)  
  - Valores válidos: 0, 1, 2, 3  
  - Cualquier otro valor: se elimina la fila  

- **`target` / `condition`**  
  - Valores válidos: 0 (saludable), 1 (enfermedad cardíaca)  
  - El código renombra internamente `target` a `condition` para mejorar la interpretación.  
  - Cualquier otro valor: se elimina la fila  

---

The Spanish bullets above apply in the same way in English; only the wording changes:

- Values outside the specified ranges are considered unreliable and the row is dropped.  
- For `ca`, out‑of‑range values are set to NA instead of dropping the entire record.  
- For all coded categorical variables, any code outside the valid set leads to row removal.  

---

## 5. Revisión y conversión de tipos de datos  
## 5. Data type review and conversion

En el código, todas las columnas numéricas se convierten a tipos **numéricos anulables**:

- Enteros: `Int64` (permite NA)  
- Reales: `Float64` (permite NA)  

Cualquier valor que no se pueda interpretar como número (por ejemplo, texto en una columna numérica) se convierte automáticamente en **NA**, lo que hace visible el problema durante el análisis.

---

In the code, all numeric columns are converted to **nullable numeric types**:

- Integers: `Int64` (allows NA)  
- Reals: `Float64` (allows NA)  

Any value that cannot be interpreted as a number (for example, text in a numeric column) is automatically converted to **NA**, making the issue explicit during analysis.

---

## 6. Tratamiento de valores atípicos y reglas especiales  
## 6. Outlier treatment and special rules

Además de los rangos generales, se aplican algunas reglas específicas acordadas por el equipo:

1. **Caso de colesterol 564 con paciente sano**  
   - Condición: `chol = 564` y `condition = 0`  
   - Acción: se elimina la fila  
   - Motivo: el colesterol extremadamente alto en un paciente clasificado como sano se considera inconsistente para este estudio.  

2. **Valores extremos de oldpeak en pacientes sanos**  
   - Condición: `oldpeak = 3.5` o `4.2` y `condition = 0`  
   - Acción: se elimina la fila  
   - Motivo: valores muy altos de depresión del ST en sujetos sin enfermedad cardiaca son sospechosos de error de medición o codificación.  

3. **Combinación clínica incompatible**  
   - Condición aproximada: `thal = 2`, `restecg = 2` y `cp` distinto de 3 en pacientes sanos  
   - Acción: se elimina la fila  
   - Motivo: combinación considerada clínicamente incoherente según el análisis del equipo, por lo que se prefiere no usar estas observaciones en el análisis estadístico.  

---

In addition to general ranges, some **specific rules** agreed by the team are applied:

1. **Cholesterol = 564 with a healthy patient**  
   - Condition: `chol = 564` and `condition = 0`  
   - Action: drop the row  
   - Reason: such an extreme cholesterol level is inconsistent with a “healthy” label in this context.  

2. **Extreme oldpeak values in healthy patients**  
   - Condition: `oldpeak = 3.5` or `4.2` with `condition = 0`  
   - Action: drop the row  
   - Reason: unusually high ST depression values in subjects labelled as healthy are likely measurement or coding errors.  

3. **Clinically incompatible combination**  
   - Approximate condition: `thal = 2`, `restecg = 2` and `cp` not equal to 3 in healthy patients  
   - Action: drop the row  
   - Reason: this combination is considered clinically inconsistent and is therefore removed from the statistical analysis.  

---

## 7. Normalización / estandarización  
## 7. Normalisation / standardisation

El pipeline incluye un módulo de normalización que **por defecto está desactivado**, tal y como se acordó en el equipo:

- El archivo de configuración permite activar una normalización de tipo **z‑score** (`(valor − media) / desviación estándar`) para algunas variables numéricas (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).  
- En la configuración actual, la opción `"apply"` está en `false`, de modo que **no se crean columnas normalizadas**.  

La decisión se documenta así:

- Para el análisis descriptivo y las pruebas estadísticas planteadas, es suficiente trabajar con las unidades originales.  
- Mantener las variables en sus unidades reales facilita la interpretación para un público no técnico.

---

The pipeline contains a normalisation module that is **disabled by default**, as agreed within the team:

- The configuration file allows enabling a **z‑score** normalisation (`(value − mean) / standard deviation`) for some numeric variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).  
- In the current configuration, the `"apply"` option is set to `false`, so **no normalised columns are created**.  

The rationale is:

- For the descriptive analysis and statistical tests required, working in original units is sufficient.  
- Keeping the variables in real‑world units makes interpretation easier for a non‑technical audience.

---

## 8. Feature engineering (variables derivadas)  
## 8. Feature engineering (derived variables)

Para facilitar la interpretación de los resultados, se crean varias variables categóricas a partir de las variables numéricas originales:

1. **Sexo categórico – `sex_cat`**  
   - 0 → "Femenino"  
   - 1 → "Masculino"  

2. **Dolor de pecho categórico – `cp_cat`**  
   - 0 → "Angina típica"  
   - 1 → "Angina atípica"  
   - 2 → "Dolor no anginal"  
   - 3 → "Asintomático"  

3. **Condición del paciente – `condition_cat`**  
   - 0 → "Saludable"  
   - 1 → "Enfermedad cardiaca"  

4. **Grupos de edad – `age_group`** (definido en la configuración)  
   - [0, 40)  → "joven"  
   - [40, 55) → "mediana_edad"  
   - [55, 70) → "mayor"  
   - [70, 120] → "anciano"  

Estas variables no sustituyen a las originales, sino que **las complementan** para generar tablas de frecuencia, gráficos y análisis más fáciles de interpretar.

---

To make results easier to interpret, several **derived categorical variables** are created:

1. **Categorical sex – `sex_cat`**  
   - 0 → "Femenino" (female)  
   - 1 → "Masculino" (male)  

2. **Categorical chest pain – `cp_cat`**  
   - 0 → "Angina típica" (typical angina)  
   - 1 → "Angina atípica" (atypical angina)  
   - 2 → "Dolor no anginal" (non‑anginal pain)  
   - 3 → "Asintomático" (asymptomatic)  

3. **Patient condition – `condition_cat`**  
   - 0 → "Saludable" (healthy)  
   - 1 → "Enfermedad cardiaca" (heart disease)  

4. **Age groups – `age_group`** (defined in the configuration)  
   - [0, 40)  → "joven" / young  
   - [40, 55) → "mediana_edad" / middle‑aged  
   - [55, 70) → "mayor" / older  
   - [70, 120] → "anciano" / elderly  

These variables do not replace the original ones; they **complement** them, making it easier to build frequency tables, plots and more interpretable analyses.

---

## 9. Cómo utilizar este documento en el informe del trabajo  
## 9. How to use this document in the course report

- Cada sección de este documento se corresponde con un subapartado del punto  
  **“Limpieza y preparación de los datos”** del trabajo.  
- El informe escrito puede referirse a estas decisiones con frases del tipo:  

  > “Se definieron rangos plausibles para las variables clínicas (edad, presión arterial, colesterol, etc.) y se eliminaron las observaciones con valores fuera de dichos rangos o con combinaciones clínicamente incoherentes, según se detalla en el documento CLEANING_OVERVIEW.”  

- El objetivo es que cualquier revisor (profesor o compañero) pueda entender **sin mirar el código** qué se ha hecho exactamente sobre el dataset antes de aplicar estadísticos y gráficos.

---

- Each section of this document matches a sub‑section of  
  **“Data cleaning and preparation”** in the written report.  
- The report can refer to these decisions with sentences such as:  

  > “Plausible clinical ranges were defined for variables such as age, blood pressure and cholesterol. Observations outside those ranges or with clinically inconsistent combinations were removed, as described in the CLEANING_OVERVIEW document.”  

- The goal is that any reviewer (lecturer or teammate) can understand **without reading the code** exactly what has been done to the dataset before applying statistics and visualisations.
