# Proyecto de Modelado Predictivo (Asignatura 7)

## Objetivo

Este repositorio contiene scripts de preprocesamiento, analisis exploratorio, modelado supervisado y analisis no supervisado para el trabajo de Machine Learning del master.

El codigo se mantiene con su estructura actual, sin cambios de logica en funciones o modelos.

## Requisitos

- Linux o Windows
- Python 3.10 o superior
- `pip`

## Configuracion recomendada con entorno virtual (venv)

### Linux

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

Desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Instalacion sin venv (alternativa)

```bash
pip install -r requirements.txt
```

## Estructura actual (scripts principales)

- Preprocesamiento de fuentes:
  - `WGI rev00.py`
  - `WGI CC rev00.py`
  - `WGI GC rev00.py`
  - `CPI ver00.py`
  - `PIB test ver00.py`
  - `WDI rev00.py`
  - `fsi ver00.py`
- Union de datasets:
  - `Merged_all_DS_rev00.py`
- Analisis y modelado:
  - `EDA rev00.py`
  - `modelo 1.py`
  - `modelo 2.py`
  - `Script maestro de analisis post entreno.py`
  - `PCA & K-means no supervisado.py`

## Estructura de datos

- `data/raw`: datos de entrada (fuentes originales).
- `data/processed`: salidas intermedias y finales del pipeline.

Archivos esperados en `data/raw`:

- `wgidataset-2025.xlsx`
- `CPI2023_Global_Results__Trends.xlsx`
- `worldbank_gdp.xlsx`
- `fsi.xlsx`
- `wdi.csv`

## Preparacion de datos (`data/raw`)

Para copiar automaticamente los archivos desde la raiz del proyecto a `data/raw`:

```bash
bash preparar_linux.sh
```

En Windows, puede copiarse manualmente el mismo conjunto de archivos a `data/raw`.

## Ejecucion recomendada: Notebook (Linux/Windows)

Notebook principal:

- `Trabajo_Asignatura7.ipynb`

Iniciar Jupyter:

```bash
python -m notebook
```

o:

```bash
python -m jupyter lab
```

Pasos sugeridos dentro del notebook:

1. Abrir `Trabajo_Asignatura7.ipynb`.
2. Confiar el notebook (`Trust Notebook`) si aparece como no confiable.
3. Ejecutar en orden hasta la seccion de construccion del dataset maestro.
4. Verificar resultados en `data/processed`.
5. Descomentar y ejecutar bloques de EDA/modelos/no supervisado segun necesidad.

## Ejecucion del pipeline (Linux)

Ejecutar todo el pipeline:

```bash
bash ejecutar_pipeline_linux.sh
```

Nota: `ejecutar_pipeline_linux.sh` usa automaticamente `.venv/bin/python` si existe.

## Ejecucion paso a paso (Linux/Windows)

Si se prefiere ejecutar manualmente:

```bash
python "WGI rev00.py"
python "WGI CC rev00.py"
python "WGI GC rev00.py"
python "CPI ver00.py"
python "PIB test ver00.py"
python "WDI rev00.py"
python "fsi ver00.py"
python "Merged_all_DS_rev00.py"
python "EDA rev00.py"
python "modelo 1.py"
python "modelo 2.py"
python "Script maestro de analisis post entreno.py"
python "PCA & K-means no supervisado.py"
```

## Notas

- Las rutas estan centralizadas en `path_utils.py` y no dependen de `C:/...`.
- Todas las salidas del proyecto se escriben en `data/processed`.
- `requirements.txt` incluye dependencias de analisis y de notebook (`notebook`, `ipykernel`, `jupyterlab`).

