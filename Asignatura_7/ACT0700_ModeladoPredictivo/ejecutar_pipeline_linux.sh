#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${BASE_DIR}"

if [[ -x "${BASE_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${BASE_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
  echo "Aviso: no se encontro .venv/bin/python, se usara python3 del sistema."
fi

echo "1) Preparando entorno de compatibilidad..."
bash "./preparar_linux.sh"

echo "2) Ejecutando preprocesamientos..."
"${PYTHON_BIN}" "./WGI rev00.py"
"${PYTHON_BIN}" "./WGI CC rev00.py"
"${PYTHON_BIN}" "./WGI GC rev00.py"
"${PYTHON_BIN}" "./CPI ver00.py"
"${PYTHON_BIN}" "./PIB test ver00.py"
"${PYTHON_BIN}" "./WDI rev00.py"
"${PYTHON_BIN}" "./fsi ver00.py"

echo "3) Uniendo dataset maestro..."
"${PYTHON_BIN}" "./Merged_all_DS_rev00.py"

echo "4) Ejecutando analisis..."
"${PYTHON_BIN}" "./EDA rev00.py"
"${PYTHON_BIN}" "./modelo 1.py"
"${PYTHON_BIN}" "./modelo 2.py"
"${PYTHON_BIN}" "./Script maestro de analisis post entreno.py"
"${PYTHON_BIN}" "./PCA & K-means no supervisado.py"

echo "Pipeline completado."
