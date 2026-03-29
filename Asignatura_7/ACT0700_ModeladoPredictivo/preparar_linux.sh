#!/usr/bin/env bash
set -euo pipefail

# Este script prepara data/raw con los archivos de entrada.

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="${BASE_DIR}/data/raw"
PROCESSED_DIR="${BASE_DIR}/data/processed"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

# Archivos de entrada esperados por los scripts.
# Si ya existen en data/raw no se tocan.
for f in \
  "wgidataset-2025.xlsx" \
  "CPI2023_Global_Results__Trends.xlsx" \
  "worldbank_gdp.xlsx" \
  "fsi.xlsx" \
  "wdi.csv"
do
  if [[ -f "${BASE_DIR}/${f}" && ! -f "${RAW_DIR}/${f}" ]]; then
    cp --update=none "${BASE_DIR}/${f}" "${RAW_DIR}/${f}"
  fi
done

echo "Entorno Linux preparado."
echo "Datos raw: ${RAW_DIR}"
echo "Datos processed: ${PROCESSED_DIR}"
