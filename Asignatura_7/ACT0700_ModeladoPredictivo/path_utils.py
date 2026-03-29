from pathlib import Path
import gzip
import shutil
import zipfile


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def raw_file(filename: str) -> str:
    """
    Devuelve la ruta de un archivo de entrada en data/raw.
    Si no existe sin comprimir, intenta reconstruirlo desde .gz o .zip.
    """
    candidate = RAW_DIR / filename
    if candidate.exists():
        return str(candidate)

    gz_candidate = RAW_DIR / f"{filename}.gz"
    if gz_candidate.exists():
        with gzip.open(gz_candidate, "rb") as compressed_file, candidate.open("wb") as out_file:
            shutil.copyfileobj(compressed_file, out_file)
        return str(candidate)

    zip_candidate = RAW_DIR / f"{filename}.zip"
    if zip_candidate.exists():
        with zipfile.ZipFile(zip_candidate, "r") as zf:
            if filename in zf.namelist():
                zf.extract(filename, RAW_DIR)
            else:
                # Compatibilidad: si el zip tiene un unico archivo, lo usamos.
                only_files = [name for name in zf.namelist() if not name.endswith("/")]
                if len(only_files) == 1:
                    extracted_name = only_files[0]
                    zf.extract(extracted_name, RAW_DIR)
                    (RAW_DIR / extracted_name).replace(candidate)
                else:
                    raise FileNotFoundError(
                        f"El zip '{zip_candidate.name}' no contiene '{filename}' "
                        "ni un unico archivo para usar como reemplazo."
                    )
        return str(candidate)

    raise FileNotFoundError(
        f"No se encontro '{filename}' en {RAW_DIR}. "
        "Coloca los archivos fuente en data/raw."
    )


def processed_file(filename: str) -> str:
    """
    Devuelve la ruta de un archivo de salida procesado.
    """
    return str(PROCESSED_DIR / filename)
