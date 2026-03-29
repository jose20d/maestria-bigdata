from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def raw_file(filename: str) -> str:
    """
    Devuelve la ruta de un archivo de entrada en data/raw.
    """
    candidate = RAW_DIR / filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"No se encontro '{filename}' en {RAW_DIR}. "
            "Coloca los archivos fuente en data/raw."
        )
    return str(candidate)


def processed_file(filename: str) -> str:
    """
    Devuelve la ruta de un archivo de salida procesado.
    """
    return str(PROCESSED_DIR / filename)
