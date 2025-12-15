from __future__ import annotations
from typing import Iterable
import pandas as pd

def require_columns(df: pd.DataFrame, expected: Iterable[str], *, where: str) -> None:
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(
            f"[{where}] Faltan columnas esperadas: {missing}. "
            f"Columnas presentes: {sorted(df.columns)}"
        )
