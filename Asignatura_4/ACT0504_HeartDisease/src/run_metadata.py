from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from src.config import RAW_DATA_FILE, REPORTS_DIR, PROCESSED_DATA_DIR

def write_run_metadata(df_raw: pd.DataFrame, df_clean: pd.DataFrame, *, outputs: Dict[str, Any], alpha: float) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "run_metadata.json"

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw": {
            "rows": int(df_raw.shape[0]),
            "cols": int(df_raw.shape[1]),
            "na_total": int(df_raw.isna().sum().sum()),
            "file": str(RAW_DATA_FILE),
        },
        "clean": {
            "rows": int(df_clean.shape[0]),
            "cols": int(df_clean.shape[1]),
            "na_total": int(df_clean.isna().sum().sum()),
            "processed_dir": str(PROCESSED_DATA_DIR),
        },
        "outputs": outputs,
        "alpha": float(alpha),
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
