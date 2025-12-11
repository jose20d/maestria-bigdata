# main.py
from __future__ import annotations

from src.data_loading import get_raw_data
from src.cleaning import clean_data, save_clean_data


def main() -> None:
    df_raw = get_raw_data()
    print("Shape del dataset crudo:", df_raw.shape)

    df_clean = clean_data(df_raw)
    print("Shape del dataset limpio:", df_clean.shape)

    save_clean_data(df_clean)

    print("\nPrimeras filas del dataset limpio:")
    print(df_clean.head())

    print("\nInformación del dataset limpio:")
    print(df_clean.info())


if __name__ == "__main__":
    main()
