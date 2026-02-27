import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = "Capstone/data/raw"
OUT_CSV = "data/processed/merged_clean.csv"

REQUIRED_COLS = [
    "Column_A","Column_B","Column_C","Column_D",
    "Column_E","Column_F","Column_G","Column_H",
    "Column_I","Column_J","Column_K","Column_L",
    "Column_M","Column_N",
    "Column_O","Column_P","Column_Q","Column_R","Column_S","Column_T",
]

def to_numeric_series(s: pd.Series) -> pd.Series:
    # remove commas and stray spaces, convert to float
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def main():
    os.makedirs("data/processed", exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Trim whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nCheck your CSV headers. If they are different, tell me the exact headers and I will adjust."
        )

    # Convert all required columns to numeric
    for c in REQUIRED_COLS:
        df[c] = to_numeric_series(df[c])

    # Drop rows with any missing required value
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).copy()
    after = len(df)

    # Optional sanity filters: remove non positive sizes or component values
    # You can relax these later if needed
    positive_cols = [
        "Column_C","Column_D",
        "Column_E","Column_F","Column_G","Column_H",
        "Column_I","Column_J","Column_K","Column_L","Column_M","Column_N"
    ]
    for c in positive_cols:
        df = df[df[c] > 0]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved clean merged dataset to {OUT_CSV}")
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna and filters: {len(df)}")

if __name__ == "__main__":
    main()