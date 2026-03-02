from pathlib import Path
import pandas as pd
import numpy as np
import re
import glob

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "test" / "raw"
OUT_DIR = BASE_DIR / "test" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUT = OUT_DIR / "merged_clean.csv"
TRAIN_OUT  = OUT_DIR / "train_clean.csv"
TEST_OUT   = OUT_DIR / "test_clean.csv"

# =========================
# Proper column names
# =========================
COLUMNS = [
    "VIN",     # Column_A
    "VB1",     # Column_B
    "C0",      # Column_C
    "R0",      # Column_D
    "L1",      # Column_E
    "W1",      # Column_F
    "L2",      # Column_G
    "W2",      # Column_H
    "L3",      # Column_I
    "W3",      # Column_J
    "L4",      # Column_K
    "W4",      # Column_L
    "L5",      # Column_M
    "W5",      # Column_N
    "Power",   # Column_O
    "Gain",    # Column_P
    "BW_3dB",  # Column_Q
    "UGB",     # Column_R
    "PM",      # Column_S
    "GM",      # Column_T
]

POSITIVE_COLS = [
    "C0","R0",
    "L1","W1","L2","W2",
    "L3","W3","L4","W4",
    "L5","W5"
]

# =========================
# Numeric Cleaning
# =========================
def to_numeric_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.replace(",", "", regex=False)

    unit_map = {
        "n": 1e-9,
        "u": 1e-6,
        "µ": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
    }

    def convert(val):
        if val is None:
            return np.nan
        v = str(val).strip()
        if v == "" or v.lower() == "nan":
            return np.nan

        v = re.sub(r"(V|A|Hz|ohm|Ω|deg)$", "", v)

        match = re.match(r"^([+-]?\d*\.?\d+(?:e[+-]?\d+)?)([a-zA-Zµ]+)?$", v)
        if not match:
            return np.nan

        num = float(match.group(1))
        suf = match.group(2)

        if not suf:
            return num

        suf = suf.lower()
        if suf in unit_map:
            return num * unit_map[suf]

        return np.nan

    return x.map(convert)

# =========================
# Main
# =========================
def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print("Found CSV files:", len(files))

    dfs = []

    for fp in files:
        df = pd.read_csv(fp, header=None)

        # Remove trailing empty column if exists
        while df.shape[1] > 20 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        if df.shape[1] < 20:
            raise ValueError(f"{fp} has only {df.shape[1]} columns, expected at least 20")

        df = df.iloc[:, :20]
        df.columns = COLUMNS

        # Numeric conversion
        for col in COLUMNS:
            df[col] = to_numeric_series(df[col])

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print("Rows before cleaning:", len(merged))

    # Drop rows with missing values
    merged = merged.dropna(subset=COLUMNS)

    # Positive filter
    for col in POSITIVE_COLS:
        merged = merged[merged[col] > 0]

    merged = merged.drop_duplicates()

    print("Rows after cleaning:", len(merged))

    merged.to_csv(MERGED_OUT, index=False)
    print("Saved merged:", MERGED_OUT)

    # 80/20 split
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(0.8 * len(merged))
    train_df = merged.iloc[:split_index]
    test_df  = merged.iloc[split_index:]

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("Saved train:", TRAIN_OUT, "rows:", len(train_df))
    print("Saved test:", TEST_OUT, "rows:", len(test_df))

if __name__ == "__main__":
    main()