#!/usr/bin/env python3
"""
preprocess_to_llm_dataset.py

Turn CSV design table (VIN, VB1, C0_value, R0_value, L1/W1..L5/W5, metrics) into LLM-ready instruction/output pairs.

Outputs JSONL:
  out/train.jsonl
  out/val.jsonl
  out/test.jsonl

Run:
  python preprocess_to_llm_dataset.py --data_dir . --out_dir out
"""

from __future__ import annotations
import argparse # Let's you read command-line arguments
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import pandas as pd


# Schema to ensure these columns exist in each CSV
REQUIRED_COLS = [
    "VIN", "VB1", "C0_value", "R0_value",
    "L1", "W1", "L2", "W2", "L3", "W3", "L4", "W4", "L5", "W5",
    "Power", "Gain", "BW_3dB", "UGB", "PM", "GM"
]

# Netlist template with placeholders for design parameters. 
# We'll fill these in for each record.

NETLIST_TEMPLATE = """\
* Two-Stage Amplifier Design in TSMC 65nm CMOS Technology

* Biasing Voltages (mV)
* VINP=VINN=VIN={VIN_mV} mV, VB1={VB1_mV} mV

* MOSFETs (L in nm, W in um)
M1 pch net3 VINP net2 VDD L={L1_nm}n W={W1_um}u
M2 pch net1 VINN net2 VDD L={L1_nm}n W={W1_um}u
M5 pch net2 VB1 VDD VDD L={L2_nm}n W={W2_um}u
M3 nch net3 net1 GND GND L={L3_nm}n W={W3_um}u
M4 nch net1 net1 GND GND L={L3_nm}n W={W3_um}u
M6 nch VOUT net3 GND GND L={L4_nm}n W={W4_um}u
M7 pch VOUT VB1 VDD VDD L={L5_nm}n W={W5_um}u

* Miller Feedback Network
C0 net3 net4 {C0_pF}pF
R0 net4 VOUT {R0_KOhm}KOhm

.END
"""

# 
def load_concat_csvs(data_dir: str, pattern: str) -> pd.DataFrame:
    # Returns list of all sorted files matching the pattern
    files = sorted(glob.glob(os.path.join(data_dir, pattern))) 
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {os.path.join(data_dir, pattern)}")
    dfs = []
    for f in files:
        d = pd.read_csv(f) # Read CSV into DataFrame
        # Adds a column which identifies the CSV each row came from (useful for debugging)
        d["__source_file__"] = os.path.basename(f)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) # combines all DataFrames into one, resetting the index


# Converts specified columns to numeric types, setting invalid values to NaN
def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    # Convert numeric columns
    df = coerce_numeric(df, REQUIRED_COLS)

    # Removes rows with missing values in required columns, since we can't use them for training
    df = df.dropna(subset=REQUIRED_COLS)

    # Remove duplicates
    df = df.drop_duplicates()

    # remove negative/zero metrics that are clearly invalid
    df = df[df["Gain"] > 0]
    df = df[df["Power"] > 0]
    df = df[df["BW_3dB"] > 0]
    df = df[df["UGB"] > 0]

    return df

def build_instruction(row):
    return (
        "Two-Stage Amplifier Design in TSMC 65nm CMOS Technology.\n\n"
        "Design a two-stage amplifier (5 MOSFET first stage, 2 MOSFET second stage) "
        "with Miller compensation.\n\n"
        # Target specifications with 6 significant digits for readability.
        "Target specifications:\n"
        f"- Power: {row['Power']:.6g}\n" 
        f"- Gain: {row['Gain']:.6g}\n"
        f"- BW_3dB: {row['BW_3dB']:.6g}\n"
        f"- UGB: {row['UGB']:.6g}\n"
        f"- PM: {row['PM']:.6g}\n"
        f"- GM: {row['GM']:.6g}\n\n"
        "Return ONLY the SPICE netlist."
    )

# Build the SPICE netlist string for a given row of design parameters, using the NETLIST_TEMPLATE.
def build_netlist(row):
    def f(x):
        return f"{float(x):.6g}"

    return NETLIST_TEMPLATE.format(
        VIN_mV=f(row["VIN"]),
        VB1_mV=f(row["VB1"]),
        C0_pF=f(row["C0_value"]),
        R0_KOhm=f(row["R0_value"]),
        L1_nm=f(row["L1"]),
        W1_um=f(row["W1"]),
        L2_nm=f(row["L2"]),
        W2_um=f(row["W2"]),
        L3_nm=f(row["L3"]),
        W3_um=f(row["W3"]),
        L4_nm=f(row["L4"]),
        W4_um=f(row["W4"]),
        L5_nm=f(row["L5"]),
        W5_um=f(row["W5"]),
    ).strip()


def make_records(df: pd.DataFrame) -> List[Dict[str, str]]:
    records = []
    for _, row in df.iterrows():
        rec = {
            "instruction": build_instruction(row),
            "input": "",
            "output": build_netlist(row),
        }
        records.append(rec)
    return records


def split_records(records: List[Dict[str, str]], seed: int = 42,
                  train_frac: float = 0.8, val_frac: float = 0.1) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    recs = records[:]
    rng.shuffle(recs)

    n = len(recs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = recs[:n_train]
    val = recs[n_train:n_train + n_val]
    test = recs[n_train + n_val:]
    return train, val, test


def write_jsonl(path: str, records: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="Folder containing data_collected_*.csv")
    ap.add_argument("--pattern", type=str, default="data_collected_*.csv")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_split", action="store_true", help="Write one dataset.jsonl instead of train/val/test")
    args = ap.parse_args()

    df = load_concat_csvs(args.data_dir, args.pattern)
    df = clean_df(df)

    records = make_records(df)

    if args.no_split:
        write_jsonl(os.path.join(args.out_dir, "dataset.jsonl"), records)
        print(f"Wrote {len(records)} records to {os.path.join(args.out_dir, 'dataset.jsonl')}")
    else:
        train, val, test = split_records(records, seed=args.seed)
        write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
        write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
        write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)
        print(f"Wrote train={len(train)}, val={len(val)}, test={len(test)} to folder: {args.out_dir}")


if __name__ == "__main__":
    main()
