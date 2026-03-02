from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "test" / "processed"
OUT_DIR = BASE_DIR  / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = PROCESSED_DIR / "train_clean.csv"
TEST_CSV  = PROCESSED_DIR / "test_clean.csv"

TRAIN_JSONL = OUT_DIR / "train.jsonl"
TEST_JSONL  = OUT_DIR / "test.jsonl"

SYSTEM_TEXT = (
    "You are an analog IC sizing assistant. "
    "Given target performance metrics and bias voltages for a fixed two-stage amplifier topology in TSMC 65nm, "
    "output only the filled netlist parameter block in the exact required format. "
    "No explanations."
)

USER_TEMPLATE = """Determine design parameters for the provided two-stage amplifier in TSMC 65nm.

Biasing voltages:
VINP = VINN = VIN = {VIN}
VB1 = {VB1}

Target performance metrics:
Power = {Power}
Gain = {Gain}
BW_3dB = {BW}
UGB = {UGB}
PM = {PM}
GM = {GM}

Return only the netlist parameter block in the exact format below, filling all values:

M1 pch net3 VINP net2 VDD L1=__ W1=__
M2 pch net1 VINN net2 VDD L1=__ W1=__
M5 pch net2 VB1 VDD VDD L2=__ W2=__
M3 nch net3 net1 GND GND L3=__ W3=__
M4 nch net1 net1 GND GND L3=__ W3=__
M6 nch VOUT net3 GND GND L4=__ W4=__
M7 pch VOUT VB1 VDD VDD L5=__ W5=__
C0 net3 net4 C0_value=__
R0 net4 VOUT R0_value=__

VINP=VINN=VIN=__
VB1=__
"""

ASSISTANT_TEMPLATE = """M1 pch net3 VINP net2 VDD L1={L1} W1={W1}
M2 pch net1 VINN net2 VDD L1={L1} W1={W1}
M5 pch net2 VB1 VDD VDD L2={L2} W2={W2}
M3 nch net3 net1 GND GND L3={L3} W3={W3}
M4 nch net1 net1 GND GND L3={L3} W3={W3}
M6 nch VOUT net3 GND GND L4={L4} W4={W4}
M7 pch VOUT VB1 VDD VDD L5={L5} W5={W5}
C0 net3 net4 C0_value={C0}
R0 net4 VOUT R0_value={R0}

VINP=VINN=VIN={VIN}
VB1={VB1}
"""

def fmt(x) -> str:
    return f"{float(x):.8g}"

def row_to_record(row) -> dict:
    user = USER_TEMPLATE.format(
        VIN=fmt(row["VIN"]),
        VB1=fmt(row["VB1"]),
        Power=fmt(row["Power"]),
        Gain=fmt(row["Gain"]),
        BW=fmt(row["BW_3dB"]),
        UGB=fmt(row["UGB"]),
        PM=fmt(row["PM"]),
        GM=fmt(row["GM"]),
    )
    assistant = ASSISTANT_TEMPLATE.format(
        L1=fmt(row["L1"]), W1=fmt(row["W1"]),
        L2=fmt(row["L2"]), W2=fmt(row["W2"]),
        L3=fmt(row["L3"]), W3=fmt(row["W3"]),
        L4=fmt(row["L4"]), W4=fmt(row["W4"]),
        L5=fmt(row["L5"]), W5=fmt(row["W5"]),
        C0=fmt(row["C0"]),
        R0=fmt(row["R0"]),
        VIN=fmt(row["VIN"]),
        VB1=fmt(row["VB1"]),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

def write_jsonl(df: pd.DataFrame, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_record(row)) + "\n")

def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    write_jsonl(train_df, TRAIN_JSONL)
    write_jsonl(test_df, TEST_JSONL)

    print("Wrote", TRAIN_JSONL, "examples:", len(train_df))
    print("Wrote", TEST_JSONL, "examples:", len(test_df))

if __name__ == "__main__":
    main()