import random
import re
from pathlib import Path
import pandas as pd
import requests

# Configuration
URL = "https://django.cair.mun.ca/aide/v1/chat/completions"
INPUT_CSV = "test_clean.csv" #change according to where the input text file  
# MAX_TOKENS = 220

MODE_WEIGHTS = {
    "design": 1.0
}

# load the dataset from the CSV file
def load_dataset(csv_path=INPUT_CSV):
    return pd.read_csv(csv_path)


# def choose_mode(mode_weights=None):
#     if mode_weights is None:
#         mode_weights = MODE_WEIGHTS

#     modes = list(mode_weights.keys())
#     weights = list(mode_weights.values())
#     return random.choices(modes, weights=weights, k=1)[0]


def choose_random_row(df):
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df.sample(n=1).iloc[0]

# Build the system prompt for the design task
def build_design_system_prompt():
    return (
        "You are an analog IC sizing assistant. Given target performance metrics "
        "and bias voltages for a fixed two-stage amplifier topology in TSMC 65nm, "
        "output only the filled netlist parameter block in the exact required format. "
        "No explanations."
    )


def build_design_user_prompt(row):
    return (
        "Determine design parameters for the provided two-stage amplifier in TSMC 65nm. "
        f"Biasing voltages: VINP = VINN = VIN = {int(row['VIN'])} "
        f"VB1 = {int(row['VB1'])} "
        "Target performance metrics: "
        f"Power = {row['Power']} "
        f"Gain = {row['Gain']} "
        f"BW_3dB = {row['BW_3dB']} "
        f"UGB = {row['UGB']} "
        f"PM = {row['PM']} "
        f"GM = {row['GM']} "
        "Return only the netlist parameter block in the exact format below, filling all values: "
        "M1 pch net3 VINP net2 VDD L1=__ W1=__ "
        "M2 pch net1 VINN net2 VDD L1=__ W1=__ "
        "M5 pch net2 VB1 VDD VDD L2=__ W2=__ "
        "M3 nch net3 net1 GND GND L3=__ W3=__ "
        "M4 nch net1 net1 GND GND L3=__ W3=__ "
        "M6 nch VOUT net3 GND GND L4=__ W4=__ "
        "M7 pch VOUT VB1 VDD VDD L5=__ W5=__ "
        "C0 net3 net4 C0_value=__ "
        "R0 net4 VOUT R0_value=__ "
        "VINP=VINN=VIN=__ "
        "VB1=__"
    )


# def build_analysis_system_prompt():
#     return (
#         "You are an analog circuit analysis assistant. "
#         "Given the filled transistor sizes and component values for a fixed two-stage "
#         "amplifier topology in TSMC 65nm, estimate the resulting performance metrics. "
#         "Return only the requested metrics in the exact format. "
#         "Use plain numeric values only. Do not output long repeated decimals. "
#         "Do not include explanations."
#     )

# def build_analysis_user_prompt(row):
#     return (
#         "Estimate the performance metrics for the following two-stage amplifier in TSMC 65nm.\n\n"
#         "Return exactly these 6 lines and nothing else:\n"
#         "Power: <number>\n"
#         "Gain: <number>\n"
#         "BW_3dB: <number>\n"
#         "UGB: <number>\n"
#         "PM: <number>\n"
#         "GM: <number>\n\n"
#         "Amplifier sizing:\n"
#         f"M1 pch net3 VINP net2 VDD L1={row['L1']} W1={row['W1']}\n"
#         f"M2 pch net1 VINN net2 VDD L1={row['L1']} W1={row['W1']}\n"
#         f"M5 pch net2 VB1 VDD VDD L2={row['L2']} W2={row['W2']}\n"
#         f"M3 nch net3 net1 GND GND L3={row['L3']} W3={row['W3']}\n"
#         f"M4 nch net1 net1 GND GND L3={row['L3']} W3={row['W3']}\n"
#         f"M6 nch VOUT net3 GND GND L4={row['L4']} W4={row['W4']}\n"
#         f"M7 pch VOUT VB1 VDD VDD L5={row['L5']} W5={row['W5']}\n"
#         f"C0 net3 net4 C0_value={row['C0']}\n"
#         f"R0 net4 VOUT R0_value={row['R0']}\n"
#         f"VINP=VINN=VIN={row['VIN']}\n"
#         f"VB1={row['VB1']}\n"
#     )


def build_payload(row, mode=None):
    if mode is None:
        mode = "design"

    if mode == "design":
        system_prompt = build_design_system_prompt()
        user_prompt = build_design_user_prompt(row)
        MAX_TOKENS = 220
    # elif mode == "analysis":
    #     system_prompt = build_analysis_system_prompt()
    #     user_prompt = build_analysis_user_prompt(row)
    #     MAX_TOKENS = 80
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    payload = {
        "model": "aide",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
    }

    return mode, payload


def call_aide(payload, url=URL):
    try:
        response = requests.post(
            url,
            json=payload,
            verify=False,
            timeout=60,
        )

        print(f"HTTP status: {response.status_code}", flush=True)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}")
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Unexpected API response format: {exc}\nResponse text:\n{response.text}"
        )


def parse_design_output(text):
    patterns = {
        "L1": r"L1\s*=\s*([\d.]+)",
        "W1": r"W1\s*=\s*([\d.]+)",
        "L2": r"L2\s*=\s*([\d.]+)",
        "W2": r"W2\s*=\s*([\d.]+)",
        "L3": r"L3\s*=\s*([\d.]+)",
        "W3": r"W3\s*=\s*([\d.]+)",
        "L4": r"L4\s*=\s*([\d.]+)",
        "W4": r"W4\s*=\s*([\d.]+)",
        "L5": r"L5\s*=\s*([\d.]+)",
        "W5": r"W5\s*=\s*([\d.]+)",
        "C0_value": r"C0_value\s*=\s*([\d.]+)",
        "R0_value": r"R0_value\s*=\s*([\d.]+)",
        "VIN": r"VINP\s*=\s*VINN\s*=\s*VIN\s*=\s*([\d.]+)|VINP=VINN=VIN=([\d.]+)",
        "VB1": r"VB1\s*=\s*([\d.]+)",
    }

    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not find {key} in model output.\n\nModel output was:\n{text}")

        if key == "VIN":
            values[key] = float(match.group(1) or match.group(2))
        else:
            values[key] = float(match.group(1))

    return values


# def parse_analysis_output(text):
#     patterns = {
#         "Power": r"Power:\s*([^\n]+)",
#         "Gain": r"Gain:\s*([^\n]+)",
#         "BW_3dB": r"BW_3dB:\s*([^\n]+)",
#         "UGB": r"UGB:\s*([^\n]+)",
#         "PM": r"PM:\s*([^\n]+)",
#         "GM": r"GM:\s*([^\n]+)",
#     }

#     values = {}
#     for key, pattern in patterns.items():
#         match = re.search(pattern, text)
#         if not match:
#             raise ValueError(f"Could not find {key} in model output.\n\nModel output was:\n{text}")
#         values[key] = match.group(1).strip()

#     return values


def format_number(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value)


# Convert LLM output to the size format expected by the Cadence scripts
def convert_design_to_size_format(values):
    return (
        "num_vars: 14\n"
        "num_Vbias: 1\n"
        "num_Trans: 7\n"
        "num_Caps: 1\n"
        "num_Recs: 1\n"
        f"Size: "
        f"{format_number(values['VIN'])} {format_number(values['VB1'])} "
        f"{format_number(values['C0_value'])} {format_number(values['R0_value'])} "
        f"{format_number(values['L1'])} {format_number(values['W1'])} "
        f"{format_number(values['L2'])} {format_number(values['W2'])} "
        f"{format_number(values['L3'])} {format_number(values['W3'])} "
        f"{format_number(values['L4'])} {format_number(values['W4'])} "
        f"{format_number(values['L5'])} {format_number(values['W5'])} "
    )

# Convert LLM output to a Cadence netlist format for the two-stage amplifier topology [Not Necessary?]
def convert_to_cadence_netlist(values):
    return (
        f"M1 (net3 VINP net2 VDD) pch l={format_number(values['L1'])}n w={format_number(values['W1'])}u\n"
        f"M2 (net1 VINN net2 VDD) pch l={format_number(values['L1'])}n w={format_number(values['W1'])}u\n"
        f"M5 (net2 VB1 VDD VDD) pch l={format_number(values['L2'])}n w={format_number(values['W2'])}u\n"
        f"M3 (net3 net1 GND GND) nch l={format_number(values['L3'])}n w={format_number(values['W3'])}u\n"
        f"M4 (net1 net1 GND GND) nch l={format_number(values['L3'])}n w={format_number(values['W3'])}u\n"
        f"M6 (VOUT net3 GND GND) nch l={format_number(values['L4'])}n w={format_number(values['W4'])}u\n"
        f"M7 (VOUT VB1 VDD VDD) pch l={format_number(values['L5'])}n w={format_number(values['W5'])}u\n"
        f"C0 (net3 net4) capacitor c={format_number(values['C0_value'])}p\n"
        f"R0 (net4 VOUT) resistor r={format_number(values['R0_value'])}k\n"
    )

# run AIDE model with the chosen row and mode, and process the output accordingly
def run_aide(df, mode=None, url=URL):
    row = choose_random_row(df)
    chosen_mode, payload = build_payload(row, mode=mode)

    print(f"Chosen mode: {chosen_mode}", flush=True)

    model_output = call_aide(payload, url=url)

    result = {
        "mode": chosen_mode,
        "payload": payload,
        "model_output": model_output,
        "source_row": row.to_dict(),
    }

    if chosen_mode == "design":
        parsed = parse_design_output(model_output)
        result["parsed_values"] = parsed
        result["size_format_output"] = convert_design_to_size_format(parsed)
        result["cadence_netlist"] = convert_to_cadence_netlist(parsed)

    # elif chosen_mode == "analysis":
    #     parsed_metrics = parse_analysis_output(model_output)
    #     result["parsed_metrics"] = parsed_metrics

    return result


if __name__ == "__main__":
    try:
        print("Script started", flush=True)

        script_dir = Path(__file__).parent
        output_dir = script_dir / "cadence_outputs"
        output_dir.mkdir(exist_ok=True)

        df = load_dataset(script_dir / INPUT_CSV)
        result = run_aide(df)

        print("Got result", flush=True)
        print(f"Mode: {result['mode']}", flush=True)

        print("\nModel output:\n", flush=True)
        print(result["model_output"], flush=True)

        raw_output_file = output_dir / "raw_model_output.txt"
        raw_output_file.write_text(result["model_output"])
        print(f"\nSaved raw model output to: {raw_output_file.resolve()}", flush=True)

        if result["mode"] == "design":
            design_inputs = {
                "VIN": result["source_row"]["VIN"],
                "VB1": result["source_row"]["VB1"],
                "Power": result["source_row"]["Power"],
                "Gain": result["source_row"]["Gain"],
                "BW_3dB": result["source_row"]["BW_3dB"],
                "UGB": result["source_row"]["UGB"],
                "PM": result["source_row"]["PM"],
                "GM": result["source_row"]["GM"],
            }

            print("\nDesign inputs used:\n", flush=True)
            print(design_inputs, flush=True)

            print("\nSize format output:\n", flush=True)
            print(result["size_format_output"], flush=True)

            size_file = output_dir / "design_size_output.txt"
            size_file.write_text(result["size_format_output"])
            print(f"\nSaved size format output to: {size_file.resolve()}", flush=True)

            print("\nCadence netlist:\n", flush=True)
            print(result["cadence_netlist"], flush=True)

            netlist_file = output_dir / "sample_output.scs"
            netlist_file.write_text(result["cadence_netlist"])
            print(f"\nSaved Cadence netlist to: {netlist_file.resolve()}", flush=True)

        # elif result["mode"] == "analysis":
        #     print("\nInput row used:\n", flush=True)
        #     print(result["source_row"], flush=True)

        #     metrics_file = output_dir / "analysis_metrics.txt"
        #     metrics_file.write_text(result["model_output"])
        #     print(f"\nSaved analysis metrics to: {metrics_file.resolve()}", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)