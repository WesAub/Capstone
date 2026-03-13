import subprocess
import sys
import csv
from datetime import datetime
import os




def send_file_to_cadence(local_filepath, remote_directory="~/"):
    """
    Sends a file from your Windows H: drive to the Linux server.
    """
    server = "hnasiri_loc@cadlams2.engr.mun.ca"
    destination = f"{server}:{remote_directory}"

    command = [
        "scp", 
        #"-o", "HostKeyAlgorithms=+ssh-rsa", 
        local_filepath, 
        destination
    ]
    
    print(f"Sending {local_filepath} to Cadence server...")
    
    # This is exactly like typing 'scp local_file wnaubynn@server:~/' in DOS
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Success! File sent.")
    else:
        print("Error sending file:")
        print(result.stderr)

def get_file_from_cadence(remote_filepath, local_directory="."):
    """
    Downloads a file from the Linux server to your Windows computer.
    """
    server = "hnasiri_loc@cadlams2.engr.mun.ca"
    source = f"{server}:{remote_filepath}"

    command = [
        "scp", 
        #"-o", "HostKeyAlgorithms=+ssh-rsa", 
        #"-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
        source,
        local_directory
    ]
    
    print(f"Downloading {remote_filepath} from Cadence server...")
    
    # This is exactly like typing 'scp wnaubynn@server:~/remote_file ./' in DOS
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Success! File downloaded.")
    else:
        print("Error downloading file:")
        print(result.stderr)


def run_cadence_simulation():
    """
    Triggers the Cadence simulation over SSH and WAITS for it to finish.
    """
    server = "hnasiri_loc@cadlams2.engr.mun.ca"
    
    # We chain everything together with '&&'. 
    # If one step fails (like a typo in a folder name), it safely stops.
    cadence_command = "cd LLM && source .cshrc_65nm && cd Evaluation_Tool_Ocean && ./EVA"
    
    print(f"Connecting to {server}...")
    print("Starting Cadence simulation... (Python will wait here until it finishes)")

    command = [
        "ssh", 
        #"-o", "HostKeyAlgorithms=+ssh-rsa", 
        server,
        cadence_command
    ]
    
    # Send the command over SSH. Python will freeze here until ./EVA is done.
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Simulation complete! Cadence finished running ./EVA.")
        
        # Optional: If you want to see the standard output from EVA in your Windows terminal, 
        # uncomment the next line:
        print(result.stdout)
    else:
        print("Simulation failed! Here is the error log from the server:")
        print(result.stderr)
        
        # If the simulation fails, we exit Python so it doesn't try to download an empty result file
        sys.exit(1)


def log_simulation_data(input_filepath, result_filepath, start_time, end_time, log_file="H:\\My Documents\\Desktop\\Capstone\\simulation_logs.csv"):
    """
    Reads the input and result text files, extracts the values, 
    and appends a new record to the CSV log.
    """
    
    # 1. --- PARSE THE INPUTS (14 values) ---
    inputs = []
    with open(input_filepath, 'r') as f:
        for line in f:
            if line.startswith("Size:"):
                # .split() automatically breaks the string at every space
                parts = line.split() 
                # parts[0] is "Size:", so we grab everything from parts[1] onward
                inputs = parts[1:] 
                break # We found what we need, stop reading the file

    # 2. --- PARSE THE OUTPUTS (6 values) ---
    outputs = []
    with open(result_filepath, 'r') as f:
        for line in f:
            if ":" in line:
                # Split at the colon, grab the second part [1], and strip away extra spaces/newlines
                value = line.split(":")[1].strip()
                outputs.append(value)

    # 3. --- CALCULATE DURATION ---
    duration = end_time - start_time

    # 4. --- COMPILE THE RECORD ---
    # Combine the lists together to form one giant row for the CSV
    new_record = inputs + outputs + [
        start_time.strftime("%Y-%m-%d %H:%M:%S"), 
        end_time.strftime("%Y-%m-%d %H:%M:%S"), 
        str(duration)
    ]

    # 5. --- APPEND TO CSV SAFELY ---
    file_exists = os.path.exists(log_file)
    
    # Open the file in 'a' (append) mode so we never overwrite old records
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If this is the very first simulation, write the exact column headers first
        if not file_exists:
            headers = [f"Input_{i}" for i in range(1, 15)] + \
                      ["Power", "Gain", "BW_3dB", "UGB", "PM", "GM"] + \
                      ["Start_Time", "End_Time", "Duration"]
            writer.writerow(headers)
            
        # Write our new simulation data at the very bottom of the file
        writer.writerow(new_record)
        
    print(f"Successfully appended record to {log_file}")

# --- Example of how to use these in your script ---
if __name__ == "__main__":
    # Send your parsed CSV file over to the server
    send_file_to_cadence("cadence_outputs\input1.txt", "~/LLM/Evaluation_Tool_Ocean/")
    
    #tracking simulation start time and end time to determine duration
    start_time = datetime.now()

    run_cadence_simulation()
    
    end_time = datetime.now()

    duration = end_time - start_time
    print(f"Simulation duration :{duration}")

    # download the Cadence simulation results back to Windows
    get_file_from_cadence("~/LLM/Evaluation_Tool_Ocean/result.txt", "H:\\My Documents\\Desktop\\Capstone\\Test data")

    log_simulation_data("Test data\input.txt","H:\\My Documents\\Desktop\\Capstone\\Test data\\result.txt", start_time,end_time,"H:\\My Documents\\Desktop\\Capstone\\simulation_logs.csv")

    print("--- Pipeline Finished Successfully ---")