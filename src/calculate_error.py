import re

def calculate_error_margin(file_path, expected_gain):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Regex to find 'Gain:' followed by a number in scientific or decimal notation
        match = re.search(r'Gain:\s*([\d\.eE\+\-]+)', content)
        
        if not match:
            print("Error: 'Gain' not found in the file.")
            return

        actual_gain = float(match.group(1))
        
        # Calculate Margin
        margin = 0.10 * expected_gain
        lower_bound = expected_gain - margin
        upper_bound = expected_gain + margin
        
        error_percent = abs((actual_gain - expected_gain) / expected_gain) * 100
        is_within = lower_bound <= actual_gain <= upper_bound

        print(f"--- Gain Analysis ---")
        print(f"Expected Gain: {expected_gain:.6e}")
        print(f"Actual Gain:   {actual_gain:.6e}")
        print(f"10% Margin:    {margin:.6e}")
        print(f"Range:         [{lower_bound:.6e}, {upper_bound:.6e}]")
        print(f"Error:         {error_percent:.2f}%")
        print(f"Result:        {'PASS (Within Margin)' if is_within else 'FAIL (Outside Margin)'}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Configuration
expected = 2.817416e+01
file_name = "Test data/result.txt"  # directory of result.txt file 

calculate_error_margin(file_name, expected)