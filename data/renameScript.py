import os

folder_path = "raw"

files = [f for f in os.listdir(folder_path) 
         if f.endswith(".csv")]

files.sort()

for index, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_filename = f"dataset_{index}.csv"
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)

print("CSV files renamed successfully.")