# parallel_run.py

import subprocess
from concurrent.futures import ProcessPoolExecutor
import glob

# Define the path to data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/"

# List all the .npy files you want to process in the data directory with name starting with completeness_map
npy_files = glob.glob(DATA_DIRECTORY + "completeness_map*.npy")

def run_script(npy_file):
    command = f"python process_data.py {npy_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

if __name__ == "__main__":
    # Ensure there are at most 8 tasks running in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_script, npy_file) for npy_file in npy_files]

        for future in futures:
            stdout, stderr = future.result()
            print(stdout)
            if stderr:
                print(stderr)
