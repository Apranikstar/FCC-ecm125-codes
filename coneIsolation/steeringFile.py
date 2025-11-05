import json
import subprocess
import numpy as np
import re
import csv

# -----------------------------
# Parameters
# -----------------------------
drmin = [0.01]
drmax = [0.2, 0.4, 0.6, 0.8, 1.0]
num_selection = 5

# Prepare list of args: [drmin, drmax, selection_array]
args = []
for dmax in drmax:
    selections = np.linspace(drmin[0], dmax, num_selection)
    args.append([drmin[0], dmax, selections])

# Storage for results
results = []

# -----------------------------
# Run parameter scan
# -----------------------------
for drmin_val, drmax_val, selections in args:
    for selection in selections:
        # Write config.json
        cfg = {
            "drmin": float(drmin_val),
            "drmax": float(drmax_val),
            "selection": float(selection)
        }
        with open("config.json", "w") as f:
            json.dump(cfg, f)

        # Run fccanalysis
        cmd = ["fccanalysis", "run", "cone.py", "--ncpus", "64"]
        print(f"\nRunning: drmin={drmin_val}, drmax={drmax_val:.4f}, selection={selection:.4f}")

        # Stream output and capture
        stdout_lines = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end="")  # live output
                stdout_lines.append(line)
            proc.wait()

        stdout = "".join(stdout_lines)

        # Extract output file path
        output_match = re.search(r"INFO: Output file path:\s*\n?\s*(.+)", stdout)
        output_file = output_match.group(1).strip() if output_match else "N/A"

        # Extract reduction factor
        reduction_match = re.search(r"Reduction factor local:\s*([0-9.]+)", stdout)
        reduction_factor = float(reduction_match.group(1)) if reduction_match else -1

        print(f"  -> Captured Output file: {output_file}, Reduction factor: {reduction_factor:.6f}")

        # Store results
        results.append({
            "drmin": float(drmin_val),
            "drmax": float(drmax_val),
            "selection": float(selection),
            "output_file": output_file,
            "reduction_factor": reduction_factor
        })

# -----------------------------
# Save results to CSV
# -----------------------------
csv_file = "scan_results.csv"
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nAll results saved to {csv_file}")
