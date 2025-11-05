### use this if you want to run the steering file for all signals. 
import re
import csv
import os

def parse_reduction_factors(logfile, csv_out=None):
    results = {}
    current_run = None
    output_path = None

    with open(logfile, "r") as f:
        for line in f:
            # Detect the "Running:" line
            if line.startswith("Running:"):
                current_run = line.strip()
                results[current_run] = []  # list of (output, reduction) tuples
            
            # Detect the output file path
            elif "Output file path:" in line:
                output_path = next(f).strip()
                output_path = os.path.basename(output_path)  # keep only filename
            
            # Detect the reduction factor
            elif "Reduction factor local:" in line:
                reduction_factor = float(line.split(":")[1].strip())
                if current_run and output_path:
                    results[current_run].append((output_path, reduction_factor))
                output_path = None  # reset for next loop
    
    # Optionally write to CSV
    if csv_out:
        with open(csv_out, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["drmin", "drmax", "selection", "output_file", "reduction_factor"])
            
            for run, entries in results.items():
                # Extract drmin, drmax, selection with regex
                match = re.search(r"drmin=([\d.]+), drmax=([\d.]+), selection=([\d.]+)", run)
                drmin, drmax, selection = match.groups() if match else ("", "", "")
                
                for output_file, red in entries:
                    writer.writerow([drmin, drmax, selection, output_file, red])

    return results


if __name__ == "__main__":
    logfile = "scan.log"   # replace with your log filename
    results = parse_reduction_factors(logfile, csv_out="resultsCSVCone.csv")
    
    for run, entries in results.items():
        print(run)
        for output, red in entries:
            print(f"  {output}  -->  Reduction factor: {red}")
