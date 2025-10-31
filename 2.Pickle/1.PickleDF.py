import warnings
import uproot
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

input_dir = Path("/eos/user/h/hfatehi/yukawaBDT/on-shell-electron/")
output_dir = Path(".")
tree_name = "events"
max_events = 100_000

for root_file in input_dir.glob("*.root"):
    print(f"\nProcessing: {root_file.name}")

    try:
        with uproot.open(root_file) as f:
            if tree_name not in f:
                print(f"  [!] No tree named '{tree_name}', skipping.")
                continue

            tree = f[tree_name]
            n_entries = tree.num_entries
            print(f"  Found {n_entries:,} events.")

            n_to_read = min(n_entries, max_events)
            if n_entries > max_events:
                print(f"  Reading first {max_events:,} events only.")

            df = tree.arrays(library="pd", entry_stop=n_to_read)

            # Optional: defragment DataFrame (makes it faster to access later)
            df = df.copy()

            output_file = output_dir / (root_file.stem + ".pkl")
            df.to_pickle(output_file)
            print(f"  Saved {len(df):,} events to {output_file}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {root_file.name}: {e}")
