import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("parsed_results.csv")

# Get unique drmax values
drmax_values = sorted(df['drmax'].unique())
processes = df['process'].unique()

# Define line styles and markers
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '*', 'x', 'p']

# Create subplots stacked vertically
fig, axes = plt.subplots(len(drmax_values), 1, figsize=(12, 8 * len(drmax_values)), sharex=True)

# Handle case of single subplot (axes not being an array)
if len(drmax_values) == 1:
    axes = [axes]

# Plot each drmax in its own subplot
for ax, drmax in zip(axes, drmax_values):
    df_drmax = df[df['drmax'] == drmax]
    
    for i, process in enumerate(processes):
        df_proc = df_drmax[df_drmax['process'] == process]
        if not df_proc.empty:
            ax.plot(
                df_proc['selection'], 
                df_proc['reduction_factor'],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                label=process
            )
    
    # Add horizontal reference line
    ax.axhline(0.35, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_title(f"drmax = {drmax}")
    ax.set_ylabel("Reduction Factor")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

axes[-1].set_xlabel("Selection")

plt.tight_layout()
plt.savefig("stacked.png")
plt.show()
