import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import os
import umap
import re
from pathlib import Path

from .library import ELMS, TIMES

def _parse_fispact_lis(filepath):
    """Parses a FISPACT .lis file to extract surface gamma dose rates.
    Args:
        filepath (str): Path to the FISPACT .lis file.
    Returns:
        pd.DataFrame: A DataFrame with 'time_s' and 'dose_sv_h' columns.
    """
    times = []
    dose_rates = []
    
    # Key string from the blueprint
    search_string = "Surface gamma dose rate"
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if search_string in line:
                    parts = line.split()
                    # Example line: "at      3.154E+07 s    Surface gamma dose rate =   1.234E+00 Sv/h"
                    time_index = parts.index("s") - 1
                    dose_index = parts.index("Sv/h") - 2
                    
                    times.append(float(parts[time_index]))
                    dose_rates.append(float(parts[dose_index]))
    except FileNotFoundError:
        print(f"Warning: FISPACT file not found: {filepath}")
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse dose from FISPACT file {filepath}: {e}")
        
    return pd.DataFrame({'time_s': times, 'dose_sv_h': dose_rates})

def plot_fispact_comparison(fispact_output, path_a_results, path_b_results, output_dir):
    """Plots FISPACT vs. OpenMC Path A vs. OpenMC Path B dose rates.
    Args:
        fispact_output (str): Path to the FISPACT .lis output file.
        path_a_results (str): Path to the HDF5 results from Path A.
        path_b_results (str): Path to the HDF5 results from Path B.
        output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot FISPACT results
    fispact_df = _parse_fispact_lis(fispact_output)
    if not fispact_df.empty:
        ax.loglog(fispact_df['time_s'], fispact_df['dose_sv_h'], 'o-', label='FISPACT (Surface Dose Eq.)')

    # Plot Path A results
    if os.path.exists(path_a_results):
        with h5py.File(path_a_results, 'r') as f:
            # Convert µSv/h to Sv/h
            ax.loglog(f['dose_times'][:], f['dose'][:] * 1e-6, 's--', label='OpenMC Path A (FISPACT Eq.)')
            
    # Plot Path B results
    if os.path.exists(path_b_results):
        with h5py.File(path_b_results, 'r') as f:
            ax.loglog(f['dose_times'][:], f['dose'][:] * 1e-6, 'v-.', label='OpenMC Path B (Photon Transport)')

    ax.set_xlabel("Time after shutdown (s)")
    ax.set_ylabel("Dose rate (Sv/h)")
    ax.set_title("Dose Rate Comparison: FISPACT vs. OpenMC Workflows")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    output_path = Path(output_dir) / "dose_rate_comparison.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved comparison plot to {output_path}")

def plot_dose_rate_vs_time(results_dir="elem_lib"):
    """Plots contact dose rate vs. cooling time for all elements.

    This function reads the dose rate results from the HDF5 files generated
    by the `run_element` function for each element in the library. It then
    plots the dose rate as a function of time on a log-log scale and saves
    the resulting figure.

    Args:
        results_dir (str, optional): The directory containing the element-specific
            subdirectories with HDF5 results files. Defaults to "elem_lib".
    """
    fig, ax = plt.subplots()

    for element in ELMS:
        h5_file = os.path.join(results_dir, element, f"{element}.h5")
        if os.path.exists(h5_file):
            with h5py.File(h5_file, 'r') as f:
                times = f['dose_times'][:]
                doses = f['dose'][:]
                ax.loglog(times, doses, 'o-', label=element)
        else:
            print(f"Warning: Results file not found for {element} at {h5_file}")

    ax.set_xlabel("Time after shutdown (s)")
    ax.set_ylabel("Contact dose rate (µSv/h)")
    ax.set_title("Contact Dose Rate vs. Cooling Time")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    output_path = os.path.join(results_dir, "dose_vs_time.png")
    fig.savefig(output_path, dpi=300)
    print(f"Saved dose rate plot to {output_path}")

def plot_umap(df, colour='dose_14d', outfile="manifold.png"):
    """Generates and saves a UMAP plot of the alloy design space.

    This function uses the UMAP dimensionality reduction technique to create a
    2D embedding of the high-dimensional alloy composition space. The points
    in the plot are colored based on a specified performance metric (e.g.,
    14-day dose rate).

    Args:
        df (pandas.DataFrame): A DataFrame containing the alloy compositions
            ('x' column) and the performance metrics to plot.
        colour (str, optional): The column in `df` to use for coloring the
            scatter plot points. Defaults to 'dose_14d'.
        outfile (str, optional): The path to save the output PNG image.
            Defaults to "manifold.png".
    """
    X = np.vstack(df['x'])
    emb = umap.UMAP().fit_transform(X)
    c   = df['dose'].str[0]   # 14‑day dose
    plt.figure(figsize=(6,5))
    sc = plt.scatter(emb[:,0], emb[:,1], c=c, s=4,
                     norm='log', cmap='viridis')
    plt.colorbar(sc,label='14‑day dose [Sv h$^{-1}$]')
    plt.axis('off'); plt.tight_layout()
    plt.savefig(outfile, dpi=300)
