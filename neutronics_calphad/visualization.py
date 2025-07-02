import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import umap

from .library import ELMS, TIMES

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
