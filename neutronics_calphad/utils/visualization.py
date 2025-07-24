import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import os
import umap
import re
from pathlib import Path
from typing import Tuple, Optional, List
import openmc
import openmc.deplete

from neutronics_calphad.neutronics.library import ELMS, TIMES

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

def _autodetect_group_structure(num_groups: int) -> str:
    """Autodetects the energy group structure based on number of groups.
    
    Args:
        num_groups: Number of energy groups in the flux file.
        
    Returns:
        Name of the energy group structure. If a data file exists for this number
        of groups, returns a descriptive name. Otherwise returns a generic name.
    """
    # Common well-known group structures
    known_structures = {
        1102: "UKAEA-1102",
        709: "CCFE-709", 
        252: "SCALE-252",
        175: "VITAMIN-J-175",
        172: "VITAMIN-J-172",
        100: "CASMO-100",
        69: "WIMS-69",
        66: "XMAS-66"
    }
    
    # Check if we have a data file for this number of groups
    data_file = Path(__file__).parent / "data" / f"ebins_{num_groups}"
    if data_file.exists():
        # If we have a specific name for this structure, use it
        if num_groups in known_structures:
            return known_structures[num_groups]
        else:
            # Generic name for unknown but supported structures
            return f"{num_groups}-group"
    else:
        # Fall back to OpenMC built-in structures
        if num_groups in known_structures:
            return known_structures[num_groups]
        else:
            raise ValueError(
                f"Unknown group structure with {num_groups} groups. "
                f"No data file found at {data_file} and not a known OpenMC structure. "
                f"Known structures: {list(known_structures.keys())}"
            )

def read_fispact_flux(flux_file: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Reads a FISPACT flux file and autodetects the group structure.
    
    Args:
        flux_file: Path to the FISPACT flux file.
        
    Returns:
        Tuple of (energy_boundaries, flux_values, group_structure_name).
        Energy boundaries are in eV (ascending order).
        Flux values are in neutrons/cm²/s.
        
    Raises:
        FileNotFoundError: If flux file doesn't exist.
        ValueError: If file format is invalid or group structure unknown.
    """
    if not flux_file.exists():
        raise FileNotFoundError(f"Flux file not found: {flux_file}")
    
    with open(flux_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError("Invalid flux file format: too few lines")
    
    # First line should be number of groups
    try:
        num_groups = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid flux file format: first line must be number of groups")
    
    # Read flux values
    flux_values = []
    for i in range(1, min(len(lines), num_groups + 1)):
        try:
            flux_values.append(float(lines[i].strip()))
        except ValueError:
            raise ValueError(f"Invalid flux value on line {i + 1}: {lines[i].strip()}")
    
    if len(flux_values) != num_groups:
        raise ValueError(
            f"Mismatch: expected {num_groups} flux values, got {len(flux_values)}"
        )
    
    flux_array = np.array(flux_values)
    
    # Autodetect group structure and get energy boundaries
    group_structure = _autodetect_group_structure(num_groups)
    print(f"Detected {group_structure} energy group structure ({num_groups} groups)")
    
    # Try to load energy boundaries from data files first
    try:
        print(f"Loading energy boundaries from data/ebins_{num_groups}...")
        energy_boundaries = _load_energy_boundaries(num_groups)
        
        print(f"Energy boundaries loaded: {len(energy_boundaries)} boundaries")
        print(f"Using flux array as-is (no reversal)")
        
        # DEBUG: Show energy and flux ordering
        print(f"First 5 energy boundaries: {energy_boundaries[:5]}")
        print(f"Last 5 energy boundaries: {energy_boundaries[-5:]}")
        print(f"First 5 flux values: {flux_array[:5]}")
        print(f"Last 5 flux values: {flux_array[-5:]}")
        
        # Find where 14.1 MeV is
        energy_14_1_mev = 14.1e6  # eV
        closest_idx = np.argmin(np.abs(energy_boundaries - energy_14_1_mev))
        print(f"14.1 MeV = {energy_14_1_mev:.1e} eV")
        print(f"Closest energy boundary: {energy_boundaries[closest_idx]:.1e} eV at index {closest_idx}")
        
        # Validate that we have the right number of groups
        if len(energy_boundaries) != num_groups + 1:
            print(f"WARNING: Energy boundaries ({len(energy_boundaries)}) != groups + 1 ({num_groups + 1})")
        
        return energy_boundaries, flux_array, group_structure
        
    except Exception as e:
        print(f"Failed to load energy boundaries from data file: {e}")
        print(f"Falling back to OpenMC built-in group structures...")
    
    # Try OpenMC's built-in group structures
    try:
        print(f"Attempting to get energy structure from OpenMC for: {group_structure}")
        energy_filter = openmc.EnergyFilter.from_group_structure(group_structure)
        print(f"Successfully created energy filter")
        
        energy_boundaries = np.array(energy_filter.bins)
        print(f"Energy boundaries shape: {energy_boundaries.shape}")
        print(f"Energy range: {energy_boundaries[0]:.2e} - {energy_boundaries[-1]:.2e} eV")
        
        # Use flux array as-is
        print(f"Using flux array as-is for {group_structure}")
        
        return energy_boundaries, flux_array, group_structure
        
    except Exception as e:
        import traceback
        print(f"OpenMC group structure failed:")
        traceback.print_exc()
        
        # Provide a fallback - create a simple energy grid
        print(f"\nFalling back to creating a simple energy grid for {num_groups} groups")
        energy_boundaries = np.logspace(-5, np.log10(20e6), num_groups + 1)  # eV
        print(f"Created fallback energy grid: {energy_boundaries[0]:.2e} - {energy_boundaries[-1]:.2e} eV")
        
        # Don't reverse for fallback grid (already in ascending order)
        return energy_boundaries, flux_array, f"{group_structure} (fallback grid)"
    
    return energy_boundaries, flux_array, group_structure

def plot_fispact_flux(flux_file: Path, output_dir: Optional[Path] = None, 
                     log_scale: bool = True, show_plot: bool = False) -> Path:
    """Plots a FISPACT flux spectrum with proper energy boundaries.
    
    Args:
        flux_file: Path to the FISPACT flux file.
        output_dir: Directory to save the plot. If None, saves in same directory as flux file.
        log_scale: Whether to use log scale for both axes.
        show_plot: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot file.
        
    Raises:
        FileNotFoundError: If flux file doesn't exist.
        ValueError: If file format is invalid.
    """
    # Read flux data
    energy_boundaries, flux_values, group_structure = read_fispact_flux(flux_file)
    
    # Calculate energy midpoints for plotting (geometric mean of bin edges)
    energy_midpoints = np.sqrt(energy_boundaries[:-1] * energy_boundaries[1:])
    
    # Calculate bin widths for proper flux density
    bin_widths = energy_boundaries[1:] - energy_boundaries[:-1]
    
    # DEBUG: Print energy-flux mapping
    print(f"\nDEBUG: Energy-Flux mapping:")
    print(f"Number of energy boundaries: {len(energy_boundaries)}")
    print(f"Number of energy midpoints: {len(energy_midpoints)}")
    print(f"Number of flux values: {len(flux_values)}")
    
    # Find the 14.1 MeV region
    energy_14_1_mev = 14.1e6  # eV
    closest_group = np.argmin(np.abs(energy_midpoints - energy_14_1_mev))
    print(f"14.1 MeV closest to group {closest_group}:")
    print(f"  Energy midpoint: {energy_midpoints[closest_group]:.2e} eV = {energy_midpoints[closest_group]/1e6:.2f} MeV")
    print(f"  Energy range: [{energy_boundaries[closest_group]:.2e}, {energy_boundaries[closest_group+1]:.2e}] eV")
    print(f"  Flux value: {flux_values[closest_group]:.2e} neutrons/cm²/s")
    
    # Show peak flux location
    peak_flux_idx = np.argmax(flux_values)
    print(f"Peak flux at group {peak_flux_idx}:")
    print(f"  Energy midpoint: {energy_midpoints[peak_flux_idx]:.2e} eV = {energy_midpoints[peak_flux_idx]/1e6:.2f} MeV")
    print(f"  Flux value: {flux_values[peak_flux_idx]:.2e} neutrons/cm²/s")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if log_scale:
        # Plot as step function to show group structure clearly
        ax.loglog(energy_boundaries[:-1] / 1e6, flux_values, 'b-', linewidth=2, 
                 label=f'Flux Spectrum ({group_structure})')
        # Also plot as steps to show the group structure
        ax.step(energy_boundaries[:-1] / 1e6, flux_values, 'r--', alpha=0.7, 
               linewidth=1, where='post', label='Group Structure')
    else:
        ax.plot(energy_midpoints / 1e6, flux_values, 'b-', linewidth=2,
               label=f'Flux Spectrum ({group_structure})')
    
    # Formatting
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Flux (neutrons/cm²/s)')
    ax.set_title(f'FISPACT Flux Spectrum\n{flux_file.name} - {group_structure} Group Structure')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    # Add some statistics as text
    total_flux = np.sum(flux_values * bin_widths)
    max_flux = np.max(flux_values)
    max_energy = energy_midpoints[np.argmax(flux_values)] / 1e6
    
    stats_text = f'Total Flux: {total_flux:.2e} n/cm²/s\n'
    stats_text += f'Peak Flux: {max_flux:.2e} n/cm²/s\n'
    stats_text += f'Peak Energy: {max_energy:.2f} MeV'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir is None:
        output_dir = flux_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_filename = f"{flux_file.stem}_spectrum.png"
    plot_path = output_dir / plot_filename
    
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved flux spectrum plot to: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plot_path

def _load_energy_boundaries(num_groups: int) -> np.ndarray:
    """Loads energy boundaries from data files based on number of groups.
    
    Args:
        num_groups: Number of energy groups to load boundaries for.
    
    Returns:
        Energy boundaries in eV (ascending order for OpenMC compatibility).
        
    Raises:
        FileNotFoundError: If the energy boundaries file is not found.
    """
    # Try to find the ebins_X file in common locations  
    filename = f"ebins_{num_groups}"
    possible_paths = [
        Path(__file__).parent / "data" / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path(__file__).parent.parent / "fispact_comparison" / filename,
        Path("fispact_comparison") / filename,
        Path("data") / filename,
        Path(filename),
    ]
    
    ebins_file = None
    for path in possible_paths:
        if path.exists():
            ebins_file = path
            break
    
    if ebins_file is None:
        raise FileNotFoundError(
            f"Energy boundaries file {filename} not found. "
            f"Searched in: {[str(p) for p in possible_paths]}"
        )
    
    print(f"Loading {num_groups}-group energy boundaries from: {ebins_file}")
    
    with open(ebins_file, 'r') as f:
        lines = f.readlines()
    
    # Skip the first two header lines ('UKAEA-1102' and '1102')
    data_lines = lines[2:]
    
    # Parse the comma-separated energy values
    energy_text = "".join(data_lines)
    energy_strings = [e.strip() for e in energy_text.replace(',', ' ').split() if e.strip()]
    
    # Convert to float
    energies_raw = np.array([float(e) for e in energy_strings])
    
    print(f"Loaded {len(energies_raw)} energy boundaries")
    print(f"Raw energy range: {energies_raw[0]:.2e} - {energies_raw[-1]:.2e} eV")
    print(f"First 5 raw energies: {energies_raw[:5]}")
    print(f"Last 5 raw energies: {energies_raw[-5:]}")
    
    # Check if already sorted
    if np.all(energies_raw[:-1] <= energies_raw[1:]):
        print("Energy boundaries are already in ascending order")
        energy_boundaries = energies_raw
    elif np.all(energies_raw[:-1] >= energies_raw[1:]):
        print("Energy boundaries are in descending order - sorting to ascending")
        energy_boundaries = np.sort(energies_raw)
    else:
        print("Energy boundaries are not monotonic - sorting to ascending")
        energy_boundaries = np.sort(energies_raw)
    
    print(f"Final energy range (ascending): {energy_boundaries[0]:.2e} - {energy_boundaries[-1]:.2e} eV")
    
    # Verify 14.1 MeV is in range
    energy_14_1_mev = 14.1e6
    if energy_boundaries[0] <= energy_14_1_mev <= energy_boundaries[-1]:
        closest_idx = np.argmin(np.abs(energy_boundaries - energy_14_1_mev))
        print(f"14.1 MeV = {energy_14_1_mev:.1e} eV found at index {closest_idx}: {energy_boundaries[closest_idx]:.1e} eV ✅")
    else:
        print(f"WARNING: 14.1 MeV = {energy_14_1_mev:.1e} eV is NOT in energy range!")
    
    return energy_boundaries


def plot_top_n_activities_vs_time(
    activity_data: List[Tuple[float, dict]],
    top_n: int = 5,
    title: str = "Top Activities vs Time",
    ylabel: str = "Activity (Bq)",
    xlabel: str = "Time",
    figsize: Tuple[int, int] = (10, 6),
    plot_total: bool = True,
    time_axis_units: str = "s"
) -> None:
    """
    Plot the top N nuclides by peak activity over time from a list of (time, activity_dict) tuples.

    Optionally, plot the total activity as a dashed black line.

    Args:
        activity_data: List of tuples (time_in_seconds, {nuclide: activity, ...}).
        top_n: Number of nuclides with highest peak activity to plot (default 5).
        title: Plot title.
        ylabel: Y-axis label.
        xlabel: X-axis label (will be updated to match time_axis_units).
        figsize: Figure size.
        plot_total: Whether to plot the total activity as a dashed black line (default True).
        time_axis_units: Units for the time axis ('s', 'm', 'h', 'd', 'y').
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Time unit conversion factors
    unit_factors = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'y': 365.25 * 24 * 3600
    }
    unit_labels = {
        's': 'Time (s)',
        'm': 'Time (min)',
        'h': 'Time (h)',
        'd': 'Time (days)',
        'y': 'Time (years)'
    }
    if time_axis_units not in unit_factors:
        raise ValueError(f"Invalid time_axis_units: {time_axis_units}. Must be one of {list(unit_factors.keys())}")

    # Extract all nuclide names
    all_nuclides = set()
    for _, d in activity_data:
        all_nuclides.update(d.keys())
    all_nuclides = sorted(all_nuclides)

    # Build time and activity arrays
    times = np.array([t for t, _ in activity_data]) / unit_factors[time_axis_units]
    activity_matrix = {nuc: np.array([d.get(nuc, 0.0) for _, d in activity_data]) for nuc in all_nuclides}

    # Find top N nuclides by peak activity
    peak_activities = {nuc: np.max(vals) for nuc, vals in activity_matrix.items()}
    top_nuclides = sorted(peak_activities, key=peak_activities.get, reverse=True)[:top_n]

    plt.figure(figsize=figsize)
    for nuc in top_nuclides:
        plt.plot(times, activity_matrix[nuc], label=nuc)

    if plot_total:
        # Calculate total activity at each time point
        total_activity = np.zeros_like(times)
        for nuc in all_nuclides:
            total_activity += activity_matrix[nuc]
        plt.plot(
            times,
            total_activity,
            label="Total",
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=10
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(unit_labels[time_axis_units])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show() 

def plot_top_n_vs_time(
    data: Tuple[np.ndarray, List[dict]],
    top_n: int = 5,
    y_units: str = "",
    y_scale: str = "log",
    time_axis_units: str = "s",
    plot_total: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (10, 6),
    x_scale: str = "log",
    y_lim: float = None
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot the top N nuclides by peak value (dose, activity, atoms, etc.) over time.

    Args:
        data: Tuple of (np.ndarray of times, list of dicts of nuclide values).
        top_n: Number of nuclides with highest peak value to plot (default 5).
        y_units: Units for the y-axis (e.g., "Bq/g", "µSv/h", "atoms").
        y_scale: Y-axis scale ("log" or "linear").
        time_axis_units: Units for the time axis ('s', 'm', 'h', 'd', 'y').
        plot_total: Whether to plot the total as a dashed black line (default True).
        title: Plot title (default blank).
        xlabel: X-axis label (default blank).
        ylabel: Y-axis label (default blank; if blank, uses y_units if provided).
        figsize: Figure size.
        x_scale: X-axis scale ("log" or "linear").
        y_lim: If set, sets the lower y-limit of the plot to this value (does not filter data).
    Returns:
        (fig, ax): The matplotlib Figure and Axes objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    times, results = data
    # Time unit conversion factors
    unit_factors = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'y': 365.25 * 24 * 3600
    }
    unit_labels = {
        's': 'Time (s)',
        'm': 'Time (min)',
        'h': 'Time (h)',
        'd': 'Time (days)',
        'y': 'Time (years)'
    }
    if time_axis_units not in unit_factors:
        raise ValueError(f"Invalid time_axis_units: {time_axis_units}. Must be one of {list(unit_factors.keys())}")
    times_converted = times / unit_factors[time_axis_units]

    # Collect all nuclide names
    all_nuclides = set()
    for d in results:
        all_nuclides.update(d.keys())
    all_nuclides = sorted(all_nuclides)

    # Build value matrix (nuclide -> np.array of values)
    # Use a very small value (1e-30) for missing entries if log scale, else 0.0
    fill_value = 1e-30 if y_scale == "log" else 0.0
    value_matrix = {
        nuc: np.array([d.get(nuc, fill_value) if d.get(nuc, None) not in [None, 0.0] else fill_value for d in results])
        for nuc in all_nuclides
    }

    # Find top N nuclides by peak value
    peak_values = {nuc: np.max(vals) for nuc, vals in value_matrix.items()}
    top_nuclides = sorted(peak_values, key=peak_values.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    for nuc in top_nuclides:
        ax.plot(times_converted, value_matrix[nuc], label=nuc)

    if plot_total:
        total = np.zeros_like(times_converted)
        for nuc in all_nuclides:
            total += value_matrix[nuc]
        ax.plot(
            times_converted,
            total,
            label="Total",
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=10
        )

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if not xlabel:
        xlabel = unit_labels[time_axis_units]
    ax.set_xlabel(xlabel)
    if not ylabel:
        ylabel = y_units
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if y_lim is not None:
        ax.set_ylim(bottom=y_lim)
    fig.tight_layout()
    return fig, ax 

def plot_top_n_vs_time_comparison(
    data1: Tuple[np.ndarray, List[dict]],
    data2: Tuple[np.ndarray, List[dict]],
    top_n: int = 5,
    y_units: str = "",
    y_scale: str = "log",
    time_axis_units: str = "s",
    plot_total: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (10, 6),
    x_scale: str = "log",
    y_lim: float = None,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    marker1: str = "o",
    marker2: str = "s"
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot a comparison of the top N nuclides from two datasets on the same axes.
    Each nuclide is plotted with the same color for both datasets, but different markers.

    Args:
        data1: Tuple of (np.ndarray of times, list of dicts of nuclide values) for the first dataset.
        data2: Tuple of (np.ndarray of times, list of dicts of nuclide values) for the second dataset.
        top_n: Number of nuclides with highest peak value (from data1) to plot (default 5).
        y_units: Units for the y-axis (e.g., "Bq/g", "µSv/h", "atoms").
        y_scale: Y-axis scale ("log" or "linear").
        time_axis_units: Units for the time axis ('s', 'm', 'h', 'd', 'y').
        plot_total: Whether to plot the total as a dashed black line (default True).
        title: Plot title (default blank).
        xlabel: X-axis label (default blank).
        ylabel: Y-axis label (default blank; if blank, uses y_units if provided).
        figsize: Figure size.
        x_scale: X-axis scale ("log" or "linear").
        y_lim: If set, sets the lower y-limit of the plot to this value (does not filter data).
        label1: Legend label for the first dataset.
        label2: Legend label for the second dataset.
        marker1: Marker style for the first dataset.
        marker2: Marker style for the second dataset.
    Returns:
        (fig, ax): The matplotlib Figure and Axes objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import cycle

    # Get top_n nuclides from data1
    times1, results1 = data1
    all_nuclides = set()
    for d in results1:
        all_nuclides.update(d.keys())
    all_nuclides = sorted(all_nuclides)
    fill_value = 1e-30 if y_scale == "log" else 0.0
    value_matrix1 = {
        nuc: np.array([d.get(nuc, fill_value) if d.get(nuc, None) not in [None, 0.0] else fill_value for d in results1])
        for nuc in all_nuclides
    }
    peak_values = {nuc: np.max(vals) for nuc, vals in value_matrix1.items()}
    top_nuclides = sorted(peak_values, key=peak_values.get, reverse=True)[:top_n]

    # Prepare color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    color_map = {nuc: next(colors) for nuc in top_nuclides}

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # Plot data1
    times1_converted = times1 / {
        's': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 365.25 * 24 * 3600
    }[time_axis_units]
    for nuc in top_nuclides:
        ax.plot(
            times1_converted,
            value_matrix1[nuc],
            label=f"{nuc} ({label1})",
            color=color_map[nuc],
            marker=marker1,
            linestyle="-"
        )

    # Plot data2
    times2, results2 = data2
    value_matrix2 = {
        nuc: np.array([d.get(nuc, fill_value) if d.get(nuc, None) not in [None, 0.0] else fill_value for d in results2])
        for nuc in top_nuclides
    }
    times2_converted = times2 / {
        's': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 365.25 * 24 * 3600
    }[time_axis_units]
    for nuc in top_nuclides:
        ax.plot(
            times2_converted,
            value_matrix2[nuc],
            label=f"{nuc} ({label2})",
            color=color_map[nuc],
            marker=marker2,
            linestyle="--"
        )

    # Plot totals if requested
    if plot_total:
        total1 = np.zeros_like(times1_converted)
        for nuc in all_nuclides:
            total1 += value_matrix1.get(nuc, 0.0)
        ax.plot(
            times1_converted,
            total1,
            label=f"Total ({label1})",
            color="black",
            linestyle="-",
            linewidth=2,
            zorder=10
        )
        total2 = np.zeros_like(times2_converted)
        for nuc in all_nuclides:
            total2 += value_matrix2.get(nuc, 0.0)
        ax.plot(
            times2_converted,
            total2,
            label=f"Total ({label2})",
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=10
        )

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    unit_labels = {
        's': 'Time (s)',
        'm': 'Time (min)',
        'h': 'Time (h)',
        'd': 'Time (days)',
        'y': 'Time (years)'
    }
    if not xlabel:
        xlabel = unit_labels[time_axis_units]
    ax.set_xlabel(xlabel)
    if not ylabel:
        ylabel = y_units
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if y_lim is not None:
        ax.set_ylim(bottom=y_lim)
    fig.tight_layout()
    return fig, ax 

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from typing import List, Tuple, Dict

# Make sure to install adjustText: pip install adjustText
from adjustText import adjust_text

def plot_top_n_vs_time_comparison(
    data1: Tuple[np.ndarray, List[Dict[str, float]]],
    data2: Tuple[np.ndarray, List[Dict[str, float]]],
    top_n: int = 5,
    y_units: str = "",
    y_scale: str = "log",
    time_axis_units: str = "s",
    plot_total: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (14, 9),
    x_scale: str = "log",
    y_lim: float = None,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    marker1: str = None,
    marker2: str = None
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plots a memory-efficient comparison of top nuclides from two datasets, 
    using the `adjustText` library for smart, non-overlapping label placement.

    Args:
        data1: Tuple of (times, results) for the first dataset.
        data2: Tuple of (times, results) for the second dataset.
        top_n: Number of top nuclides to select from each dataset.
        y_units: String for the y-axis units.
        y_scale: Y-axis scale ("log" or "linear").
        time_axis_units: Units for the time axis ('s', 'm', 'h', 'd', 'y').
        plot_total: Whether to plot the total activity/dose rate.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        x_scale: X-axis scale ("log" or "linear").
        y_lim: Lower y-limit for the plot.
        label1: Legend label for the first dataset.
        label2: Legend label for the second dataset.
        marker1: Marker style for the first dataset's lines.
        marker2: Marker style for the second dataset's lines.

    Returns:
        (fig, ax): The matplotlib Figure and Axes objects.
    """
    fill_value = 1e-30 if y_scale == "log" else 0.0

    # --- Helper functions for memory-efficient processing ---

    def _find_top_nuclides(results: List[Dict[str, float]]) -> List[str]:
        """Efficiently finds top N nuclides by checking peaks without storing full arrays."""
        peak_values = {}
        for d in results:
            for nuc, val in d.items():
                peak_values[nuc] = max(peak_values.get(nuc, -1.0), val)
        return sorted(peak_values, key=peak_values.get, reverse=True)[:top_n]

    def _get_value_matrix_for_selection(
        data: Tuple[np.ndarray, List[Dict[str, float]]],
        nuclides_to_get: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extracts the time-series ONLY for a pre-selected list of nuclides."""
        _, results = data
        temp_matrix = {nuc: [] for nuc in nuclides_to_get}
        for d in results:
            for nuc in nuclides_to_get:
                temp_matrix[nuc].append(d.get(nuc, fill_value))
        return {nuc: np.array(vals) for nuc, vals in temp_matrix.items()}

    def _get_total(data: Tuple[np.ndarray, List[Dict[str, float]]]) -> np.ndarray:
        """Efficiently calculates the total sum over time."""
        _, results = data
        return np.array([sum(d.values()) for d in results])

    # --- Main Logic ---

    # 1. Efficiently find top nuclides from each dataset
    top_nuclides1 = _find_top_nuclides(data1[1])
    top_nuclides2 = _find_top_nuclides(data2[1])
    nuclides_to_plot = sorted(list(set(top_nuclides1) | set(top_nuclides2)))

    # 2. Extract value matrices ONLY for the selected nuclides
    value_matrix1 = _get_value_matrix_for_selection(data1, nuclides_to_plot)
    value_matrix2 = _get_value_matrix_for_selection(data2, nuclides_to_plot)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=figsize)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    color_map = {nuc: next(colors) for nuc in nuclides_to_plot}
    
    time_conversion_factor = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 365.25 * 24 * 3600}[time_axis_units]
    times1_converted = data1[0] / time_conversion_factor
    times2_converted = data2[0] / time_conversion_factor

    # --- Plotting and Collecting Labels ---
    texts = []
    for nuc in nuclides_to_plot:
        # Plot data lines
        y1 = value_matrix1[nuc]
        y2 = value_matrix2[nuc]
        ax.plot(times1_converted, y1, color=color_map[nuc], marker=marker1, linestyle="-", label=f"__nolegend__")
        ax.plot(times2_converted, y2, color=color_map[nuc], marker=marker2, linestyle="--", label=f"__nolegend__")
        
        # Determine the peak position for the label (from the more dominant curve)
        if np.max(y1) > np.max(y2):
            idx = np.argmax(y1)
            x_pos, y_pos = times1_converted[idx], y1[idx]
        else:
            idx = np.argmax(y2)
            x_pos, y_pos = times2_converted[idx], y2[idx]
        
        # Create a text object and add it to a list for later adjustment
        texts.append(ax.text(x_pos, y_pos, nuc, color=color_map[nuc], fontsize=9))

    # --- Plot Totals ---
    if plot_total:
        total1 = _get_total(data1)
        ax.plot(times1_converted, total1, color="black", linestyle="-", linewidth=2.5, zorder=10)
        total2 = _get_total(data2)
        ax.plot(times2_converted, total2, color="black", linestyle="--", linewidth=2.5, zorder=10)

    # --- Auto-adjust all labels at once ---
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # --- Final Touches ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label=f'Total ({label1})'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label=f'Total ({label2})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    unit_labels = {'s': 'Time (s)', 'm': 'Time (min)', 'h': 'Time (h)', 'd': 'Time (days)', 'y': 'Time (years)'}
    ax.set_xlabel(xlabel or unit_labels[time_axis_units])
    ax.set_ylabel(ylabel or y_units)
    ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(bottom=y_lim)
        
    #fig.tight_layout()
    
    return fig, ax
