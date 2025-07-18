import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import os
import umap
import re
from pathlib import Path
from typing import Tuple, Optional
import openmc

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
