"""Compare TCHEA8 and Smith2000 thermophysical properties for V-4Cr-4Ti alloy.

This script reads property data from two CSV files and generates comparison plots
for various thermophysical properties as a function of temperature.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Define color palette with alpha
COLORS = {
    "tchea8": "#2A33C3",
    "smith2000": "#A35D00",
    "color3": "#0B7285",
    "color4": "#8F2D56",
    "color5": "#6E8B00",
}
ALPHA = 0.58


def load_data(file_path: Path) -> pd.DataFrame:
    """Load thermophysical property data from CSV file.

    Args:
        file_path: Path to the CSV file containing property data.

    Returns:
        DataFrame containing the property data.
    """
    df = pd.read_csv(file_path)
    return df


def create_comparison_plot(
    smith2000_df: pd.DataFrame,
    tchea8_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create comparison plots for all thermophysical properties.

    Args:
        smith2000_df: DataFrame containing Smith2000 property data.
        tchea8_df: DataFrame containing TCHEA8 property data.
        output_dir: Directory to save the output plots.
    """
    # Properties to plot (exclude Temperature, Density and Poisson's Ratio will be combined)
    properties = [
        "Young's Modulus",
        "CTE",
        "Thermal Conductivity",
        "Heat Capacity",
        "Yield Strength",
    ]

    # Property units for axis labels
    units = {
        "Density": "g/cm³",
        "Poisson's Ratio": "",
        "Young's Modulus": "GPa",
        "CTE": "10⁻⁶/K",
        "Thermal Conductivity": "W/(m·K)",
        "Heat Capacity": "J/(kg·K)",
        "Yield Strength": "MPa",
    }

    # Create subplots (3x2 grid for 6 plots: 1 combined + 5 individual)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(
        "TCHEA8 vs Smith2000: V-4Cr-4Ti Thermophysical Properties",
        fontsize=16,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    # First plot: Combined Density and Poisson's Ratio bar chart
    ax = axes_flat[0]

    # Get mean values for Density (ignoring NaN)
    smith2000_density = smith2000_df["Density"].dropna().mean()
    tchea8_density = tchea8_df["Density"].dropna().mean()

    # Get mean values for Poisson's Ratio (ignoring NaN)
    smith2000_poisson = smith2000_df["Poisson's Ratio"].dropna().mean()
    tchea8_poisson = tchea8_df["Poisson's Ratio"].dropna().mean()

    # Prepare data for plotting
    properties_to_plot = ["Density", "Poisson's Ratio"]
    smith2000_values = [
        smith2000_density if not pd.isna(smith2000_density) else 0,
        smith2000_poisson if not pd.isna(smith2000_poisson) else 0,
    ]
    tchea8_values = [
        tchea8_density if not pd.isna(tchea8_density) else 0,
        tchea8_poisson if not pd.isna(tchea8_poisson) else 0,
    ]

    # Bar positions
    x_pos = range(len(properties_to_plot))
    bar_width = 0.35

    # Plot bars
    bars1 = ax.bar(
        [x - bar_width/2 for x in x_pos],
        smith2000_values,
        bar_width,
        label="Smith2000",
        color=COLORS["smith2000"],
        alpha=ALPHA,
    )

    bars2 = ax.bar(
        [x + bar_width/2 for x in x_pos],
        tchea8_values,
        bar_width,
        label="TCHEA8",
        color=COLORS["tchea8"],
        alpha=ALPHA,
    )

    # Add data labels to bars
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")

    # Format plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(properties_to_plot, fontsize=10, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Plot remaining properties
    for idx, prop in enumerate(properties, start=1):
        ax = axes_flat[idx]

        # Plot Smith2000 data
        smith2000_mask = smith2000_df[prop].notna()
        if smith2000_mask.any():
            ax.plot(
                smith2000_df.loc[smith2000_mask, "Temperature"],
                smith2000_df.loc[smith2000_mask, prop],
                marker="o",
                markersize=8,
                linewidth=2,
                label="Smith2000",
                color=COLORS["smith2000"],
                alpha=ALPHA,
            )

        # Plot TCHEA8 data
        tchea8_mask = tchea8_df[prop].notna()
        if tchea8_mask.any():
            ax.plot(
                tchea8_df.loc[tchea8_mask, "Temperature"],
                tchea8_df.loc[tchea8_mask, prop],
                marker="s",
                markersize=8,
                linewidth=2,
                label="TCHEA8",
                color=COLORS["tchea8"],
                alpha=ALPHA,
            )

        # Format plot
        ax.set_xlabel("Temperature (K)", fontsize=10, fontweight="bold")
        unit_str = f" ({units[prop]})" if units[prop] else ""
        ax.set_ylabel(f"{prop}{unit_str}", fontsize=10, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save figure in both PNG and PDF formats
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_png = output_dir / "tchea8_vs_smith2000_comparison.png"
    output_path_pdf = output_dir / "tchea8_vs_smith2000_comparison.pdf"
    
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_path_png}")
    
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_path_pdf}")


def create_individual_plots(
    smith2000_df: pd.DataFrame,
    tchea8_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create individual comparison plots for each property.

    Args:
        smith2000_df: DataFrame containing Smith2000 property data.
        tchea8_df: DataFrame containing TCHEA8 property data.
        output_dir: Directory to save the output plots.
    """
    properties = [
        "Density",
        "Poisson's Ratio",
        "Young's Modulus",
        "CTE",
        "Thermal Conductivity",
        "Heat Capacity",
        "Yield Strength",
    ]

    units = {
        "Density": "g/cm³",
        "Poisson's Ratio": "",
        "Young's Modulus": "GPa",
        "CTE": "10⁻⁶/K",
        "Thermal Conductivity": "W/(m·K)",
        "Heat Capacity": "J/(kg·K)",
        "Yield Strength": "MPa",
    }

    individual_dir = output_dir / "individual_properties"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for prop in properties:
        # Check if there's any data for this property
        has_smith = smith2000_df[prop].notna().any()
        has_tchea = tchea8_df[prop].notna().any()

        if not has_smith and not has_tchea:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Smith2000 data
        if has_smith:
            smith2000_mask = smith2000_df[prop].notna()
            ax.plot(
                smith2000_df.loc[smith2000_mask, "Temperature"],
                smith2000_df.loc[smith2000_mask, prop],
                marker="o",
                markersize=10,
                linewidth=2.5,
                label="Smith2000",
                color=COLORS["smith2000"],
                alpha=ALPHA,
            )

        # Plot TCHEA8 data
        if has_tchea:
            tchea8_mask = tchea8_df[prop].notna()
            ax.plot(
                tchea8_df.loc[tchea8_mask, "Temperature"],
                tchea8_df.loc[tchea8_mask, prop],
                marker="s",
                markersize=10,
                linewidth=2.5,
                label="TCHEA8",
                color=COLORS["tchea8"],
                alpha=ALPHA,
            )

        # Format plot
        ax.set_xlabel("Temperature (K)", fontsize=12, fontweight="bold")
        unit_str = f" ({units[prop]})" if units[prop] else ""
        ax.set_ylabel(f"{prop}{unit_str}", fontsize=12, fontweight="bold")
        ax.set_title(
            f"V-4Cr-4Ti: {prop}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.9, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()

        # Save figure in both PNG and PDF formats
        safe_filename = prop.replace("'", "").replace(" ", "_").lower()
        output_path_png = individual_dir / f"{safe_filename}_comparison.png"
        output_path_pdf = individual_dir / f"{safe_filename}_comparison.pdf"
        
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path_png}")
        
        plt.savefig(output_path_pdf, bbox_inches="tight")
        print(f"Saved: {output_path_pdf}")

        plt.close()


def main() -> None:
    """Main function to run the comparison analysis."""
    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / "data" / "thermo-physical-data"

    smith2000_file = (
        data_dir / "Smith2000_ThermophysicalV4Cr4Ti_Properties.csv"
    )
    tchea8_file = (
        data_dir / "TCHEA8_ThermophysicalV4Cr4Ti_Properties.csv"
    )

    output_dir = base_dir / "plots"

    # Load data
    print("Loading data...")
    smith2000_df = load_data(smith2000_file)
    tchea8_df = load_data(tchea8_file)

    print(f"Smith2000 data shape: {smith2000_df.shape}")
    print(f"TCHEA8 data shape: {tchea8_df.shape}")

    # Create comparison plots
    print("\nCreating combined comparison plot...")
    create_comparison_plot(smith2000_df, tchea8_df, output_dir)

    print("\nCreating individual property plots...")
    create_individual_plots(smith2000_df, tchea8_df, output_dir)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

