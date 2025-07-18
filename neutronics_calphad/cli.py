#!/usr/bin/env python3
"""
Command-line interface for neutronics-calphad package.
"""

import argparse
import sys
from pathlib import Path

from . import (
    create_model, plot_model, build_library, 
    plot_dose_rate_vs_time, build_manifold,
    plot_fispact_comparison, plot_fispact_flux,
    ARC_D_SHAPE, SPHERICAL
)
from .io import cmd_chain_builder, cmd_prepare_data


CONFIGS = {
    'arc_d_shape': ARC_D_SHAPE,
    'spherical': SPHERICAL,
}

def cmd_plot_geometry(args):
    """Plot geometry for a given element."""
    print(f"--- Plotting Geometry for {args.config} ---")
    if args.config not in CONFIGS:
        print(f"Error: Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
        sys.exit(1)
    
    config = CONFIGS[args.config]
    model = create_model(config)
    plot_model(model, output_dir=args.output_dir)
    print(f"Geometry plot saved to {args.output_dir}")


def cmd_build_library(args):
    """Build the element library."""
    print(f"--- Building Element Library (Workflow: {args.workflow}, Config: {args.config}) ---")
    config = CONFIGS.get(args.config)
    if not config:
        print(f"Error: Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
        sys.exit(1)
    
    elements_to_run = [args.element] if args.element else None
    
    build_library(
        elements=elements_to_run,
        config=config, 
        workflow=args.workflow, 
        power=args.power, 
        printlib_file=args.printlib_file, 
        verbose=args.verbose, 
        cross_sections=args.cross_sections, 
        chain_file=args.chain_file,
        use_reduced_chain=not args.no_reduce_chain,
        debug_particles=args.debug_particles
    )
    print("Library building complete!")


def cmd_plot_dose(args):
    """Plot dose rate vs time for all elements."""
    print("--- Plotting Dose Rate vs. Time ---")
    plot_dose_rate_vs_time(results_dir=args.results_dir)
    print("Dose rate plot complete!")


def cmd_build_manifold(args):
    """Build the alloy performance manifold."""
    print(f"--- Building Manifold ({args.samples} samples) ---")
    df = build_manifold(n=args.samples, workers=args.workers)
    print(f"Manifold with {len(df)} samples saved to manifold.parquet")


def cmd_compare_dose(args):
    """Compare FISPACT and OpenMC dose rate results."""
    print("--- Comparing Dose Rate Results ---")
    plot_fispact_comparison(
        fispact_output=args.fispact_output,
        path_a_results=args.path_a_results,
        path_b_results=args.path_b_results,
        output_dir=args.output_dir
    )

def cmd_plot_flux(args):
    """Plot FISPACT flux spectrum."""
    print(f"--- Plotting FISPACT Flux Spectrum: {args.flux_file} ---")
    from pathlib import Path
    
    flux_file = Path(args.flux_file)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        plot_path = plot_fispact_flux(
            flux_file=flux_file,
            output_dir=output_dir,
            log_scale=args.log_scale,
            show_plot=args.show
        )
        print(f"✅ Flux spectrum plot saved to: {plot_path}")
    except Exception as e:
        print(f"❌ Error plotting flux spectrum: {e}")
        sys.exit(1)

def cmd_full_workflow(args):
    """Run the complete workflow."""
    print("=== Full Neutronics-CALPHAD Workflow ===")
    
    # 0. Select config
    config = CONFIGS.get(args.config)
    if not config:
        print(f"Error: Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    # 1. Plot geometry
    print(f"\n--- Step 1: Plotting Geometry ({args.config}) ---")
    model = create_model(config)
    plot_model(model, output_dir=args.output_dir)
    
    # 2. Build library
    print("\n--- Step 2: Building Element Library ---")
    build_library(config=config, workflow=args.workflow, power=args.power, printlib_file=args.printlib_file, verbose=args.verbose, cross_sections=args.cross_sections, chain_file=args.chain_file)
    
    # 3. Plot results
    print("\n--- Step 3: Plotting Dose Rate vs. Time ---")
    plot_dose_rate_vs_time()
    
    # 4. Build manifold (optional)
    if args.build_manifold:
        print("\n--- Step 4: Building Alloy Manifold ---")
        build_manifold(n=args.manifold_samples, workers=args.workers)
    
    print("\n=== Workflow Complete! ===")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neutronics-calphad",
        description="Neutronics simulations for CALPHAD-based alloy optimization"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Plot geometry command
    plot_parser = subparsers.add_parser("plot-geometry", help="Plot tokamak geometry")
    plot_parser.add_argument("config", help=f"The geometry configuration to plot. Available: {list(CONFIGS.keys())}")
    plot_parser.add_argument("-o", "--output-dir", default="results/plot-geometry", 
                           help="Output directory for plots")
    plot_parser.set_defaults(func=cmd_plot_geometry)
    
    # Build library command
    lib_parser = subparsers.add_parser("build-library", help="Build element library")
    lib_parser.add_argument(
        "--config",
        choices=CONFIGS.keys(),
        default='arc_d_shape',
        help="The geometry and material configuration to use."
    )
    lib_parser.add_argument(
        "--workflow", 
        choices=['r2s', 'fispact_path_a', 'fispact_path_b'],
        default='r2s',
        help="The calculation workflow to use."
    )
    lib_parser.add_argument(
        "--power",
        type=float,
        default=500e6,
        help="Fusion power in Watts for normalization."
    )
    lib_parser.add_argument(
        "--printlib-file",
        default=None,
        help="Path to the FISPACT printlib file for 'fispact_path_b' workflow."
    )
    lib_parser.add_argument("--cross-sections", default=None, help="Path to cross_sections.xml file.")
    lib_parser.add_argument("--chain-file", default=None, help="Path to depletion chain XML file.")
    lib_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output during simulations.")
    lib_parser.add_argument("--no-reduce-chain", action="store_true", help="Disable depletion chain reduction.")
    lib_parser.add_argument("--element", type=str, default=None, help="Specify a single element to run for debugging.")
    lib_parser.add_argument("--debug-particles", type=int, default=None, help="Override particle count for quick debugging.")
    lib_parser.set_defaults(func=cmd_build_library)
    
    # Plot dose command
    dose_parser = subparsers.add_parser("plot-dose", help="Plot dose rate vs time")
    dose_parser.add_argument("-r", "--results-dir", default="elem_lib",
                           help="Results directory containing element data")
    dose_parser.set_defaults(func=cmd_plot_dose)
    
    # Compare dose command
    compare_parser = subparsers.add_parser("compare-dose", help="Compare FISPACT and OpenMC dose rates.")
    compare_parser.add_argument("--fispact-output", required=True, help="Path to FISPACT .lis output file.")
    compare_parser.add_argument("--path-a-results", required=True, help="Path to Path A HDF5 results file.")
    compare_parser.add_argument("--path-b-results", required=True, help="Path to Path B HDF5 results file.")
    compare_parser.add_argument("-o", "--output-dir", default="results", help="Output directory for the plot.")
    compare_parser.set_defaults(func=cmd_compare_dose)

    # Plot flux command
    flux_parser = subparsers.add_parser("plot-flux", help="Plot FISPACT flux spectrum from flux file.")
    flux_parser.add_argument("flux_file", help="Path to FISPACT flux file (fispact_flux.txt)")
    flux_parser.add_argument("-o", "--output-dir", default=None, 
                           help="Output directory for the plot (default: same as flux file)")
    flux_parser.add_argument("--linear-scale", dest="log_scale", action="store_false", 
                           help="Use linear scale instead of log scale")
    flux_parser.add_argument("--show", action="store_true", 
                           help="Display plot interactively")
    flux_parser.set_defaults(func=cmd_plot_flux)

    # Build manifold command
    manifold_parser = subparsers.add_parser("build-manifold", help="Build alloy manifold")
    manifold_parser.add_argument("-n", "--samples", type=int, default=15000,
                               help="Number of alloy samples")
    manifold_parser.add_argument("-w", "--workers", type=int, default=8,
                               help="Number of parallel workers")
    manifold_parser.set_defaults(func=cmd_build_manifold)
    
    # Chain builder command
    chain_parser = subparsers.add_parser(
        "chain-builder", 
        help="Create a depletion chain from ENDF files.",
        description="Create an OpenMC depletion chain from TENDL, FISPACT, and GEFY data."
    )
    chain_parser.add_argument("--neutron-dir", required=True, help="Path to directory with TENDL neutron files.")
    chain_parser.add_argument("--decay-dir", required=True, help="Path to directory with FISPACT decay files.")
    chain_parser.add_argument("--fpy-dir", required=True, help="Path to directory with GEFY FPY files.")
    chain_parser.add_argument("--output-file", default="chain.xml", help="Path to write the output chain file.")
    chain_parser.set_defaults(func=cmd_chain_builder)
    
    # Data preparation command
    data_parser = subparsers.add_parser('prepare-data', help='Download and process necessary data files.')
    data_parser.set_defaults(func=cmd_prepare_data)

    # Full workflow command
    workflow_parser = subparsers.add_parser("run", help="Run complete workflow")
    workflow_parser.add_argument(
        "--config",
        choices=CONFIGS.keys(),
        default='arc_d_shape',
        help="The geometry and material configuration to use."
    )
    workflow_parser.add_argument("-o", "--output-dir", default="results",
                                help="Output directory for plots")
    workflow_parser.add_argument(
        "--workflow", 
        choices=['r2s', 'fispact_path_a', 'fispact_path_b'],
        default='r2s',
        help="The calculation workflow to use for the library build step."
    )
    workflow_parser.add_argument(
        "--power",
        type=float,
        default=500e6,
        help="Fusion power in Watts for normalization."
    )
    workflow_parser.add_argument(
        "--printlib-file",
        default=None,
        help="Path to the FISPACT printlib file for 'fispact_path_b' workflow."
    )
    workflow_parser.add_argument("--cross-sections", default=None, help="Path to cross_sections.xml file.")
    workflow_parser.add_argument("--chain-file", default=None, help="Path to depletion chain XML file.")
    workflow_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output during simulations.")
    workflow_parser.add_argument("--build-manifold", action="store_true",
                                help="Also build alloy manifold")
    workflow_parser.add_argument("--manifold-samples", type=int, default=15000,
                                help="Number of manifold samples")
    workflow_parser.add_argument("-w", "--workers", type=int, default=8,
                                help="Number of parallel workers")
    workflow_parser.set_defaults(func=cmd_full_workflow)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # In Python 3.7+, the 'dest' argument to add_subparsers is required.
    # We check if 'func' is present, as it's set by set_defaults.
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no subcommand was specified, and we are not handling it above,
        # it's good practice to show help.
        parser.print_help()


if __name__ == "__main__":
    main() 