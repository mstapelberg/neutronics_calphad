#!/usr/bin/env python3
"""
Command-line interface for neutronics-calphad package.
"""

import argparse
import sys
from pathlib import Path

from . import (
    create_model, plot_model, build_library, 
    plot_dose_rate_vs_time, build_manifold
)


def cmd_plot_geometry(args):
    """Plot geometry for a given element."""
    print(f"--- Plotting Geometry for {args.element} ---")
    model = create_model(args.element)
    plot_model(model, output_dir=args.output_dir)
    print(f"Geometry plot saved to {args.output_dir}")


def cmd_build_library(args):
    """Build the element library."""
    print("--- Building Element Library ---")
    build_library()
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


def cmd_full_workflow(args):
    """Run the complete workflow."""
    print("=== Full Neutronics-CALPHAD Workflow ===")
    
    # 1. Plot geometry
    print("\n--- Step 1: Plotting Geometry ---")
    model = create_model('V')  # Representative element
    plot_model(model, output_dir=args.output_dir)
    
    # 2. Build library
    print("\n--- Step 2: Building Element Library ---")
    build_library()
    
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
    plot_parser.add_argument("element", help="Element symbol (e.g., V, Cr, Ti)")
    plot_parser.add_argument("-o", "--output-dir", default=".", 
                           help="Output directory for plots")
    plot_parser.set_defaults(func=cmd_plot_geometry)
    
    # Build library command
    lib_parser = subparsers.add_parser("build-library", help="Build element library")
    lib_parser.set_defaults(func=cmd_build_library)
    
    # Plot dose command
    dose_parser = subparsers.add_parser("plot-dose", help="Plot dose rate vs time")
    dose_parser.add_argument("-r", "--results-dir", default="elem_lib",
                           help="Results directory containing element data")
    dose_parser.set_defaults(func=cmd_plot_dose)
    
    # Build manifold command
    manifold_parser = subparsers.add_parser("build-manifold", help="Build alloy manifold")
    manifold_parser.add_argument("-n", "--samples", type=int, default=15000,
                               help="Number of alloy samples")
    manifold_parser.add_argument("-w", "--workers", type=int, default=8,
                               help="Number of parallel workers")
    manifold_parser.set_defaults(func=cmd_build_manifold)
    
    # Full workflow command
    workflow_parser = subparsers.add_parser("run", help="Run complete workflow")
    workflow_parser.add_argument("-o", "--output-dir", default="results",
                                help="Output directory for plots")
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
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 