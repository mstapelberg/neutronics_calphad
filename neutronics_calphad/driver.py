# driver.py
from .geometry_maker import create_model, plot_model
from .library import build_library
from .visualization import plot_dose_rate_vs_time

def main():
    """Main driver for the neutronics module.

    This script orchestrates the entire simulation and analysis workflow. It
    is designed to be run as the main entry point of the module.

    The workflow consists of three main steps:
    1.  A representative tokamak geometry ('V' element case) is plotted and
        saved to disk to allow for visual inspection.
    2.  The `build_library` function is called to run the R2S (Rigorous 2-Step)
        depletion simulations for every element defined in the library. This is
        the most computationally intensive step.
    3.  After the library is built, the results are collated, and a final
        plot comparing the contact dose rate versus cooling time for all
        elements is generated and saved.
    """
    # 1. Plot the geometry for a representative element
    print("--- Plotting Geometry ---")
    model = create_model('V')
    plot_model(model, output_dir="/home/myless/Packages/neutronics_calphad/results/v_results")

    
    # 2. Build the element library
    print("\n--- Building Element Library ---")
    build_library()
    
    # 3. Plot the final results
    print("\n--- Plotting Dose Rate vs. Time ---")
    plot_dose_rate_vs_time()

if __name__ == "__main__":
    main()
