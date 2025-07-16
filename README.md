# Neutronics CALPHAD

Neutronics simulations for CALPHAD-based alloy optimization in tokamak environments.

## Installation

Install the package in development mode:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

After installation, you can use the `neutronics-calphad` command:

```bash
# Plot geometry for the ARC D-shape configuration
neutronics-calphad plot-geometry arc_d_shape -o results/

# Build the complete element library using the default R2S workflow
neutronics-calphad build-library

# Build the library using the FISPACT Path A workflow
neutronics-calphad build-library --workflow fispact_path_a --power 500e6

# Prepare data files (e.g., for FISPACT dose calculation)
neutronics-calphad prepare-data

# Create a depletion chain file from nuclear data libraries
neutronics-calphad chain-builder \
    --neutron-dir /path/to/tendl/neutrons \
    --decay-dir /path/to/fispact/decay \
    --fpy-dir /path/to/gefy/fpy \
    --output-file chain_custom.xml

# Plot dose rate results
neutronics-calphad plot-dose

# Compare dose rates from different workflows
neutronics-calphad compare-dose \
    --fispact-output path/to/fispact.lis \
    --path-a-results path/to/results_path_a.h5 \
    --path-b-results path/to/results_path_b.h5

# Build alloy performance manifold
neutronics-calphad build-manifold -n 5000

# Run complete workflow
neutronics-calphad run --build-manifold
```

### Python API

Use from Python scripts or Jupyter notebooks:

```python
import neutronics_calphad as nc

# Create and plot a model
model = nc.create_model(nc.ARC_D_SHAPE)
nc.plot_model(model, output_dir='results/')

# Build element library
nc.build_library()

# Evaluate an alloy composition
composition = [0.8, 0.05, 0.05, 0.05, 0.05]  # V-Cr-Ti-W-Zr
result = nc.evaluate(composition)

# Sample and evaluate many compositions
nc.build_manifold(n=1000)
```

## Package Structure

- `neutronics_calphad.geometry_maker` - Tokamak geometry and OpenMC model creation
- `neutronics_calphad.library` - Element library building and R2S simulations  
- `neutronics_calphad.evaluate` - Single alloy composition evaluation
- `neutronics_calphad.manifold` - Design space sampling and evaluation
- `neutronics_calphad.visualization` - Plotting and visualization tools
- `neutronics_calphad.io` - Nuclear data I/O, data preparation, and depletion chain generation
- `neutronics_calphad.calphad` - CALPHAD coupling, batch calculations, and optimization

## Requirements

- OpenMC
- openmc-plasma-source
- matplotlib
- numpy
- h5py
- pandas
- umap-learn
