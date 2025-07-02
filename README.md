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
# Plot geometry for vanadium
neutronics-calphad plot-geometry V -o results/

# Build the complete element library (this takes time!)
neutronics-calphad build-library

# Plot dose rate results
neutronics-calphad plot-dose

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
model = nc.create_model('V')
nc.plot_model(model, output_dir='results/')

# Build element library
nc.build_library()

# Evaluate an alloy composition
composition = [0.5, 0.2, 0.1, 0.1, 0.1]  # V-Cr-Ti-W-Zr
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

## Requirements

- OpenMC
- openmc-plasma-source
- matplotlib
- numpy
- h5py
- pandas
- umap-learn
