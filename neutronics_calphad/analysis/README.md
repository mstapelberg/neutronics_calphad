# Analysis Tools for Sequential Materials Pipeline

This directory contains comprehensive analysis and visualization tools for the sequential materials design pipeline results, including UMAP dimensionality reduction for composition space exploration.

## Overview

The analysis tools have been updated to handle the new Gaussian process-based approach (instead of Bayesian optimization) and provide advanced visualization capabilities including:

- **UMAP dimensionality reduction** for composition space exploration
- **Interactive plots** using Plotly
- **Comprehensive pipeline analysis** across neutronics and CALPHAD stages
- **Performance metrics** and convergence analysis
- **Phase stability analysis** for CALPHAD results

## Key Features

### 1. UMAP Visualization
- **Dimensionality reduction** of high-dimensional composition space to 2D
- **Multiple coloring schemes**: constraint satisfaction, performance metrics, clustering
- **Interactive plots** with hover information
- **Static plots** for publications

### 2. Pipeline Analysis
- **Stage-by-stage analysis** of neutronics and CALPHAD results
- **Success rate tracking** across iterations
- **Performance metrics** evolution
- **Composition space coverage** analysis

### 3. CALPHAD Integration
- **Phase stability analysis** of neutronics-passing compositions
- **Phase distribution** and dominant phase identification
- **Violation analysis** for phase constraints
- **Pipeline efficiency** metrics

## Installation

Install the additional dependencies for enhanced analysis:

```bash
pip install umap-learn scikit-learn seaborn plotly
```

## Usage

### Quick Analysis

For a quick overview of your pipeline results:

```python
from neutronics_calphad.analysis.bo_results_analyzer import quick_analysis

# Basic analysis
quick_analysis("sequential_materials_pipeline_run_1")

# With interactive plots
quick_analysis("sequential_materials_pipeline_run_1", interactive=True)

# Without UMAP (if dependencies not available)
quick_analysis("sequential_materials_pipeline_run_1", create_umap=False)
```

### Command Line Usage

```bash
# Basic analysis
python neutronics_calphad/analysis/bo_results_analyzer.py sequential_materials_pipeline_run_1

# With interactive plots
python neutronics_calphad/analysis/bo_results_analyzer.py sequential_materials_pipeline_run_1 --interactive

# Without UMAP
python neutronics_calphad/analysis/bo_results_analyzer.py sequential_materials_pipeline_run_1 --no-umap
```

### Example Script

Use the provided example script for detailed analysis:

```bash
# Basic analysis
python examples/analyze_pipeline_results.py sequential_materials_pipeline_run_1

# Detailed analysis with interactive plots
python examples/analyze_pipeline_results.py sequential_materials_pipeline_run_1 --interactive --detailed
```

## Output Files

The analysis generates several output files in your results directory:

### Plots
- `optimization_progress.png` - Iteration progress and success rates
- `composition_heatmap.png` - Correlation matrix between composition and performance
- `umap_visualization.png` - Static UMAP plot of composition space
- `umap_interactive.html` - Interactive UMAP plot (if --interactive used)

### Reports
- `analysis_report.json` - Comprehensive analysis report
- `materials_data.csv` - Extracted data for further analysis

## UMAP Visualization Details

### What UMAP Shows

UMAP (Uniform Manifold Approximation and Projection) reduces the high-dimensional composition space (5 elements = 5D) to 2D for visualization while preserving local and global structure.

### Coloring Schemes

1. **Constraint Satisfaction**: 
   - Green: Both dose and gas constraints satisfied
   - Orange: Only dose constraint satisfied
   - Blue: Only gas constraint satisfied
   - Red: Neither constraint satisfied
   - Purple: Predicted good compositions (squares)

2. **Performance Metrics**:
   - Dose rate (14-day)
   - He production (appm)
   - H production (appm)

3. **Clustering**:
   - K-means clusters (5 clusters by default)
   - Shows natural groupings in composition space

### Interactive Features

When using `--interactive`, you get:
- **Hover information**: Composition details and performance metrics
- **Zoom and pan**: Explore specific regions
- **Legend**: Toggle different data series
- **Export**: Save as HTML for sharing

## Analysis Functions

### Core Functions

```python
# Load pipeline results
results = load_pipeline_results("results_directory")

# Extract optimization history
df = extract_optimization_history(results['neutronics'])

# Extract good compositions from surrogate
good_comps = extract_good_compositions(results['neutronics'])

# Create UMAP visualization
create_umap_visualization(df, good_comps, elements, interactive=True)
```

### Analysis Functions

```python
# Convergence analysis
convergence = analyze_convergence(df)

# Composition space analysis
comp_analysis = analyze_composition_space(df)

# CALPHAD analysis
calphad_analysis = analyze_calphad_results(results['calphad'])

# Generate comprehensive report
report = generate_analysis_report("results_directory")
```

## Understanding the Results

### Success Rates

- **Dose success rate**: Fraction of compositions satisfying dose constraints
- **Gas success rate**: Fraction of compositions satisfying gas constraints
- **Combined success rate**: Fraction satisfying both constraints
- **Phase success rate**: Fraction of neutronics-passing compositions that are phase-stable

### Performance Metrics

- **14-day dose rate**: Contact dose rate after 2 weeks cooling
- **He production**: Helium production in atomic parts per million
- **H production**: Hydrogen production in atomic parts per million

### Composition Space

- **Coverage**: How well the composition space has been explored
- **Successful regions**: Areas where compositions satisfy constraints
- **Clustering**: Natural groupings in the composition space

## Troubleshooting

### UMAP Not Available

If you get "UMAP not available" warnings:

```bash
pip install umap-learn scikit-learn
```

### Plotly Not Available

If interactive plots don't work:

```bash
pip install plotly
```

### Memory Issues

For large datasets, you can:
- Reduce the number of UMAP neighbors: `n_neighbors=10`
- Use fewer clusters: `n_clusters=3`
- Process data in batches

### Performance Issues

- Use `create_umap=False` for faster analysis
- Reduce the number of compositions analyzed
- Use static plots instead of interactive ones

## Advanced Usage

### Custom UMAP Parameters

```python
# Custom UMAP visualization
from neutronics_calphad.analysis.bo_results_analyzer import create_umap_visualization

# You can modify the UMAP parameters in the function
# Default: n_neighbors=15, min_dist=0.1, metric='euclidean'
```

### Custom Analysis

```python
# Load and analyze specific aspects
results = load_pipeline_results("results_directory")
df = extract_optimization_history(results['neutronics'])

# Custom filtering
successful_comps = df[df['satisfy_dose'] & df['satisfy_gas']]
print(f"Successful compositions: {len(successful_comps)}")

# Custom plotting
import matplotlib.pyplot as plt
plt.scatter(successful_comps['comp_V'], successful_comps['comp_Cr'])
plt.xlabel('V fraction')
plt.ylabel('Cr fraction')
plt.title('Successful Compositions')
plt.show()
```

## Contributing

To add new analysis features:

1. Add new functions to `bo_results_analyzer.py`
2. Include proper type hints and docstrings
3. Add error handling for missing data
4. Update the example script if needed
5. Add tests for new functionality

## References

- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- Plotly: https://plotly.com/python/
- Scikit-learn: https://scikit-learn.org/ 