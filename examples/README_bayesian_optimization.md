# Bayesian Optimization with Result Saving

This directory contains scripts for running Bayesian optimization of material compositions with comprehensive result saving and analysis capabilities.

## Files

- `run_bo_composition_search.py` - Main script for running Bayesian optimization with JSON result saving
- `analyze_bo_results.py` - Analysis script for visualizing and analyzing optimization results
- `README_bayesian_optimization.md` - This documentation file

## Running the Optimization

### Prerequisites

1. Ensure you have the required nuclear data files:
   - TENDL-2021 cross-sections
   - FISPACT-II decay data
   - Update the paths in the script if needed

2. Install required dependencies:
   ```bash
   pip install matplotlib pandas numpy
   ```

### Basic Usage

Run the optimization script:

```bash
python run_bo_composition_search.py
```

This will:
- Run Bayesian optimization for material composition search
- Save all results to `bayesian_optimization_results.json`
- Print a summary of the optimization

### Configuration

You can modify the following parameters in `run_bo_composition_search.py`:

```python
# Optimization parameters
N_ITERATIONS = 2  # Number of optimization iterations
ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']  # Elements to optimize
optimizer = BayesianOptimizer(ELEMENTS, batch_size=3, minimize=False)

# Evaluation criteria
CRIT_LIMITS = {"He_appm": 1000, "H_appm": 1250}  # Gas production limits
DOSE_LIMITS = {14: 1e3, 365: 1e-1, 3650: 1e-2, 36500: 1e-4}  # Dose rate limits

# Simulation parameters
POWER_MW = 500  # Fusion power in MW
```

## Analyzing Results

### Using the Analysis Script

After running the optimization, analyze the results:

```bash
python analyze_bo_results.py bayesian_optimization_results.json
```

This will:
- Print detailed summary statistics
- Generate visualization plots
- Save plots to `analysis_output/` directory
- Export data to CSV for further analysis

### Command Line Options

```bash
python analyze_bo_results.py results.json --output-dir my_analysis --no-plots
```

Options:
- `results_file`: Path to the JSON results file (required)
- `--output-dir`: Directory to save analysis outputs (default: `analysis_output`)
- `--no-plots`: Skip generating plots (useful for quick text-only analysis)

### Generated Outputs

The analysis script creates:

1. **Summary Statistics** (printed to console):
   - Optimization metadata
   - Score statistics
   - Feasibility analysis
   - Best material details

2. **Visualization Plots** (saved as PNG files):
   - `score_evolution.png`: Score progression across iterations
   - `composition_analysis.png`: Element composition vs score correlations
   - `gas_production_analysis.png`: Gas production and dose rate analysis

3. **Data Export**:
   - `materials_data.csv`: All material evaluations in tabular format

## JSON Results Structure

The saved JSON file contains:

```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "elements": ["V", "Cr", "Ti", "W", "Zr"],
    "critical_limits": {"He_appm": 1000, "H_appm": 1250},
    "dose_limits": {14: 1e3, 365: 1e-1, 3650: 1e-2, 36500: 1e-4},
    "n_iterations": 2,
    "batch_size": 3,
    "power_mw": 500
  },
  "iterations": [
    {
      "iteration": 1,
      "materials": [
        {
          "material_name": "V-2Cr-4Ti-3W-1Zr",
          "composition": {"V": 0.9, "Cr": 0.02, "Ti": 0.04, "W": 0.03, "Zr": 0.01},
          "composition_array": [0.9, 0.02, 0.04, 0.03, 0.01],
          "score": 0.75,
          "satisfy_dose": true,
          "satisfy_gas": false,
          "dose_rates": {14: 1e2, 365: 1e-3, 3650: 1e-3, 36500: 1e-5},
          "gas_production": {"He_appm": 800, "H_appm": 600},
          "times_s": [0, 3600, 86400, ...],
          "total_dose": {"0": 0.0, "3600": 1e2, ...},
          "final_irr_time": 31536000,
          "cool_start_time": 31557600,
          "source_rates": [1.77e20, 1.77e20, ..., 0.0, 0.0, ...]
        }
      ]
    }
  ]
}
```

## Key Features

### Comprehensive Data Collection

The optimization script captures:
- **Material compositions**: Atomic fractions for all elements
- **Scores**: Continuous fitness scores (0-1)
- **Dose rates**: Contact dose rates at multiple cooling times
- **Gas production**: Helium and hydrogen production in appm
- **Feasibility**: Whether materials satisfy dose and gas limits
- **Time series**: Complete dose evolution over time
- **Metadata**: Optimization parameters and settings

### Analysis Capabilities

The analysis script provides:
- **Statistical summaries**: Mean, std, min, max scores
- **Feasibility analysis**: Percentage of materials meeting criteria
- **Correlation analysis**: Relationships between composition and performance
- **Visualization**: Multiple plot types for different aspects
- **Data export**: CSV format for further analysis

### Extensibility

The modular design allows easy extension:
- Add new evaluation criteria
- Modify scoring functions
- Include additional material properties
- Customize visualization plots

## Example Workflow

1. **Run optimization**:
   ```bash
   python run_bo_composition_search.py
   ```

2. **Analyze results**:
   ```bash
   python analyze_bo_results.py bayesian_optimization_results.json
   ```

3. **Review outputs**:
   - Check console output for summary
   - Examine plots in `analysis_output/`
   - Use CSV data for custom analysis

4. **Iterate**:
   - Modify parameters in the optimization script
   - Run again with different settings
   - Compare results across different runs

## Troubleshooting

### Common Issues

1. **Nuclear data paths**: Update paths in the script to match your system
2. **Memory usage**: Large numbers of iterations may require significant memory
3. **Plot generation**: Ensure matplotlib is installed for visualization
4. **File permissions**: Ensure write permissions for output directories

### Performance Tips

- Start with small `N_ITERATIONS` for testing
- Use smaller `batch_size` for faster iterations
- Consider using `--no-plots` for quick analysis
- Monitor disk space for large result files

## Advanced Usage

### Custom Scoring

Modify the `evaluate_material_detailed` function to implement custom scoring strategies:

```python
def custom_score_function(dose_at_limit, gas_production_rates, dose_limits, gas_limits):
    # Implement custom scoring logic
    return custom_score
```

### Batch Processing

For multiple optimization runs:

```bash
for i in {1..5}; do
    python run_bo_composition_search.py
    mv bayesian_optimization_results.json results_run_${i}.json
done
```

### Integration with Other Tools

The JSON format is compatible with:
- Jupyter notebooks for interactive analysis
- Database storage for large-scale studies
- Web applications for result sharing
- Machine learning pipelines for model training 