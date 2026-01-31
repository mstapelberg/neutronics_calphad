# Geometry Sensitivity Analysis: First Wall & Vessel Thickness Sweep

## Overview

`geometry_sweep_neutronics_run.py` performs a systematic sensitivity analysis to assess how first wall and vessel layer thicknesses affect neutron flux spectra and subsequent depletion predictions.

## Sweep Parameters

### First Wall (Pure W)
- **Thicknesses**: 1, 2, 5, 10 mm (0.1, 0.2, 0.5, 1.0 cm)
- **Material**: Pure tungsten (W)

### Vessel (V-4Cr-4Ti)
- **Thicknesses**: 0.5, 1, 2, 5 cm  
- **Material**: V-92%, Cr-4%, Ti-4%

### Reference Geometry
- **First wall**: 2 mm (0.2 cm)
- **Vessel**: 1 cm
- All deviations computed relative to this baseline

## Total Runs
- **16 geometry combinations** (4 × 4 independent sweep)
- Each geometry requires flux/microXS calculation (~expensive!)
- Depletion run for V-4Cr-4Ti in each geometry

## Output Structure

```
analysis_results/geometry_sweep_neutronics_run/
├── fw_0.1cm_vessel_0.5cm/
│   ├── microxs_and_flux/
│   │   ├── flux_spectrum_1102.txt
│   │   └── microxs_1102.csv
│   └── depletion_results/
│       └── V-4Cr-4Ti/
│           └── depletion_results.h5
├── fw_0.1cm_vessel_1.0cm/
│   └── ...
├── ... (16 total geometry directories)
├── analysis_plots/
│   ├── flux_spectrum_geometry_sweep.png
│   ├── activity_sweep_V-4Cr-4Ti.png
│   └── geometry_sensitivity_heatmap.png
└── metrics/
    ├── relative_deviation_metrics.json
    ├── relative_deviation_metrics.csv
    └── (future: recommendation file)
```

## Key Outputs

### 1. Flux Spectrum Comparison
**File**: `analysis_plots/flux_spectrum_geometry_sweep.png`

- Overlay of all 16 flux spectra (1102-group UKAEA structure)
- Shows how first wall and vessel thickness affect neutron energy distribution
- **What to look for**:
  - Significant flux spectrum shape changes indicate geometry-sensitive energy spectrum
  - Magnitude changes affect reaction rates directly

### 2. Activity Time Series
**File**: `analysis_plots/activity_sweep_V-4Cr-4Ti.png`

- Total activity (Bq/kg) vs. cooling time for all geometries
- Shows sensitivity of depletion predictions to geometry
- **What to look for**:
  - Divergence at specific cooling times (e.g., 30d, 1y, 5y, 100y)
  - Early cooling (hours-days): short-lived isotopes
  - Late cooling (years-decades): long-lived isotopes

### 3. Sensitivity Heatmaps
**File**: `analysis_plots/geometry_sensitivity_heatmap.png`

Two side-by-side heatmaps:

#### Left: Flux Spectrum Sensitivity
- Relative L2 deviation: `||flux_test - flux_ref|| / ||flux_ref||`
- Shows which geometries produce significantly different flux spectra
- **Green**: Low deviation (< 2%)
- **Yellow**: Moderate deviation (2-5%)  
- **Red**: High deviation (> 5%)

#### Right: Activity Sensitivity
- Mean relative L2 deviation across cooling time series
- Shows which geometries produce different activation predictions
- Same color scale as flux

### 4. Quantitative Metrics

**Files**: 
- `metrics/relative_deviation_metrics.json` (structured data)
- `metrics/relative_deviation_metrics.csv` (spreadsheet-friendly)

Columns:
- `first_wall_cm`: First wall thickness
- `vessel_cm`: Vessel thickness  
- `metric`: Either 'flux' or 'activity_total'
- `material`: Material name (empty for flux)
- `relative_deviation`: Relative L2 norm vs. reference

## Interpretation Guide

### Flux Spectrum Sensitivity

**Physical Meaning**: 
- First wall acts as a neutron moderator/absorber
- Thicker first wall → more moderation → softer spectrum
- Vessel thickness affects neutron attenuation entering vessel material

**Critical Question**: Does geometry change flux spectrum *shape* or just *magnitude*?
- Shape changes → energy-dependent reaction rates affected → non-linear depletion effects
- Magnitude only → scales linearly with flux (less concerning)

**Recommended Analysis**:
```python
# Load two flux spectra
flux_ref = np.loadtxt('fw_0.2cm_vessel_1.0cm/microxs_and_flux/flux_spectrum_1102.txt')
flux_test = np.loadtxt('fw_1.0cm_vessel_5.0cm/microxs_and_flux/flux_spectrum_1102.txt')

# Normalize to check shape vs. magnitude
flux_ref_norm = flux_ref[:, 1] / np.sum(flux_ref[:, 1])
flux_test_norm = flux_test[:, 1] / np.sum(flux_test[:, 1])

# Shape deviation (should be small if only magnitude changes)
shape_dev = np.linalg.norm(flux_test_norm - flux_ref_norm) / np.linalg.norm(flux_ref_norm)
print(f"Shape deviation: {shape_dev:.4f}")
```

### Activity Sensitivity

**Physical Meaning**:
- Different flux spectra → different reaction rates → different isotope inventories
- Sensitive reactions: (n,γ), (n,2n), (n,p), (n,α)
- Some nuclides highly sensitive to thermal vs. fast neutrons

**Critical Question**: Does geometry affect dose/gas limits compliance?

**Production Code Limits**:
```python
DOSE_LIMITS = {
    30: 1e3,      # 30 days:  1000 Sv/h
    365: 1.0,     # 1 year:   1 Sv/h
    1825: 0.01,   # 5 years:  0.01 Sv/h
    36500: 0.0001 # 100 years: 0.0001 Sv/h
}

GAS_LIMITS = {
    'He_2y': 396,    # appm (2-year irradiation)
    'H_2y': 1200     # appm
}
```

### Decision Criteria

#### ✅ Geometry is NOT critical if:
- Flux relative deviation < 2% for all geometries
- Activity relative deviation < 2% for all geometries
- Pass/fail status unchanged for production limits

#### ⚠️ Geometry requires attention if:
- Flux deviation 2-5% (moderate sensitivity)
- Activity deviation 2-5%
- Some geometries near pass/fail boundary

#### 🚨 Geometry is CRITICAL if:
- Flux deviation > 5% (strong sensitivity)
- Activity deviation > 5%
- Pass/fail status changes with geometry
- **Action**: Use detailed geometry in production runs OR bound uncertainty

## Running the Script

### Basic Run (Full 16 geometries)
```bash
cd /home/myless/Packages/neutronics_calphad/examples
python geometry_sweep_neutronics_run.py
```

**Expected Runtime**: 
- ~30-45 min per geometry (flux + depletion)
- **Total**: ~8-12 hours for 16 geometries (with caching)

### Analysis-Only Mode (Re-plot existing data)
```python
# In geometry_sweep_neutronics_run.py, set:
ANALYZE_ONLY = True
```
Then run again. This skips expensive simulations and regenerates plots from cached results.

### Subset Testing
To test with fewer geometries during development:
```python
# In geometry_sweep_neutronics_run.py, modify:
FIRST_WALL_THICKNESSES_CM = [0.2, 1.0]  # Just 2 values
VESSEL_THICKNESSES_CM = [1.0, 5.0]      # Just 2 values
# Now only 2×2 = 4 runs (~2 hours)
```

## Next Steps

### 1. Review Sensitivity Results
- Check heatmaps for concerning red regions
- Identify which geometry parameter (first wall vs. vessel) dominates sensitivity

### 2. Extend to More Materials
If sensitivity is significant, test other vessel candidates:
```python
test_materials = [
    {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04},  # V-4Cr-4Ti (baseline)
    {'V': 0.90, 'Cr': 0.05, 'Ti': 0.05},  # V-5Cr-5Ti
    {'V': 1.0},                            # Pure V
]
```

### 3. Assess Production Impact
- If deviations < 2%: Current geometry sufficient
- If deviations 2-5%: Add geometry uncertainty to error budget
- If deviations > 5%: Consider parametric geometry in production workflow

### 4. Optimize Geometry Representation
Based on results, you might:
- Fix non-sensitive dimension (e.g., if first wall insensitive, always use 2mm)
- Bound sensitive dimension (e.g., vessel must be 1-2 cm for <2% uncertainty)
- Include geometry as a design variable in optimization

## Technical Notes

### Flux Calculation Method
- Uses `get_flux_and_microxs()` from `neutronics.flux` module
- UKAEA-1102 group structure (consistent with production)
- Flux tallied in vessel material region
- MicroXS collapsed for depletion chain

### Depletion Settings
- **Irradiation**: 2 years, 24 substeps
- **Cooling times**: 1s → 100 years (14 points)
- **Source rate**: 500 MW fusion power, spherical scaling
- **Chain**: TENDL-2021 + FISPACT-2020 + GEFY-6.1

### Relative L2 Metric
$$
\text{Relative Deviation} = \frac{||\vec{x}_{\text{test}} - \vec{x}_{\text{ref}}||_2}{||\vec{x}_{\text{ref}}||_2}
$$

- Flux: Computed over 1102 energy groups
- Activity: Computed over cooling time series
- Dimensionless metric: 0.02 = 2% deviation

### Caching Behavior
The script caches flux and depletion results. To force recomputation:
```bash
# Delete specific geometry
rm -rf analysis_results/geometry_sweep_neutronics_run/fw_0.2cm_vessel_1.0cm/

# Delete all
rm -rf analysis_results/geometry_sweep_neutronics_run/
```

## Troubleshooting

### Issue: "Flux not found and ANALYZE_ONLY is True"
**Solution**: Set `ANALYZE_ONLY = False` or run full simulation first

### Issue: Long runtime
**Solution**: 
- Reduce number of test points
- Use cached results with `ANALYZE_ONLY = True`
- Parallelize across nodes (manual - no built-in parallelism yet)

### Issue: High memory usage
**Solution**: 
- Reduce `FIXED_PARTICLES` (default 10,000)
- Process geometries sequentially (already default behavior)
- Check for memory leaks in OpenMC (rare but possible)

### Issue: Unexpected high sensitivity
**Possible Causes**:
- Statistical noise in flux calculation → increase particles/batches
- Geometry bug → check that layers defined correctly
- Material composition mismatch → verify V-4Cr-4Ti composition

## References

- Geometry definition: `neutronics_calphad/neutronics/config.py`
- Flux calculation: `neutronics_calphad/neutronics/flux.py`
- Depletion: `neutronics_calphad/neutronics/depletion.py`
- Similar analysis: `examples/particle_sweep_neutronics_run.py`

