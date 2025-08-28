# Deprecation Notice

## Linear Impulse Method - Deprecated

The linear impulse response method has been deprecated in favor of a more robust LightGBM-based approach.

### Deprecated Files:
- `impulse_depletion.py` - Linear impulse response library implementation
- `pairwise_fit.py` - Pairwise correction fitting for impulse method
- `high_throughput_pipeline.py` - Pipeline using impulse synthesis

### Migration Guide:

The new approach uses:
1. **LightGBM quantile regression** for uncertainty quantification
2. **ILR (Isometric Log-Ratio) transform** for compositional features
3. **BoTorch acquisition functions** for active learning
4. **Joint probability of feasibility** for constraint satisfaction

### Key Improvements:
- Better uncertainty quantification through quantile regression
- Proper handling of compositional data via ILR transform
- Active learning to efficiently explore the design space
- Direct optimization of feasibility probability

### Migration Steps:
1. Use `lightgbm_workflow.py` as the new entry point
2. Replace `run_simulator` with your actual depletion calculations
3. Set proper limits in `LIMITS` dictionary
4. Configure `CNO_FRAC` for fixed impurities

The old files are preserved here for reference but should not be used for new work.

Last updated: $(date)
