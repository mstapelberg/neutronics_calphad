# Tests for neutronics-calphad

This directory contains comprehensive test suites for the neutronics-calphad package, with special focus on the new CALPHAD module.

## Test Structure

- `conftest.py` - Pytest fixtures and common test data
- `test_depletion_result.py` - Tests for DepletionResult class and mix function
- `test_activation_manifold.py` - Tests for ActivationConstraints and ActivationManifold
- `test_calphad_batch.py` - Tests for CALPHADBatchCalculator
- `test_bayesian_searcher.py` - Tests for BayesianSearcher (requires BoTorch)
- `test_outlier_detector.py` - Tests for OutlierDetector
- `test_integration.py` - Integration tests combining multiple components

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test files
```bash
pytest tests/test_depletion_result.py
pytest tests/test_activation_manifold.py
```

### Run with coverage
```bash
pytest --cov=neutronics_calphad --cov-report=html
```

### Skip slow tests
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

## Dependencies

The tests require the following packages:
- pytest
- numpy
- pandas
- h5py
- scikit-learn
- scipy

Optional dependencies:
- torch (for Bayesian optimization tests)
- botorch (for Bayesian optimization tests)
- tc_python (for Thermo-Calc integration tests)

## Test Coverage

The test suite aims for comprehensive coverage of:
- Core functionality
- Edge cases and error handling
- Integration between components
- HDF5 serialization/deserialization
- Input validation
- Numerical accuracy

## Continuous Integration

Tests are designed to run in CI environments with different Python versions (3.10-3.12) and with/without optional dependencies. 