#!/usr/bin/env python3
"""
Test runner script for neutronics-calphad package.

This script provides a convenient way to run different test suites
with proper configuration and error handling.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for neutronics-calphad package"
    )
    parser.add_argument(
        "--suite", 
        choices=["all", "unit", "integration", "calphad", "fast"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=neutronics_calphad", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection based on suite
    if args.suite == "unit":
        cmd.extend(["-m", "unit"])
    elif args.suite == "integration": 
        cmd.extend(["-m", "integration"])
    elif args.suite == "calphad":
        cmd.extend(["tests/test_depletion_result.py", 
                   "tests/test_activation_manifold.py",
                   "tests/test_calphad_batch.py", 
                   "tests/test_bayesian_searcher.py",
                   "tests/test_outlier_detector.py"])
    elif args.suite == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.file:
        cmd.append(args.file)
    else:
        # Run all tests
        cmd.append("tests/")
    
    # Check if we're in the right directory
    if not Path("neutronics_calphad").exists():
        print("❌ Error: neutronics_calphad package not found")
        print("Please run this script from the project root directory")
        return 1
    
    # Run the tests
    success = run_command(cmd, f"Test suite: {args.suite}")
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/")
        print(f"Open htmlcov/index.html in your browser to view the report")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 