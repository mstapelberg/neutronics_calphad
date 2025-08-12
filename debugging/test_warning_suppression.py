#!/usr/bin/env python3
"""
Test script to verify that OpenMC warning suppression is working.
"""

import sys
import os
import warnings
import io
import contextlib
from pathlib import Path

# Set up warning suppression BEFORE importing OpenMC
class OpenMCWarningFilter:
    """Filter to suppress OpenMC warnings that are printed to stdout/stderr."""
    
    def __init__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.buffer = ""
    
    def write(self, text: str) -> None:
        # Add text to buffer
        self.buffer += text
        
        # Check if we have a complete line (ends with newline)
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Process all complete lines except the last one (which might be incomplete)
            for line in lines[:-1]:
                if not self._should_suppress(line):
                    self.original_stdout.write(line + '\n')
            
            # Keep the last part (might be incomplete line)
            self.buffer = lines[-1]
    
    def _should_suppress(self, line: str) -> bool:
        """Check if a line should be suppressed."""
        line_lower = line.lower()
        return (
            ("ltt" in line_lower and "elastic scattering" in line_lower and "legendre only" in line_lower) or
            "gnds naming convention" in line_lower or
            "cross_sections" in line_lower
        )
    
    def flush(self) -> None:
        # Write any remaining buffer content
        if self.buffer and not self._should_suppress(self.buffer):
            self.original_stdout.write(self.buffer)
        self.original_stdout.flush()

# Apply warning filter globally
stdout_filter = OpenMCWarningFilter()
stderr_filter = OpenMCWarningFilter()
sys.stdout = stdout_filter
sys.stderr = stderr_filter

# Add the parent directory to the path so we can import neutronics_calphad
sys.path.insert(0, str(Path(__file__).parent.parent))

from neutronics_calphad.utils.utils import filter_openmc_warnings, suppress_openmc_warnings
import openmc

def test_warning_suppression():
    """Test that OpenMC warnings are properly suppressed."""
    
    print("🧪 Testing OpenMC Warning Suppression")
    print("=" * 50)
    
    # Set up OpenMC configuration
    chain_file = '/Users/myless/nuclear_data/tendl2021_fispact2020_chain.xml'
    cross_sections = '/Users/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'
    
    if not os.path.exists(chain_file):
        print(f"❌ Chain file not found: {chain_file}")
        return
    
    if not os.path.exists(cross_sections):
        print(f"❌ Cross sections not found: {cross_sections}")
        return
    
    openmc.config['chain_file'] = chain_file
    openmc.config['cross_sections'] = cross_sections
    
    print("✅ OpenMC configuration set up")
    
    # Test 1: Without warning suppression
    print("\n📋 Test 1: Loading chain WITHOUT warning suppression")
    print("-" * 50)
    try:
        chain = openmc.deplete.Chain.from_xml(chain_file)
        print("✅ Chain loaded successfully")
    except Exception as e:
        print(f"❌ Error loading chain: {e}")
    
    # Test 2: With warning suppression
    print("\n📋 Test 2: Loading chain WITH warning suppression")
    print("-" * 50)
    try:
        with suppress_openmc_warnings():
            chain = openmc.deplete.Chain.from_xml(chain_file)
        print("✅ Chain loaded successfully with warning suppression")
    except Exception as e:
        print(f"❌ Error loading chain: {e}")
    
    # Test 3: Global warning filter
    print("\n📋 Test 3: Testing global warning filter")
    print("-" * 50)
    filter_openmc_warnings()
    try:
        chain = openmc.deplete.Chain.from_xml(chain_file)
        print("✅ Chain loaded successfully with global filter")
    except Exception as e:
        print(f"❌ Error loading chain: {e}")
    
    print("\n🎉 Warning suppression test completed!")

if __name__ == "__main__":
    test_warning_suppression() 