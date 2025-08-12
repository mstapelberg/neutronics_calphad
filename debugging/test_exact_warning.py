#!/usr/bin/env python3
"""
Test script to verify exact warning pattern matching.
"""

def test_warning_pattern():
    """Test if our warning pattern matches the actual warning."""
    
    # The actual warning from OpenMC
    actual_warning = "n-001_H_002.endf: Warning, LTT (3) for elastic scattering, using Legendre only"
    
    print(f"Actual warning: {actual_warning}")
    
    # Test our pattern matching
    text_lower = actual_warning.lower()
    
    print(f"Lowercase: {text_lower}")
    print(f"Contains 'ltt': {'ltt' in text_lower}")
    print(f"Contains 'elastic scattering': {'elastic scattering' in text_lower}")
    print(f"Contains 'legendre only': {'legendre only' in text_lower}")
    
    # Test our condition
    condition = ("ltt" in text_lower and "elastic scattering" in text_lower and "legendre only" in text_lower)
    print(f"Condition result: {condition}")
    
    if condition:
        print("✅ Warning would be suppressed!")
    else:
        print("❌ Warning would NOT be suppressed!")

if __name__ == "__main__":
    test_warning_pattern() 