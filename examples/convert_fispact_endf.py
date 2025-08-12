"""
FISPACT ENDF to HDF5 Conversion Script

This script converts FISPACT ENDF .asc files to .h5 format for use with OpenMC.
FISPACT ENDF files may be in a different format than standard ENDF files.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

def examine_file_format(file_path: Path) -> str:
    """Examine the file format to determine how to process it."""
    try:
        with open(file_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        print(f"First few lines of {file_path.name}:")
        for i, line in enumerate(first_lines):
            print(f"  Line {i+1}: {line[:80]}")
        
        # Check for ENDF format indicators
        # Look for ZA format (Z*1000 + A) in the second line
        if len(first_lines) >= 2:
            second_line = first_lines[1]
            # Check if second line contains ZA format (e.g., 24052.0000 for Cr-52)
            if any(char.isdigit() for char in second_line[:10]) and '.' in second_line[:10]:
                return 'endf'
        
        # Also check for ENDF keywords
        if any('ENDF' in line for line in first_lines):
            return 'endf'
        elif any('MAT' in line and 'MF' in line and 'MT' in line for line in first_lines):
            return 'endf'
        else:
            return 'unknown'
    except Exception as e:
        print(f"Error reading file {file_path.name}: {e}")
        return 'unknown'

def convert_fispact_endf_to_h5(endf_dir: str, output_dir: str, nuclides: Optional[List[str]] = None) -> None:
    """
    Convert FISPACT ENDF .asc files to HDF5 format.
    
    Args:
        endf_dir: Directory containing FISPACT ENDF .asc files
        output_dir: Directory to save HDF5 files
        nuclides: List of specific nuclides to convert (if None, convert all)
    """
    try:
        import openmc.data
    except ImportError:
        print("Error: openmc.data not available. Please install OpenMC with data processing capabilities.")
        return
    
    endf_path = Path(endf_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all .asc files
    asc_files = list(endf_path.glob("*.asc"))
    if not asc_files:
        print(f"No .asc files found in {endf_dir}")
        return
    
    print(f"Found {len(asc_files)} .asc files")
    
    # Filter by nuclides if specified
    if nuclides:
        filtered_files = []
        for file in asc_files:
            file_name = file.stem.lower()
            if any(nuc.lower() in file_name for nuc in nuclides):
                filtered_files.append(file)
        asc_files = filtered_files
        print(f"Filtered to {len(asc_files)} files matching nuclides: {nuclides}")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for asc_file in asc_files:
        try:
            print(f"\nConverting {asc_file.name}...")
            
            # Examine file format first
            file_format = examine_file_format(asc_file)
            print(f"  Detected format: {file_format}")
            
            if file_format == 'endf':
                # Try different approaches for ENDF files
                conversion_success = False
                
                # Approach 1: Try direct ENDF reading
                try:
                    print(f"  Trying direct ENDF reading...")
                    data = openmc.data.IncidentNeutron.from_endf(asc_file)
                    h5_file = output_path / f"{data.name}.h5"
                    data.export_to_hdf5(h5_file)
                    print(f"  Successfully converted to {h5_file.name}")
                    conversion_success = True
                except Exception as e:
                    print(f"  Direct ENDF reading failed: {e}")
                
                # Approach 2: Try NJOY processing if direct reading failed
                if not conversion_success:
                    try:
                        print(f"  Trying NJOY processing...")
                        data = openmc.data.IncidentNeutron.from_njoy(
                            asc_file,
                            temperatures=[293.0],
                            evaluation='endf'
                        )
                        h5_file = output_path / f"{data.name}.h5"
                        data.export_to_hdf5(h5_file)
                        print(f"  Successfully converted to {h5_file.name}")
                        conversion_success = True
                    except Exception as e:
                        print(f"  NJOY processing failed: {e}")
                
                # Approach 3: Try with different NJOY parameters
                if not conversion_success:
                    try:
                        print(f"  Trying NJOY with different parameters...")
                        data = openmc.data.IncidentNeutron.from_njoy(
                            asc_file,
                            temperatures=[300.0],
                            evaluation='endf',
                            error=0.001
                        )
                        h5_file = output_path / f"{data.name}.h5"
                        data.export_to_hdf5(h5_file)
                        print(f"  Successfully converted to {h5_file.name}")
                        conversion_success = True
                    except Exception as e:
                        print(f"  NJOY with different parameters failed: {e}")
                
                if conversion_success:
                    successful += 1
                else:
                    print(f"  All conversion methods failed for {asc_file.name}")
                    failed += 1
                    
            else:
                print(f"  Unknown file format, skipping {asc_file.name}")
                failed += 1
            
        except Exception as e:
            print(f"  Failed to convert {asc_file.name}: {e}")
            failed += 1
    
    print(f"\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def create_cross_sections_xml(h5_dir: str, output_file: str) -> None:
    """
    Create cross_sections.xml file from HDF5 files.
    
    Args:
        h5_dir: Directory containing HDF5 files
        output_file: Path to output cross_sections.xml file
    """
    try:
        import openmc.data
    except ImportError:
        print("Error: openmc.data not available.")
        return
    
    h5_path = Path(h5_dir)
    h5_files = list(h5_path.glob("*.h5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {h5_dir}")
        return
    
    print(f"Creating cross_sections.xml from {len(h5_files)} HDF5 files...")
    
    # Create cross sections library
    library = openmc.data.DataLibrary()
    
    for h5_file in h5_files:
        try:
            library.register_file(h5_file)
            print(f"  Added {h5_file.name}")
        except Exception as e:
            print(f"  Failed to add {h5_file.name}: {e}")
    
    # Export to XML
    library.export_to_xml(output_file)
    print(f"Cross sections library saved to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert FISPACT ENDF .asc files to HDF5 format")
    parser.add_argument("endf_dir", help="Directory containing FISPACT ENDF .asc files")
    parser.add_argument("output_dir", help="Directory to save HDF5 files")
    parser.add_argument("--nuclides", nargs="+", help="Specific nuclides to convert")
    parser.add_argument("--create-xml", action="store_true", 
                       help="Create cross_sections.xml file after conversion")
    parser.add_argument("--xml-output", default="cross_sections.xml",
                       help="Output file for cross_sections.xml")
    
    args = parser.parse_args()
    
    # Convert ENDF files
    convert_fispact_endf_to_h5(args.endf_dir, args.output_dir, args.nuclides)
    
    # Create cross_sections.xml if requested
    if args.create_xml:
        create_cross_sections_xml(args.output_dir, args.xml_output)

if __name__ == "__main__":
    main() 