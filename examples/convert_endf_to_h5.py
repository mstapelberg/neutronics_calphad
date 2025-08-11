"""
ENDF to HDF5 Conversion Script

This script converts ENDF .asc files to .h5 format for use with OpenMC.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

def convert_endf_to_h5(endf_dir: str, output_dir: str, nuclides: Optional[List[str]] = None) -> None:
    """
    Convert ENDF .asc files to HDF5 format.
    
    Args:
        endf_dir: Directory containing ENDF .asc files
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
            print(f"Converting {asc_file.name}...")
            
            # Determine file type and convert
            # These are ENDF neutron cross section files (format: ElementMassg.asc)
            if asc_file.name.endswith('g.asc') and len(asc_file.stem) >= 3:
                try:
                    # Try NJOY processing first (these are raw ENDF files)
                    print(f"  Processing {asc_file.name} with NJOY...")
                    data = openmc.data.IncidentNeutron.from_njoy(
                        asc_file,
                        temperatures=[293.0],
                        evaluation='endf'
                    )
                    h5_file = output_path / f"{data.name}.h5"
                    data.export_to_hdf5(h5_file)
                    print(f"  Successfully converted to {h5_file.name}")
                except Exception as e:
                    print(f"  Failed to convert {asc_file.name}: {e}")
                    continue
                    
            elif "decay" in asc_file.name.lower():
                # Decay data
                data = openmc.data.Decay.from_endf(asc_file)
                h5_file = output_path / f"{data.nuclide['name']}.h5"
                data.export_to_hdf5(h5_file)
                
            elif "nfy" in asc_file.name.lower() or "fission" in asc_file.name.lower():
                # Fission product yields
                data = openmc.data.FissionProductYields.from_endf(asc_file)
                h5_file = output_path / f"{data.nuclide['name']}_fpy.h5"
                data.export_to_hdf5(h5_file)
                
            else:
                print(f"  Skipping {asc_file.name} - unknown file type")
                continue
            
            print(f"  Successfully converted to {h5_file.name}")
            successful += 1
            
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
    parser = argparse.ArgumentParser(description="Convert ENDF .asc files to HDF5 format")
    parser.add_argument("endf_dir", help="Directory containing ENDF .asc files")
    parser.add_argument("output_dir", help="Directory to save HDF5 files")
    parser.add_argument("--nuclides", nargs="+", help="Specific nuclides to convert")
    parser.add_argument("--create-xml", action="store_true", 
                       help="Create cross_sections.xml file after conversion")
    parser.add_argument("--xml-output", default="cross_sections.xml",
                       help="Output file for cross_sections.xml")
    
    args = parser.parse_args()
    
    # Convert ENDF files
    convert_endf_to_h5(args.endf_dir, args.output_dir, args.nuclides)
    
    # Create cross_sections.xml if requested
    if args.create_xml:
        create_cross_sections_xml(args.output_dir, args.xml_output)

if __name__ == "__main__":
    main() 