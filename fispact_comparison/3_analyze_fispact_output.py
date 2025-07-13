import pypact as pp
from pathlib import Path
import sys

def main():
    """
    Analyzes the 'printlib' output file from a FISPACT-II simulation
    to extract and display microscopic cross-section data for key nuclides.
    """
    script_dir = Path(__file__).parent
    
    # The output file name is based on the 'output_name' in '2_run_fispact.py'
    fispact_output_file = script_dir / 'vv_vanadium.out'

    if not fispact_output_file.exists():
        print(f"❌ ERROR: FISPACT output file not found at {fispact_output_file}")
        print("➡️ Please run '2_run_fispact.py' first to generate it.")
        sys.exit(1)

    print(f"📖 Analyzing FISPACT 'printlib' output from: {fispact_output_file}\n")
    
    with pp.PrintLib(str(fispact_output_file)) as lib:
        print(f"Found data for {len(lib.nuclides)} nuclides at the final time step.")
        
        # You can customize this list with nuclides you want to investigate.
        # These are common activation products for vanadium.
        target_nuclides = ['V51', 'V52', 'Ti50', 'Sc47', 'Cr51'] 
        
        for nuclide_name in target_nuclides:
            nuclide_data = next((n for n in lib.nuclides if n.name == nuclide_name), None)
            
            if nuclide_data:
                print(f"\n--- Analysis for {nuclide_name} ---")
                
                # Sort reactions by average cross-section (descending) and take the top 5
                sorted_reactions = sorted(nuclide_data.reactions, key=lambda r: r.cross_section, reverse=True)
                
                if sorted_reactions:
                    print(f"  Top 5 reactions by cross-section ({len(sorted_reactions)} total):")
                    for reaction in sorted_reactions[:5]:
                        if reaction.cross_section > 0.0:
                            print(f"    - {reaction.name:<15}: {reaction.cross_section:.4e} barns")
                else:
                    print("  No reaction data found in printlib output.")
            else:
                print(f"\n--- No data found for {nuclide_name} in the output ---")
    
    print("\n✅ Analysis complete.")
    print("➡️ You can modify the 'target_nuclides' list in this script to investigate other nuclides.")


if __name__ == "__main__":
    main() 