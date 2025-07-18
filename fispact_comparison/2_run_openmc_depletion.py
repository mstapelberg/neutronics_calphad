import openmc
import openmc.deplete
import numpy as np
import pandas as pd
from pathlib import Path
import re
import os
import warnings
from typing import List

# --- Simulation Constants ---
POWER_MW = 500  # Megawatts (match step 1)
ENERGY_PER_FUSION_MEV = 14.1  # MeV (match step 1)

# Define a standard irradiation/cooling schedule
IRRADIATION_TIME_S = 1.0 * 365.25 * 24 * 3600  # 1 year
COOLING_TIMES_S = np.array([
    1, 3600, 10*3600, 24*3600, 7*24*3600, 14*24*3600, 30*24*3600,
    60*24*3600, 365*24*3600, 5*365*24*3600, 10*365*24*3600,
    25*365*24*3600, 100*365*24*3600
])

def calculate_source_rate(power_mw: float, energy_per_fusion_mev: float) -> float:
    """Calculates the total neutron source rate from fusion power."""
    power_watts = power_mw * 1e6
    energy_joules = energy_per_fusion_mev * 1.602e-13
    return power_watts / energy_joules

def extract_fispact_microxs(printlib_path: Path, outdir: Path) -> Path:
    """
    Parses a FISPACT printlib file to extract one-group cross sections.
    This function has been updated with more robust parsing logic and debugging output.
    """
    # Suppress OpenMC nuclide naming warnings during parsing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GNDS naming convention.*", category=UserWarning)
        
        collapsed_data: List[dict] = []
        in_xs_block = False
        print("DEBUG: Starting to parse FISPACT printlib file...")
        lines_processed = 0
        non_empty_lines = 0
        
        # Map FISPACT reaction notation to OpenMC reaction notation
        fispact_to_openmc = {
            # Gas production reactions (critical for fusion materials)
            'n,p': '(n,p)',        # H-1 production
            'n,d': '(n,d)',        # H-2 production  
            'n,t': '(n,t)',        # H-3 production
            'n,h': '(n,p)',        # H-1 production (same as n,p)
            'n,a': '(n,a)',        # He-4 production
            
            # Neutron multiplication reactions
            'n,2n': '(n,2n)',
            'n,3n': '(n,3n)', 
            'n,4n': '(n,4n)',
            
            # Other transmutation reactions
            'n,g': '(n,gamma)',    # Neutron capture
            'n,gamma': '(n,gamma)',
            
            # Fission (if present in FISPACT data)
            'n,f': 'fission',
            'n,F': 'fission',
            'fission': 'fission',
            
            # Skip reactions not required by OpenMC:
            # 'n,E' - elastic scattering (no transmutation)
            # 'n,n' - elastic scattering (no transmutation) 
            # 'n,total' - total cross sections (not specific reactions)
            # 'n,Xp', 'n,Xa' - residual production (not in required list)
            # 'n,2a', 'n,pa', 'n,2p' - not in OpenMC required list
        }
        
        # Pattern to match: "Nuclide (reaction) Product CrossSection+-Error"
        pattern = re.compile(r'([A-Z][a-z]?\d+m?)\s+\(([^)]+)\)\s+([A-Z][a-z]?\d+m?)\s+([\d.E+-]+)')

        with open(printlib_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if "The cross section for the specified reaction" in line:
                    print(f"DEBUG: Found start marker at line {line_num}. Activating parser.")
                    in_xs_block = True
                    next(f, None)
                    continue
                if in_xs_block and line.strip().startswith('1') and 'B R E M S S T R A H L U N G' in line:
                    print(f"DEBUG: Found end marker at line {line_num}. Deactivating parser.")
                    break
                if not in_xs_block:
                    continue
                
                lines_processed += 1
                line_content = line.strip()
                
                if not line_content: 
                    continue
                    
                non_empty_lines += 1
                
                # Use the simple pattern to find all cross-section entries on the line
                matches = pattern.findall(line_content)
                
                if non_empty_lines <= 5:  # Debug first few lines
                    print(f"DEBUG: Line {line_num}: Found {len(matches)} matches")
                    if matches:
                        print(f"DEBUG: Matches: {matches}")
                
                for match in matches:
                    nuclide, reaction, product, cross_section_str = match
                    
                    # Clean up the reaction string and map to OpenMC notation
                    reaction_clean = reaction.strip()
                    openmc_reaction = fispact_to_openmc.get(reaction_clean)
                    
                    if openmc_reaction is None:
                        if non_empty_lines <= 5:
                            print(f"DEBUG: Skipping unsupported reaction: '{reaction_clean}'")
                        continue
                    
                    try:
                        # Get the MT number for this reaction
                        mt = openmc.data.REACTION_MT.get(openmc_reaction)
                        if mt is None:
                            if non_empty_lines <= 5:
                                print(f"DEBUG: OpenMC doesn't recognize reaction: '{openmc_reaction}'")
                            continue
                        
                        # Extract cross section value (before the +- error)
                        xs_val = float(cross_section_str.split('+-')[0])
                        
                        # Convert nuclide name to OpenMC format
                        nuclide_name = re.sub(r'm(\d*)', r'_m\1', nuclide)
                        
                        # Validate the nuclide exists in OpenMC
                        try:
                            openmc.Nuclide(nuclide_name)
                        except (ValueError, openmc.exceptions.DataError):
                            if non_empty_lines <= 5:
                                print(f"DEBUG: Invalid nuclide: '{nuclide_name}'")
                            continue
                        
                        if xs_val > 0.0:
                            collapsed_data.append({
                                'nuclide': nuclide_name, 
                                'reaction': openmc_reaction,  # Store the string, not MT number
                                'xs': xs_val
                            })
                            if len(collapsed_data) <= 10:
                                print(f"DEBUG: Added entry: {nuclide_name}, MT={mt} ({openmc_reaction}), xs={xs_val}")
                                
                    except (ValueError, IndexError) as e:
                        if non_empty_lines <= 5:
                            print(f"DEBUG: Parse error for match {match}: {e}")
                        continue

        print(f"DEBUG: Processed {lines_processed} total lines, {non_empty_lines} non-empty lines in cross-section block")
        print(f"DEBUG: Found {len(collapsed_data)} cross-section entries")

        if not collapsed_data:
            raise ValueError("Could not parse any cross-section data from the printlib file.")

        fispact_xs_csv = outdir / "collapsed_microxs_fispact.csv"
        
        # Convert to DataFrame and format for OpenMC compatibility
        df = pd.DataFrame(collapsed_data)
        
        print(f"DEBUG: Raw data shape: {df.shape}")
        print(f"DEBUG: Unique nuclides: {df['nuclide'].nunique()}")
        print(f"DEBUG: Unique reactions: {df['reaction'].nunique()}")
        print(f"DEBUG: Sample nuclides: {df['nuclide'].unique()[:10]}")
        print(f"DEBUG: Sample reactions: {df['reaction'].unique()[:10]}")
        
        # Add the missing 'groups' column (collapsed to single group)
        df['groups'] = 1
        
        # Rename columns to match OpenMC format
        df = df.rename(columns={'nuclide': 'nuclides', 'reaction': 'reactions'})
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['nuclides', 'reactions']).sort_values(['nuclides', 'reactions'])
        
        print(f"DEBUG: After deduplication: {df.shape}")
        
        # Create complete matrix: OpenMC expects ALL nuclide-reaction combinations
        all_nuclides = df['nuclides'].unique()
        
        # Use the exact reactions required by OpenMC (from chain analysis)
        required_openmc_reactions = [
            '(n,2n)',
            '(n,3n)', 
            '(n,4n)',
            '(n,a)',
            '(n,d)',
            '(n,gamma)',
            '(n,p)',
            '(n,t)',
            'fission',
        ]
        fispact_reactions = df['reactions'].unique()
        all_reactions = list(set(required_openmc_reactions) | set(fispact_reactions))
        all_groups = [1]  # Single energy group
        
        print(f"DEBUG: Creating complete matrix:")
        print(f"  - {len(all_nuclides)} nuclides")
        print(f"  - {len(all_reactions)} reactions")
        print(f"  - {len(all_groups)} groups")
        print(f"  - Expected total: {len(all_nuclides) * len(all_reactions) * len(all_groups)}")
        
        # Create all combinations
        import itertools
        all_combinations = list(itertools.product(all_nuclides, all_reactions, all_groups))
        
        # Create complete DataFrame
        complete_df = pd.DataFrame(all_combinations, columns=['nuclides', 'reactions', 'groups'])
        
        # Merge with our data, filling missing values with 0.0
        complete_df = complete_df.merge(df[['nuclides', 'reactions', 'xs']], 
                                       on=['nuclides', 'reactions'], 
                                       how='left')
        complete_df['xs'] = complete_df['xs'].fillna(0.0)
        
        print(f"DEBUG: Complete matrix shape: {complete_df.shape}")
        print(f"DEBUG: Final unique nuclides: {complete_df['nuclides'].nunique()}")
        print(f"DEBUG: Final unique reactions: {complete_df['reactions'].nunique()}")
        print(f"DEBUG: Non-zero entries: {(complete_df['xs'] > 0).sum()}")
        
        # Reorder columns to match OpenMC format: nuclides, reactions, groups, xs
        df = complete_df[['nuclides', 'reactions', 'groups', 'xs']]
        
        df.to_csv(fispact_xs_csv, index=False)
        print(f"✅ Extracted {len(df)} unique FISPACT cross sections, saved to {fispact_xs_csv}")
        print(f"DEBUG: CSV columns: {list(df.columns)}")
        print(f"DEBUG: First few rows:")
        print(df.head())
        return fispact_xs_csv

def run_depletion_with_microxs(flux: np.ndarray,
                               microxs_path: Path,
                               chain_file: str,
                               outdir: Path,
                               out_suffix: str,
                               volume: float) -> None:
    """Runs an OpenMC depletion calculation with a given set of cross sections."""
    
    # Debug: Check the input CSV format
    print(f"DEBUG: Loading MicroXS from: {microxs_path}")
    if microxs_path.exists():
        df_check = pd.read_csv(microxs_path)
        print(f"DEBUG: Input CSV shape: {df_check.shape}")
        print(f"DEBUG: Input CSV columns: {list(df_check.columns)}")
        print(f"DEBUG: Input unique nuclides: {df_check['nuclides'].nunique()}")
        print(f"DEBUG: Input unique reactions: {df_check['reactions'].nunique()}")
        print(f"DEBUG: Input unique groups: {df_check['groups'].nunique()}")
        print(f"DEBUG: Sample input data:")
        print(df_check.head())
    
    material = openmc.Material(name="vv_material")
    material.add_element("V", 1.0, "ao")
    material.set_density("g/cm3", 6.1)
    material.depletable = True
    material.volume = volume  # Set the volume here

    microxs = openmc.deplete.MicroXS.from_csv(microxs_path)
    
    operator = openmc.deplete.IndependentOperator(
        materials=[material],
        fluxes=[flux],
        micros=[microxs],
        chain_file=chain_file,
        normalization_mode='source-rate'
    )

    timesteps_s = np.insert(COOLING_TIMES_S, 0, IRRADIATION_TIME_S)
    source_rate = calculate_source_rate(POWER_MW, ENERGY_PER_FUSION_MEV)
    source_rates = np.insert(np.zeros_like(COOLING_TIMES_S, dtype=float), 0, source_rate).tolist()
    
    print(f"DEBUG: Source rate = {source_rate:.3e} n/s")
    print(f"DEBUG: Source rates = {source_rates[:3]}... (first 3 values)")
    
    integrator = openmc.deplete.PredictorIntegrator(
        operator, 
        timesteps_s, 
        source_rates=source_rates,
        timestep_units='s'
    )
    integrator.integrate(path=str(outdir / "depletion_results.h5"))
    
    results_file = outdir / "depletion_results.h5"
    if results_file.exists():
        results_file.rename(outdir / f"depletion_results_{out_suffix}.h5")
    print(f"✅ Depletion run with {out_suffix} cross sections complete.")

def main():
    """Main function to run comparative depletion calculations."""
    script_dir = Path(__file__).parent
    
    # --- File Paths ---
    flux_file = script_dir / 'vv_flux.npy'
    openmc_microxs_file = script_dir / 'microxs_1102.csv'
    fispact_printlib_file = script_dir / 'printlib.out'
    volume_file = script_dir / 'vv_volume.txt'
    chain_file = os.environ.get('OPENMC_CHAIN_FILE')

    if not all([flux_file.exists(), openmc_microxs_file.exists(), volume_file.exists(), chain_file]):
        raise FileNotFoundError("Missing required files from step 1. Run '1_get_flux_and_xs.py' first.")

    if not fispact_printlib_file.exists():
        raise FileNotFoundError("FISPACT 'printlib.out' not found. Please run FISPACT first.")

    # --- Load Data from Step 1 ---
    flux = np.load(flux_file)
    with open(volume_file, 'r') as f:
        volume = float(f.read())

    # --- Extract FISPACT Cross Sections ---
    collapsed_fispact_xs_path = extract_fispact_microxs(fispact_printlib_file, script_dir)
    

    # --- Run Depletion with OpenMC's Multi-Group Cross Sections ---
    print("\n--- Running Depletion with OpenMC Multi-Group Cross Sections ---")
    run_depletion_with_microxs(flux, openmc_microxs_file, chain_file, script_dir, 'openmc_xs', volume)
    
    # --- Run Depletion with FISPACT's Collapsed Cross Sections ---
    print("\n--- Running Depletion with FISPACT Collapsed Cross Sections ---")
    run_depletion_with_microxs(flux, collapsed_fispact_xs_path, chain_file, script_dir, 'fispact_xs', volume)

    print("\nAll depletion runs complete.")

if __name__ == "__main__":
    main()
