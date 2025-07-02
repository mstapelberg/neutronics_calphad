#!/usr/bin/env python3
"""
Check if gas-producing reactions are present in the depletion chain.
"""

import openmc.deplete
from pathlib import Path

def check_gas_production():
    """Verify gas-producing reactions exist in the depletion chain."""
    
    print("🔍 Checking Depletion Chain for Gas Production")
    print("="*50)
    
    # Path to full chain
    full_chain_file = Path.home() / 'nuclear_data' / 'chain-endf-b8.0.xml'
    
    if not full_chain_file.exists():
        print(f"❌ Chain file not found: {full_chain_file}")
        return
    
    # Load the chain
    chain = openmc.deplete.Chain.from_xml(full_chain_file)
    
    # Elements to check
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    # Gas products we're looking for
    gas_products = ['He3', 'He4', 'H1', 'H2', 'H3']
    
    print(f"Checking elements: {elements}")
    print(f"Looking for gas products: {gas_products}")
    print()
    
    for element in elements:
        print(f"\n{element} Gas-Producing Reactions:")
        print("-" * 40)

        gas_reactions_found = 0

        # chain.nuclides is a list of Nuclide objects
        for nuclide in chain.nuclides:
            # nuclide.name is e.g. "V-51"
            if not nuclide.name.startswith(element):
                continue

            # check each ReactionTuple
            for reaction in nuclide.reactions:
                product = reaction.target  # string or None
                if product and product in gas_products:
                    gas_reactions_found += 1
                    reaction_type = getattr(reaction, "type", "unknown")
                    br = getattr(reaction, "branching_ratio", None)
                    br_str = f", BR={br:.3f}" if br is not None else ""
                    print(f"  {nuclide.name} + n → {product} ({reaction_type}{br_str})")

                    # only show first few
                    if gas_reactions_found > 5:
                        print("  ... and more")
                        break
            if gas_reactions_found > 5:
                break

        if gas_reactions_found == 0:
            print(f"  ⚠️  No gas-producing reactions found for {element}!")
        else:
            print(f"  ✅ Found {gas_reactions_found} gas-producing reactions")

    
    # Check for important threshold reactions
    print(f"\n\nChecking Key Threshold Reactions:")
    print("-"*40)
    
    key_reactions = [
        ('V51', 'He4', '(n,α)'),
        ('V51', 'H1', '(n,p)'),
        ('Cr52', 'He4', '(n,α)'),
        ('Ti48', 'He4', '(n,α)'),
        ('W182', 'He4', '(n,α)'),
    ]
    
    for isotope, product, rxn_type in key_reactions:
        if isotope in chain.nuclides:
            nuclide = chain.nuclides[isotope]
            found = False
            for reaction in nuclide.reactions:
                for prod in reaction.products:
                    if hasattr(prod, 'name') and prod.name == product:
                        found = True
                        break
                if found:
                    break
            
            if found:
                print(f"  ✅ {isotope} {rxn_type} → {product} present")
            else:
                print(f"  ❌ {isotope} {rxn_type} → {product} NOT FOUND")
    
    print("\n💡 Summary:")
    print("  If gas reactions are missing, check:")
    print("  1. Nuclear data library version (need ENDF/B-VIII.0)")
    print("  2. Chain file completeness")
    print("  3. Cross section availability for high-energy neutrons")

if __name__ == "__main__":
    check_gas_production() 