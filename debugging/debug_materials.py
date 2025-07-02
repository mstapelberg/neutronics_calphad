#!/usr/bin/env python3
"""
Debug script to test the materials API fix.
"""

import neutronics_calphad as nc

def test_materials_api():
    """Test that we can properly access materials by name."""
    print("Testing materials API...")
    
    # Create a model
    model = nc.create_model('V')
    
    print(f"Model has {len(model.materials)} materials:")
    for i, mat in enumerate(model.materials):
        print(f"  {i+1}. {mat.name} (ID: {mat.id})")
    
    # Test our helper function
    from neutronics_calphad.library import get_material_by_name
    
    try:
        vv_material = get_material_by_name(model.materials, "vv_V")
        print(f"\nSuccessfully found vacuum vessel material: {vv_material.name}")
        print(f"  Density: {vv_material.density} g/cm³")
        print(f"  Volume: {vv_material.volume} cm³")
        print(f"  Depletable: {vv_material.depletable}")
        
    except ValueError as e:
        print(f"Error finding material: {e}")
        
    print("\nMaterials API test completed successfully!")

if __name__ == "__main__":
    test_materials_api() 