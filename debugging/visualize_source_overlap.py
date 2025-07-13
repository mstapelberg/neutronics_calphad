#!/usr/bin/env python3
"""
Visualize source-geometry overlap to debug why neutrons aren't reaching the VV.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import ARC_D_SHAPE
import matplotlib.pyplot as plt

def visualize_source_overlap():
    """Check if the tokamak source overlaps with the geometry."""
    
    print("🎯 Visualizing Source-Geometry Overlap")
    print("="*50)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    config = ARC_D_SHAPE
    model = create_model(config)
    
    # Get VV cell bounding box
    vv_cell = model.vv_cell
    vv_bbox = vv_cell.bounding_box
    print(f"VV Bounding Box:")
    print(f"  X: {vv_bbox[0][0]:.1f} to {vv_bbox[1][0]:.1f} cm")
    print(f"  Y: {vv_bbox[0][1]:.1f} to {vv_bbox[1][1]:.1f} cm")
    print(f"  Z: {vv_bbox[0][2]:.1f} to {vv_bbox[1][2]:.1f} cm")
    
    # Get source information
    sources = model.settings.source
    print(f"\nTotal sources: {len(sources)}")
    
    # Sample positions from first few sources
    n_sources_to_check = min(100, len(sources))
    r_values = []
    z_values = []
    
    for i in range(n_sources_to_check):
        source = sources[i]
        if hasattr(source.space, 'r') and hasattr(source.space, 'z'):
            r_dist = source.space.r
            z_dist = source.space.z
            
            # Get values from discrete distributions
            if hasattr(r_dist, 'x'):
                r_values.extend(r_dist.x)
            if hasattr(z_dist, 'x'):
                z_values.extend(z_dist.x)
    
    if r_values and z_values:
        r_min, r_max = min(r_values), max(r_values)
        z_min, z_max = min(z_values), max(z_values)
        
        print(f"\nSource spatial ranges from {n_sources_to_check} sources:")
        print(f"  R: {r_min:.1f} to {r_max:.1f} cm")
        print(f"  Z: {z_min:.1f} to {z_max:.1f} cm")
        
        # Expected plasma/source region
        geo_config = config['geometry']
        plasma_r_min = geo_config['major_radius'] - geo_config['minor_radius'] * 0.9
        plasma_r_max = geo_config['major_radius'] + geo_config['minor_radius'] * 0.9
        plasma_z_min = -geo_config['elongation'] * geo_config['minor_radius'] * 0.9
        plasma_z_max = geo_config['elongation'] * geo_config['minor_radius'] * 0.9
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot expected plasma region
        plasma_rect = plt.Rectangle((plasma_r_min, plasma_z_min), 
                                  plasma_r_max - plasma_r_min,
                                  plasma_z_max - plasma_z_min,
                                  fill=False, edgecolor='green', linewidth=2,
                                  label='Expected Plasma Region')
        ax.add_patch(plasma_rect)
        
        # Plot VV region (approximate from bounding box)
        # Convert from 3D bbox to R-Z projection
        vv_r_min = 0  # Inner edge at axis
        vv_r_max = max(vv_bbox[1][0], vv_bbox[1][1])  # Outer edge
        vv_z_min = vv_bbox[0][2]
        vv_z_max = vv_bbox[1][2]
        
        vv_rect = plt.Rectangle((vv_r_min, vv_z_min),
                              vv_r_max - vv_r_min,
                              vv_z_max - vv_z_min,
                              fill=False, edgecolor='blue', linewidth=2,
                              label='VV Bounding Box')
        ax.add_patch(vv_rect)
        
        # Plot actual source points
        if len(r_values) < 1000:  # Only plot if not too many
            ax.scatter(r_values, z_values, c='red', s=10, alpha=0.5, label='Source Points')
        else:
            # Plot source region
            source_rect = plt.Rectangle((r_min, z_min),
                                      r_max - r_min,
                                      z_max - z_min,
                                      fill=True, facecolor='red', alpha=0.3,
                                      edgecolor='red', linewidth=2,
                                      label='Source Region')
            ax.add_patch(source_rect)
        
        ax.set_xlabel('R [cm]')
        ax.set_ylabel('Z [cm]')
        ax.set_title('Source vs. Geometry Overlap Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        ax.set_xlim(0, 600)
        ax.set_ylim(-300, 300)
        
        plt.tight_layout()
        plt.savefig('source_geometry_overlap.png', dpi=150)
        print(f"\nVisualization saved to source_geometry_overlap.png")
        
        # Check for overlap
        print(f"\n🔍 Overlap Analysis:")
        
        # Check R overlap
        r_overlap = not (r_max < vv_r_min or r_min > vv_r_max)
        print(f"  R overlap: {'YES' if r_overlap else 'NO'}")
        if not r_overlap:
            print(f"    Source R: {r_min:.1f}-{r_max:.1f}, VV R: {vv_r_min:.1f}-{vv_r_max:.1f}")
        
        # Check Z overlap  
        z_overlap = not (z_max < vv_z_min or z_min > vv_z_max)
        print(f"  Z overlap: {'YES' if z_overlap else 'NO'}")
        if not z_overlap:
            print(f"    Source Z: {z_min:.1f}-{z_max:.1f}, VV Z: {vv_z_min:.1f}-{vv_z_max:.1f}")
        
        if not (r_overlap and z_overlap):
            print(f"\n SOURCE AND VV DO NOT OVERLAP!")
            print(f"   This explains why no neutrons reach the VV")
    else:
        print("Could not extract source position data")

if __name__ == "__main__":
    visualize_source_overlap() 