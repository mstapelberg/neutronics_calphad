import openmc
import os
from openmc_plasma_source import tokamak_source
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

# a dictionary of densities for the elements of interest in g/cm3
ELEMENT_DENSITIES = {
    'V': 6.11,
    'Cr': 7.19,
    'Ti': 4.54,
    'W': 19.35,
    'Zr': 6.51,
}

# CSG Model parameters
major_radius = 330
minor_radius = 113
elongation = 1.8
triangularity = 0.5
fw_thickness = 0.2
vv_thickness = 2
blanket_thickness = 20
tank_thickness = 3
shield_thickness = 30

def make_d_shape_region(offset=0.0):
    """Creates a D-shaped toroidal region from surfaces with a given offset.

    This function defines the geometry of a D-shaped cross-section commonly
    used in tokamaks. The shape is defined by a combination of a toroidal arc
    on the outboard side, a cylindrical surface on the inboard side, and
    flat top and bottom planes.

    Args:
        offset (float, optional): The radial offset to apply to the D-shape,
            used for creating layered components. Defaults to 0.0.

    Returns:
        openmc.Region: The region corresponding to the D-shape.
    """
    a = minor_radius + offset
    R0 = major_radius
    kappa = elongation
    delta = triangularity

    x_outboard = R0 + a
    x_inboard_corner = R0 - delta * a
    z_top = kappa * a
    x_inboard = R0 - a

    x_c = (x_inboard_corner**2 + z_top**2 - x_outboard**2) / (2 * (x_inboard_corner - x_outboard))
    r = x_outboard - x_c

    outboard_torus = openmc.ZTorus(x0=0, y0=0, z0=0, a=x_c, b=r, c=r)
    inboard_cylinder = openmc.ZCylinder(r=x_inboard)
    top_plane = openmc.ZPlane(z0=z_top)
    bottom_plane = openmc.ZPlane(z0=-z_top)

    return -outboard_torus & +inboard_cylinder & -top_plane & +bottom_plane

def create_model(element_symbol: str):
    """Creates an OpenMC model of a tokamak with a vacuum vessel made of the
    specified element.

    This function builds a complete OpenMC model including geometry, materials,
    and source definition. The vacuum vessel material is parameterized to allow
    for testing different elements.

    Args:
        element_symbol (str): The symbol of the element to use for the
            vacuum vessel (e.g., 'V', 'Cr').

    Returns:
        openmc.Model: The complete OpenMC model for the simulation.
    """

    # Define wedge boundaries for quarter torus
    wedge_x_plane = openmc.XPlane(x0=0, boundary_type='reflective')
    wedge_y_plane = openmc.YPlane(y0=0, boundary_type='reflective')

    # Define boundaries for each layer to create a clear, non-overlapping geometry
    plasma_boundary = make_d_shape_region() & +wedge_x_plane & +wedge_y_plane
    fw_boundary = make_d_shape_region(offset=fw_thickness) & +wedge_x_plane & +wedge_y_plane
    vv_boundary = make_d_shape_region(offset=fw_thickness + vv_thickness) & +wedge_x_plane & +wedge_y_plane
    blanket_boundary = make_d_shape_region(offset=fw_thickness + vv_thickness + blanket_thickness) & +wedge_x_plane & +wedge_y_plane
    tank_boundary = make_d_shape_region(offset=fw_thickness + vv_thickness + blanket_thickness + tank_thickness) & +wedge_x_plane & +wedge_y_plane

    # Define cell regions by taking the space between boundaries
    plasma_region = plasma_boundary
    fw_region = fw_boundary & ~plasma_boundary
    vv_region = vv_boundary & ~fw_boundary
    blanket_region = blanket_boundary & ~vv_boundary
    tank_region = tank_boundary & ~blanket_boundary

    offset_tank = fw_thickness + vv_thickness + blanket_thickness + tank_thickness
    a_tank = minor_radius + offset_tank
    x_min_tank, x_max_tank = major_radius - a_tank, major_radius + a_tank
    z_max_tank = elongation * a_tank

    shield_left = openmc.XPlane(x_min_tank - shield_thickness)
    shield_right = openmc.XPlane(x_max_tank + shield_thickness)
    shield_bottom = openmc.ZPlane(-z_max_tank - shield_thickness)
    shield_top = openmc.ZPlane(z_max_tank + shield_thickness)

    # add missing ±Y faces so the shield really is a box
    shield_y_extent = 600
    y_min_shield = openmc.YPlane(-shield_y_extent)
    y_max_shield = openmc.YPlane(shield_y_extent)
    shield_box_region = (+shield_left & -shield_right & +shield_bottom & -shield_top
                         & +y_min_shield & -y_max_shield)
    shield_region = shield_box_region & ~tank_boundary & +wedge_x_plane & +wedge_y_plane

    # Define the outer bounding box for the quarter torus
    pad = 50.0 # cm
    x_max = (x_max_tank + shield_thickness) + pad
    y_max = shield_y_extent
    z_max = (z_max_tank + shield_thickness) + pad
    z_min = -z_max
    
    # Outer boundaries should be vacuum, not reflective!
    x_max_plane = openmc.XPlane(x_max, boundary_type='vacuum')
    y_max_plane = openmc.YPlane(y_max, boundary_type='vacuum')
    z_min_plane = openmc.ZPlane(z_min, boundary_type='vacuum')
    z_max_plane = openmc.ZPlane(z_max, boundary_type='vacuum')

    # Bounding box for the quarter-torus world
    bounding_box_region = (+wedge_x_plane & -x_max_plane &
                           +wedge_y_plane & -y_max_plane &
                           +z_min_plane & -z_max_plane)

    # NEW - vacuum world cell eliminates "undefined space"
    world_region = ~bounding_box_region
    world_cell   = openmc.Cell(name='world_void', region=world_region)

    # Define the void region as everything within the bounding box that is not
    # occupied by another component. This is a more robust way to avoid gaps.
    core_components_region = tank_boundary | shield_region
    void_region = bounding_box_region & ~core_components_region

    # Materials
    vv_material = openmc.Material(name=f'vv_{element_symbol}')
    vv_material.add_element(element_symbol, 1.0, 'ao')
    vv_material.set_density('g/cm3', ELEMENT_DENSITIES[element_symbol])
    vv_material.depletable = True

    w_material = openmc.Material(name='mat1_tungsten')
    w_material.add_element('W', 1, percent_type="ao")
    w_material.set_density("g/cm3", 19.3)
    w_material.depletable = True

    flibe_material = openmc.Material(name='mat3_flibe')
    flibe_material.add_element('Li', 0.2857, percent_type='ao')
    flibe_material.add_element('F', 0.5714, percent_type='ao')
    flibe_material.add_element('Be', 0.1429, percent_type='ao')
    flibe_material.set_density('g/cm3', 1.95)
    flibe_material.depletable = True

    bc_material = openmc.Material(name='mat4_boron_carbide')
    bc_material.add_element('B', 0.8, percent_type='ao')
    bc_material.add_element('C', 0.2, percent_type='ao')
    bc_material.set_density('g/cm3', 2.5)
    bc_material.depletable = True

    steel_material = openmc.Material(name='mat5_steel')
    steel_material.add_element('Fe', 0.95, percent_type='ao')
    steel_material.add_element('Cr', 0.18, percent_type='ao')
    steel_material.add_element('Mn', 0.02, percent_type='ao')
    steel_material.add_element('Si', 0.01, percent_type='ao')
    steel_material.add_element('Ni', 0.10, percent_type='ao')
    steel_material.add_element('Mo', 0.02, percent_type='ao')
    steel_material.set_density('g/cm3', 8.03)
    steel_material.depletable = True

    # Cells
    plasma_cell = openmc.Cell(region=plasma_region, name='plasma')
    fw_cell = openmc.Cell(region=fw_region, fill=w_material, name='first_wall')
    vv_cell = openmc.Cell(region=vv_region, fill=vv_material, name='vacuum_vessel')
    blanket_cell = openmc.Cell(region=blanket_region, fill=flibe_material, name='blanket')
    tank_cell = openmc.Cell(region=tank_region, fill=steel_material, name='blanket_tank')
    shield_cell = openmc.Cell(region=shield_region, fill=bc_material, name='shield')
    void_cell = openmc.Cell(region=void_region, name='void')

    universe = openmc.Universe(cells=[plasma_cell, fw_cell, vv_cell, blanket_cell,
                                      tank_cell, shield_cell, void_cell, world_cell])
    geometry = openmc.Geometry(universe)

    # --- Set volumes for depletion ---
    def get_d_shape_volume(offset_outer, offset_inner=0.0):
        """Calculates the volume of a D-shaped toroidal shell."""
        a_outer = minor_radius + offset_outer
        a_inner = minor_radius + offset_inner
        # This is an approximation for the volume of a D-shaped torus
        # For a quarter torus, we divide by 4
        return math.pi * major_radius * (a_outer**2 - a_inner**2) * elongation / 4.0

    # First Wall
    w_material.volume = get_d_shape_volume(fw_thickness, 0)
    
    # Vacuum Vessel
    vv_material.volume = get_d_shape_volume(
        fw_thickness + vv_thickness,
        fw_thickness
    )

    # Blanket
    flibe_material.volume = get_d_shape_volume(
        fw_thickness + vv_thickness + blanket_thickness,
        fw_thickness + vv_thickness
    )
    
    # Tank (assumed to be filled with steel)
    # Note: the tank region uses steel_material, so we assign the volume here.
    # If there were another steel component, this would need refinement.
    steel_material.volume = get_d_shape_volume(
        fw_thickness + vv_thickness + blanket_thickness + tank_thickness,
        fw_thickness + vv_thickness + blanket_thickness
    )

    # Shield
    # The shield is a box with the D-shaped components subtracted.
    # We need to calculate the volume of the box and subtract the volumes
    # of the components inside it.
    shield_box_outer_radius = x_max_tank + shield_thickness
    shield_box_inner_radius = x_min_tank - shield_thickness
    shield_box_height = (z_max_tank + shield_thickness) * 2
    
    # Volume of a hollow cylinder approximation for the shield box
    # For a quarter torus, we divide by 4
    shield_box_volume = math.pi * (shield_box_outer_radius**2 - shield_box_inner_radius**2) * shield_box_height / 4.0
    
    total_d_shape_volume = get_d_shape_volume(offset_tank, 0)
    
    bc_material.volume = shield_box_volume - total_d_shape_volume

    materials = openmc.Materials([vv_material, w_material, flibe_material, bc_material, steel_material])

    # Revert source to a 90-degree wedge for the quarter torus
    # Reduce source size to ensure it stays within plasma boundary
    source_minor_radius = minor_radius * 0.9  # 5% smaller than plasma boundary
    source = tokamak_source(
        elongation=elongation,
        ion_density_centre=1.09e20,
        ion_density_pedestal=1.09e20,
        ion_density_peaking_factor=1,
        ion_density_separatrix=3e19,
        ion_temperature_centre=45.9e3,
        ion_temperature_pedestal=6.09e3,
        ion_temperature_separatrix=0.1e3,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_beta=6,
        major_radius=major_radius,
        minor_radius=source_minor_radius,  # Use smaller radius
        pedestal_radius=0.8 * source_minor_radius,  # Adjust pedestal accordingly
        mode="H",
        shafranov_factor=0.0,
        angles=(0, math.pi/2),
        sample_seed=42,
        triangularity=triangularity,
        fuel={"D": 0.5, "T": 0.5},
    )

    # Constrain the source to the y=0 plane for 2D simulation

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = source
    settings.batches = 10
    settings.inactive = 0
    settings.particles = 10000
    
    model = openmc.model.Model(materials=materials, geometry=geometry, settings=settings)
    model.vv_cell = vv_cell

    return model

def plot_model(model: openmc.Model, output_dir: str = "."):
    """Generates and saves a plot of the model geometry and neutron source.

    This function creates a 2D 'xz' plot of the OpenMC model, colors it by
    cell, and overlays the neutron source distribution. It also exports a 3D
    voxel plot. The resulting plot is saved as a PNG image.

    Args:
        model (openmc.Model): The OpenMC model to plot.
        output_dir (str, optional): The directory where the plot files will be
            saved. Defaults to the current directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_2d = openmc.Plot()
    plot_2d.filename = 'geometry_plot_2d'
    plot_2d.basis = 'xz'
    plot_2d.origin = (330, 0, 0)
    plot_2d.width = (600, 600)
    plot_2d.pixels = (2000, 2000)
    plot_2d.color_by = 'cell'
    
    # Get cells from the model's geometry
    cells = model.geometry.get_all_cells()
    plasma_cell = next(iter(model.geometry.get_cells_by_name('plasma')))
    fw_cell = next(iter(model.geometry.get_cells_by_name('first_wall')))
    vv_cell = model.vv_cell
    blanket_cell = next(iter(model.geometry.get_cells_by_name('blanket')))
    tank_cell = next(iter(model.geometry.get_cells_by_name('blanket_tank')))
    shield_cell = next(iter(model.geometry.get_cells_by_name('shield')))


    colors = {
        fw_cell: 'blue',
        vv_cell: 'darkgray',
        blanket_cell: 'yellow',
        tank_cell: 'darkgray',
        shield_cell: 'purple',
        plasma_cell: (255, 255, 255)
    }
    plot_2d.colors = colors

    # Create a 3D voxel plot
    plot_3d = openmc.Plot()
    plot_3d.filename = 'geometry_plot_3d'
    plot_3d.type = 'voxel'
    plot_3d.origin = (300, 300, 0)
    plot_3d.width = (600., 600., 700.)
    plot_3d.pixels = (200, 200, 200)
    plot_3d.color_by = 'cell'
    plot_3d.colors = colors

    plots_collection = openmc.Plots([plot_2d, plot_3d])
    
    # Export model (geometry, materials, settings) and plots to XML files
    # so that the openmc.plot_geometry() command can find them.
    model.export_to_xml(directory=output_dir) # openmc is stupid and model needs directory
    plots_collection.export_to_xml(path=output_dir) # openmc is stupid and plots needs path

    openmc.plot_geometry(cwd=output_dir)

    img_path = os.path.join(output_dir, plot_2d.filename + '.png')
    if not os.path.exists(img_path):
        print(f"Warning: Plot image not found at {img_path}")
        return

    img = plt.imread(img_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_extent = [
        plot_2d.origin[0] - plot_2d.width[0] / 2, plot_2d.origin[0] + plot_2d.width[0] / 2,
        plot_2d.origin[2] - plot_2d.width[1] / 2, plot_2d.origin[2] + plot_2d.width[1] / 2
    ]
    ax.imshow(img, extent=plot_extent)

    # Sample source positions for plotting
    x_coords = []
    z_coords = []
    for source in model.settings.source:
        r_val = source.space.r.x[0]
        z_val = source.space.z.x[0]
        x_coords.append(r_val)
        z_coords.append(z_val)

    ax.scatter(x_coords, z_coords, color='red', s=1, alpha=0.5)

    ax.set_xlabel("X coordinate [cm]")
    ax.set_ylabel("Z coordinate [cm]")
    ax.set_title("Tokamak Cross-section with Neutron Source")

    legend_handles = [
        mlines.Line2D(
            [], [], color='red', marker='o', linestyle='None',
            markersize=5, label='Source Particles'
        ),
        mpatches.Patch(color='blue', label='First Wall (Tungsten)'),
        mpatches.Patch(color='darkgray', label='Vessel/Tank (Steel)'),
        mpatches.Patch(color='yellow', label='Blanket (Flibe)'),
        mpatches.Patch(color='purple', label='Shield (B4C)')
    ]
    ax.legend(handles=legend_handles)
    
    output_filename = os.path.join(output_dir, 'geometry_with_source.png')
    fig.savefig(output_filename, dpi=300)
    print(f"Saved plot to {output_filename}")
