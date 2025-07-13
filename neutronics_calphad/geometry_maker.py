import openmc
import os
from openmc_plasma_source import tokamak_source
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math
from .config import ELEMENT_DENSITIES


def _create_materials(config: dict) -> dict:
    """Creates OpenMC materials from a configuration dictionary."""
    materials = {}
    for name, mat_config in config['materials'].items():
        material = openmc.Material(name=name)
        for element, percent in mat_config['elements'].items():
            material.add_element(element, percent, percent_type=mat_config.get('percent_type', 'ao'))
        material.set_density('g/cm3', mat_config['density'])
        if mat_config.get('depletable'):
            material.depletable = True
        materials[name] = material
    return materials

def _create_tokamak_source(config: dict):
    """Creates a tokamak source from a configuration dictionary."""
    geo_params = config['geometry']
    source_params = config['source']['parameters']
    
    source_minor_radius = geo_params['minor_radius'] * 0.9
    
    return tokamak_source(
        major_radius=geo_params['major_radius'],
        minor_radius=source_minor_radius,
        elongation=geo_params['elongation'],
        triangularity=geo_params['triangularity'],
        **source_params
    )

def _create_point_source(config: dict):
    """Creates a point source from a configuration dictionary."""
    source_params = config['source']['parameters']
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(source_params['coordinate'])
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([source_params['energy']], [1.0])
    return source

SOURCE_BUILDERS = {
    'tokamak': _create_tokamak_source,
    'point': _create_point_source,
}

def make_d_shape_region(major_radius, minor_radius, elongation, triangularity, offset=0.0):
    """Creates a D-shaped toroidal region from surfaces with a given offset."""
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


def _build_d_shape_geometry(config: dict, materials: dict):
    """Builds a D-shaped tokamak geometry from a configuration dictionary."""
    geo_config = config['geometry']
    major_radius = geo_config['major_radius']
    minor_radius = geo_config['minor_radius']
    elongation = geo_config['elongation']
    triangularity = geo_config['triangularity']

    # Define wedge boundaries for quarter torus
    wedge_x_plane = openmc.XPlane(x0=0, boundary_type='reflective')
    wedge_y_plane = openmc.YPlane(y0=0, boundary_type='reflective')

    # Define boundaries for each layer
    layer_boundaries = []
    current_offset = 0
    for layer in geo_config['layers']:
        current_offset += layer['thickness']
        layer_boundaries.append(make_d_shape_region(major_radius, minor_radius, elongation, triangularity, offset=current_offset))

    # Create cell regions
    cells = []
    
    # Plasma cell
    plasma_boundary = make_d_shape_region(major_radius, minor_radius, elongation, triangularity) & +wedge_x_plane & +wedge_y_plane
    plasma_cell = openmc.Cell(region=plasma_boundary, name='plasma')
    cells.append(plasma_cell)

    # Material layers
    for i, layer in enumerate(geo_config['layers']):
        outer_boundary = layer_boundaries[i] & +wedge_x_plane & +wedge_y_plane
        inner_boundary = layer_boundaries[i-1] if i > 0 else plasma_boundary
        
        # for the first layer, inner boundary is the plasma boundary
        if i == 0:
            inner_boundary = plasma_boundary
        else: # for subsequent layers, it's the outer boundary of the previous layer
            prev_offset = sum(l['thickness'] for l in geo_config['layers'][:i])
            inner_boundary = make_d_shape_region(major_radius, minor_radius, elongation, triangularity, offset=prev_offset) & +wedge_x_plane & +wedge_y_plane

        
        layer_region = outer_boundary & ~inner_boundary
        layer_cell = openmc.Cell(region=layer_region, fill=materials[layer['material']], name=layer['name'])
        cells.append(layer_cell)

    # Shield for D-shape
    shield_config = next((l for l in geo_config['layers'] if l['name'] == 'shield'), None)
    if shield_config:
        offset_tank = sum(l['thickness'] for l in geo_config['layers'] if l['name'] != 'shield')
        a_tank = minor_radius + offset_tank
        x_min_tank, x_max_tank = major_radius - a_tank, major_radius + a_tank
        z_max_tank = elongation * a_tank

        shield_thickness = shield_config['thickness']
        shield_left = openmc.XPlane(x_min_tank - shield_thickness)
        shield_right = openmc.XPlane(x_max_tank + shield_thickness)
        shield_bottom = openmc.ZPlane(-z_max_tank - shield_thickness)
        shield_top = openmc.ZPlane(z_max_tank + shield_thickness)

        shield_y_extent = geo_config.get('shield_y_extent', 600)
        y_min_shield = openmc.YPlane(-shield_y_extent)
        y_max_shield = openmc.YPlane(shield_y_extent)
        shield_box_region = (+shield_left & -shield_right & +shield_bottom & -shield_top & +y_min_shield & -y_max_shield)

        tank_boundary = layer_boundaries[-2] & +wedge_x_plane & +wedge_y_plane # boundary before shield
        shield_region = shield_box_region & ~tank_boundary & +wedge_x_plane & +wedge_y_plane
        shield_cell = openmc.Cell(region=shield_region, fill=materials[shield_config['material']], name='shield')
        cells.append(shield_cell)

    # Outer bounding box
    pad = geo_config['bounding_box_pad']
    last_layer_offset = sum(l['thickness'] for l in geo_config['layers'])
    a_outer = minor_radius + last_layer_offset
    x_max_outer = (major_radius + a_outer) + pad
    y_max_outer = geo_config.get('shield_y_extent', a_outer) + pad
    z_max_outer = (elongation * a_outer) + pad

    x_max_plane = openmc.XPlane(x_max_outer, boundary_type='vacuum')
    y_max_plane = openmc.YPlane(y_max_outer, boundary_type='vacuum')
    z_min_plane = openmc.ZPlane(-z_max_outer, boundary_type='vacuum')
    z_max_plane = openmc.ZPlane(z_max_outer, boundary_type='vacuum')
    
    bounding_box_region = (+wedge_x_plane & -x_max_plane &
                           +wedge_y_plane & -y_max_plane &
                           +z_min_plane & -z_max_plane)
    
    # Void and world
    core_components_region = plasma_boundary
    for cell in cells:
        if cell.name != 'plasma':
            core_components_region |= cell.region
    
    void_region = bounding_box_region & ~core_components_region
    void_cell = openmc.Cell(region=void_region, name='void')
    cells.append(void_cell)

    world_region = ~bounding_box_region
    world_cell = openmc.Cell(name='world_void', region=world_region)
    cells.append(world_cell)

    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)

    # Set volumes
    def get_d_shape_volume(offset_outer, offset_inner=0.0):
        a_outer = minor_radius + offset_outer
        a_inner = minor_radius + offset_inner
        return math.pi * major_radius * (a_outer**2 - a_inner**2) * elongation / 4.0

    current_offset = 0
    for layer in geo_config['layers']:
        if layer['name'] != 'shield':
            outer = current_offset + layer['thickness']
            materials[layer['material']].volume = get_d_shape_volume(outer, current_offset)
            current_offset = outer

    return geometry

def _build_spherical_geometry(config: dict, materials: dict):
    """Builds a spherical geometry from a configuration dictionary."""
    geo_config = config['geometry']
    layers = geo_config['layers']
    
    surfaces = []
    current_radius = 0
    for layer in layers:
        current_radius += layer['thickness']
        surfaces.append(openmc.Sphere(r=current_radius))
        
    # Bounding sphere
    pad = geo_config['bounding_box_pad']
    bounding_sphere = openmc.Sphere(r=current_radius + pad, boundary_type='vacuum')

    # Create cells
    cells = []
    # Core (plasma or void)
    cells.append(openmc.Cell(region=-surfaces[0], name='core'))

    for i in range(len(surfaces) - 1):
        region = -surfaces[i+1] & +surfaces[i]
        layer_info = layers[i]
        cell = openmc.Cell(region=region, fill=materials[layer_info['material']], name=layer_info['name'])
        cells.append(cell)
        
    # Last layer
    last_layer_info = layers[-1]
    last_layer_region = -surfaces[-1] & +surfaces[-2] if len(surfaces) > 1 else -surfaces[0]
    cells.append(openmc.Cell(region=last_layer_region, fill=materials[last_layer_info['material']], name=last_layer_info['name']))

    # Void cell
    void_region = -bounding_sphere & +surfaces[-1]
    cells.append(openmc.Cell(region=void_region, name='void'))

    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)
    
    # Set volumes
    current_radius = 0
    for layer in layers:
        outer_r = current_radius + layer['thickness']
        volume = 4/3 * math.pi * (outer_r**3 - current_radius**3)
        materials[layer['material']].volume = volume
        current_radius = outer_r
        
    return geometry

GEOMETRY_BUILDERS = {
    'd_shape': _build_d_shape_geometry,
    'spherical': _build_spherical_geometry,
}

def create_model(config: dict):
    """Creates an OpenMC model from a configuration dictionary.
    
    Args:
        config (dict): A dictionary defining the model parameters.

    Returns:
        openmc.Model: The complete OpenMC model for the simulation.
    """
    
    # Create materials
    materials_dict = _create_materials(config)
    openmc_materials = openmc.Materials(materials_dict.values())

    # Create geometry
    geom_type = config['geometry']['type']
    if geom_type not in GEOMETRY_BUILDERS:
        raise ValueError(f"Unknown geometry type: {geom_type}")
    geometry = GEOMETRY_BUILDERS[geom_type](config, materials_dict)

    # Create source
    source_type = config['source']['type']
    if source_type not in SOURCE_BUILDERS:
        raise ValueError(f"Unknown source type: {source_type}")
    source = SOURCE_BUILDERS[source_type](config)

    # Create settings
    settings = openmc.Settings()
    settings_config = config.get('settings', {})
    settings.run_mode = settings_config.get('run_mode', "fixed source")
    settings.source = source
    settings.batches = settings_config.get('batches', 10)
    settings.inactive = settings_config.get('inactive', 0)
    settings.particles = settings_config.get('particles', 10000)
    
    model = openmc.model.Model(materials=openmc_materials, geometry=geometry, settings=settings)
    
    # Attach vv_cell to model for later use if it exists
    for cell in geometry.get_all_cells().values():
        if 'vessel' in cell.name:
            model.vv_cell = cell
            break

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
    
    geo_type = 'unknown'
    if hasattr(model, 'config'):
        geo_type = model.config['geometry']['type']

    if geo_type == 'd_shape':
        plot_2d.origin = (330, 0, 0)
        plot_2d.width = (600, 600)
    else: # spherical or default
        plot_2d.origin = (0, 0, 0)
        plot_2d.width = (model.geometry.bounding_box[1][0]*2.2, model.geometry.bounding_box[1][2]*2.2)

    plot_2d.pixels = (2000, 2000)
    plot_2d.color_by = 'cell'
    
    # Get cells from the model's geometry
    cells = model.geometry.get_all_cells()
    
    colors = {}
    legend_patches = []
    
    # Dynamically create colors and legend patches
    # A bit of a hack to get some color variation
    color_map = ['blue', 'darkgray', 'yellow', 'purple', 'green', 'orange', 'cyan']
    
    for i, cell in enumerate(cells.values()):
        if cell.name not in ['void', 'world_void', 'plasma', 'core']:
            color = color_map[i % len(color_map)]
            colors[cell] = color
            legend_patches.append(mpatches.Patch(color=color, label=f'{cell.name}'))

    # Add plasma separately if it exists
    plasma_cell = next((c for c in cells.values() if c.name in ['plasma', 'core']), None)
    if plasma_cell:
        colors[plasma_cell] = (255, 255, 255) # white

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
    y_coords = []
    z_coords = []
    
    if model.settings.source:
        # handle both single source and list of sources
        sources = model.settings.source if isinstance(model.settings.source, list) else [model.settings.source]
        for source in sources:
            if hasattr(source, 'space') and isinstance(source.space, openmc.stats.Point):
                 x_coords.append(source.space.x)
                 y_coords.append(source.space.y)
                 z_coords.append(source.space.z)
            else: # Tokamak source
                # This part is tricky as tokamak_source is a composite source
                # For visualization, we just show the major radius
                 if hasattr(source, 'major_radius'):
                     x_coords.append(source.major_radius)
                     z_coords.append(0)


    ax.scatter(x_coords, z_coords, color='red', s=1, alpha=0.5)

    ax.set_xlabel("X coordinate [cm]")
    ax.set_ylabel("Z coordinate [cm]")
    ax.set_title("Tokamak Cross-section with Neutron Source")

    legend_handles = [
        mlines.Line2D(
            [], [], color='red', marker='o', linestyle='None',
            markersize=5, label='Source Particles'
        ),
        *legend_patches
    ]
    ax.legend(handles=legend_handles)
    
    output_filename = os.path.join(output_dir, 'geometry_with_source.png')
    fig.savefig(output_filename, dpi=300)
    print(f"Saved plot to {output_filename}")
