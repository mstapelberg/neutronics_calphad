# Steps to take in this script:

# 1) Create a spherical geometry using the config file 
# 2) run get_microxs_and_flux to get the microxs and flux
# 3) Replace the material at the vacuum vessel with a candidate material 
# 4) use independent operator with predictor integrator to get the depletion results 
# 5) save the dose rate results at the specific time steps for that material 
# 6) repeat for the whole batch of materials 
# 7) evaluate the batch of material's dose rates and gas production rates 
# 8) create a new batch using BO to expore the composition space further 
# 9) repeat until our batch prediction is within 1% of the true value? Or find a better metric. 

# 10) deliver a list of N materials that satisfy our constraints using the Surrogate model 


from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.geometry_maker import create_model
import openmc
import openmc.deplete
from typing import Dict
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.optimizer.evaluate_updated import evaluate_material 
from neutronics_calphad.utils.io import material_string, create_material
import os


openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'

chain = openmc.deplete.Chain.from_xml(openmc.config['chain_file'])

print(dir(chain))
print(chain.reactions)


# 1) Create a spherical geometry using the config file 
model = create_model(config=SPHERICAL)
model.settings.particles = 1000

all_cells = model.geometry.get_all_cells()
cells_list = list(all_cells.values())


# 2) run get_microxs_and_flux to get the microxs and flux
flux, microxs = openmc.deplete.get_microxs_and_flux(model,
                                                    cells_list,
                                                    energies='UKAEA-1102',
                                                    chain_file=openmc.config['chain_file'],
                                                    run_kwargs={'cwd': 'microxs_and_flux'},
                                                    path_statepoint=os.path.join('microxs_and_flux', 'microxs_statepoint.10.h5'))


test_material = {
    'V': 0.9,
    'Cr': 0.02,
    'Ti': 0.04,
    'W': 0.03,
    'Zr': 0.01
}

material_name = material_string(test_material, 'V')
new_material = create_material(test_material, material_name)
# make new material depletable 
new_material.depletable = True

# add to materials
cell_name = 'vessel'
model.materials.append(new_material)
vv_cell = model.geometry.get_cells_by_name(cell_name)
for mat in model.materials:
    if mat.name == 'vcrtiwzr':
        new_volume = mat.volume
new_material.volume = new_volume
vv_cell[0].fill = new_material

# 4) use independent operator with predictor integrator to get the depletion results 

# calculate the source_rate 
POWER_MW = 500
FUSION_POWER_MEV = 17.6
MEV_TO_J = 1.602176634e-13
SOURCE_RATE = POWER_MW * 1E6 / (FUSION_POWER_MEV * MEV_TO_J)
print(SOURCE_RATE)

# irradiation timesteps


cooling_times = ['2 weeks', '1 year', '10 years', '100 years']
# create the time scheduler 
time_scheduler = TimeScheduler(irradiation_time='1 year',
                               cooling_times=cooling_times,
                               source_rate=SOURCE_RATE,
                               irradiation_steps=12)

# get the index of the vessel cell
vessel_cell_index = next(index for index, cell in enumerate(cells_list) if cell.name == 'vessel')

results = run_independent_depletion(
                                    model=model,
                                    depletable_cell=cell_name,
                                    microxs=microxs[vessel_cell_index],
                                    flux=flux[vessel_cell_index],
                                    chain_file=openmc.config['chain_file'],
                                    timesteps=time_scheduler.get_timesteps_and_source_rates()[0],
                                    source_rates=time_scheduler.get_timesteps_and_source_rates()[1],
                                    outdir=os.path.join('depletion_results', material_name)
)

# 7) evaluate the batch of material's dose rates and gas production rates 
material_score = evaluate_material(results=results,
                                   chain_file=openmc.config['chain_file'],
                                   abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')

print(f" Score for {material_name} is {material_score}")


# 8) create a new batch using BO to expore the composition space further 

# 9) repeat until our batch prediction is within 1% of the true value? Or find a better metric. 