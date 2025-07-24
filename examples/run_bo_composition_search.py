"""Example script demonstrating Bayesian optimization driven composition search.

This script mirrors ``run_fispact_depletion.py`` but performs the material
exploration in a loop using a simple Bayesian optimizer.  At each iteration a
batch of candidate compositions is suggested, depleted with the independent
operator, evaluated using :func:`evaluate_material`, and the results fed back
to the optimizer.

The search terminates after a fixed number of iterations.  Custom gas and dose
criteria can be passed in ``critical_limits`` and ``dose_limits`` dictionaries.
"""

from __future__ import annotations

import os
from typing import Dict, List

import openmc
import openmc.deplete
import numpy as np

from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.optimizer.evaluate_updated import evaluate_material
from neutronics_calphad.utils.io import material_string, create_material
from neutronics_calphad.optimizer.bayesian_optimizer import BayesianOptimizer

# -----------------------------------------------------------------------------
# OpenMC configuration (update paths to your local data as needed)
openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'

# Load chain once for efficiency
openmc.deplete.Chain.from_xml(openmc.config['chain_file'])

# -----------------------------------------------------------------------------
# Geometry and flux preparation
model = create_model(config=SPHERICAL)
model.settings.particles = 1000
cells = list(model.geometry.get_all_cells().values())

# Collapsed XS and flux for each cell
flux, microxs = openmc.deplete.get_microxs_and_flux(
    model,
    cells,
    energies='UKAEA-1102',
    chain_file=openmc.config['chain_file'],
    run_kwargs={'cwd': 'microxs_and_flux'},
    path_statepoint=os.path.join('microxs_and_flux', 'microxs_statepoint.10.h5'),
)

# Index of vessel cell to replace material
VESSEL_CELL = next(i for i, c in enumerate(cells) if c.name == 'vessel')

# -----------------------------------------------------------------------------
# Time scheduler for irradiation / cooling
POWER_MW = 500
FUSION_POWER_MEV = 17.6
MEV_TO_J = 1.602176634e-13
SOURCE_RATE = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J)

scheduler = TimeScheduler(
    irradiation_time='1 year',
    cooling_times=['2 weeks', '1 year', '10 years', '100 years'],
    source_rate=SOURCE_RATE,
    irradiation_steps=12,
)

TIMESTEPS, SOURCES = scheduler.get_timesteps_and_source_rates()

# -----------------------------------------------------------------------------
# Optimization setup
ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']
optimizer = BayesianOptimizer(ELEMENTS, batch_size=3, minimize=False)

CRIT_LIMITS = {"He_appm": 1000, "H_appm": 500}
DOSE_LIMITS = {14: 1e2, 365: 1e-2, 3650: 1e-2, 36500: 1e-4}

N_ITERATIONS = 2  # demonstration only

for iteration in range(N_ITERATIONS):
    print(f"\n=== Iteration {iteration+1} ===")
    batch = optimizer.suggest()

    scores: List[float] = []
    for comp in batch:
        comp_dict = dict(zip(ELEMENTS, comp))
        mat_name = material_string(comp_dict, 'V')
        material = create_material(comp_dict, mat_name)
        material.depletable = True

        # Replace vessel material
        model.materials.append(material)
        vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
        material.volume = next(m.volume for m in model.materials if m.name == 'vcrtiwzr')
        vessel_cell.fill = material

        # Run depletion
        results = run_independent_depletion(
            model=model,
            depletable_cell='vessel',
            microxs=microxs[VESSEL_CELL],
            flux=flux[VESSEL_CELL],
            chain_file=openmc.config['chain_file'],
            timesteps=TIMESTEPS,
            source_rates=SOURCES,
            outdir=os.path.join('depletion_results', mat_name),
        )

        score = evaluate_material(
            results=results,
            chain_file=openmc.config['chain_file'],
            abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012',
            critical_gas_limits=CRIT_LIMITS,
            dose_limits=DOSE_LIMITS,
        )
        scores.append(score)
        print(f"{mat_name}: score={score}")

    optimizer.update(batch, np.array(scores))

print("\nOptimization complete")
