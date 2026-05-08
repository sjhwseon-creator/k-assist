from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import time
import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.style.use('seaborn-v0_8-pastel')

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.autolayout'] = True

######################## Initial condition ##############################
Mechpath = r'C:\Users\Hwan\Desktop\RWTH\Projekt\Combustion and Flame\Optimazation Modell\\'
mech = 'HGD_Seong_V55.yaml'   # HGD_Seong_V29.yaml
gas = ct.Solution(Mechpath + mech)

# Inlet gas conditions
reactorTemperature = 600  # Kelvin
reactorPressure = ct.one_atm  # in atm
inletConcentrations = {'COC*OOC': 0.01, 'O2': 0.012, 'N2': 0.978}
gas.TPX = reactorTemperature, reactorPressure, inletConcentrations

########################################################################

# Reactor parameters
residenceTime = 1.5  # s
reactorVolume = 9.2 * (1e-2) ** 3  # m3

# Instrument parameters
pressureValveCoefficient = 0.01
maxPressureRiseAllowed = 0.01
maxSimulationTime = 50.0  # seconds

fuelAirMixtureTank = ct.Reservoir(gas)
exhaust = ct.Reservoir(gas)

stirredReactor = ct.IdealGasReactor(gas, energy='off', volume=reactorVolume)

massFlowController = ct.MassFlowController(
    upstream=fuelAirMixtureTank,
    downstream=stirredReactor,
    mdot=stirredReactor.mass / residenceTime,
)

pressureRegulator = ct.Valve(
    upstream=stirredReactor,
    downstream=exhaust,
    K=pressureValveCoefficient,
)

reactorNetwork = ct.ReactorNet([stirredReactor])

# Determine fuel species key from inlet composition (first non-O2/N2 species)
fuel_species = next(
    species for species in inletConcentrations.keys() if species not in {'O2', 'N2'}
)

# Species to export (fuel means the fuel species key from inletConcentrations)
species_to_export = ['CH4', 'C2H6', 'C2H4', 'CH3OH', 'H2', 'CO2', 'CO', fuel_species, 'O2']

# Optional sanity check: make sure species are present in mechanism
missing_species = [sp for sp in species_to_export if sp not in gas.species_names]
if missing_species:
    raise ValueError(
        'These requested species are not in the mechanism: {}'.format(', '.join(missing_species))
    )

# Define all temperatures
T = [825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100]

# Output DataFrame only for selected species
tempDependence = pd.DataFrame(columns=species_to_export)
tempDependence.index.name = 'Temperature'
concentrations = inletConcentrations

for temperature in T:
    reactorTemperature = temperature
    reactorPressure = ct.one_atm
    reactorVolume = 9.2 * (1e-2) ** 3

    gas.TPX = reactorTemperature, reactorPressure, inletConcentrations

    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)

    # use previous converged composition as initial guess
    gas.TPX = reactorTemperature, reactorPressure, concentrations

    stirredReactor = ct.IdealGasReactor(gas, energy='off', volume=reactorVolume)
    massFlowController = ct.MassFlowController(
        upstream=fuelAirMixtureTank,
        downstream=stirredReactor,
        mdot=stirredReactor.mass / residenceTime,
    )
    pressureRegulator = ct.Valve(
        upstream=stirredReactor,
        downstream=exhaust,
        K=pressureValveCoefficient,
    )
    reactorNetwork = ct.ReactorNet([stirredReactor])

    tic = time.time()
    t = 0
    print('Start Simulation at ' + str(temperature))
    while t < maxSimulationTime:
        t = reactorNetwork.step()

    toc = time.time()
    print('Simulation at T={}K took {:3.2f}s to compute'.format(temperature, toc - tic))

    concentrations = stirredReactor.thermo.X

    # Store only selected species mole fractions
    selected_state = {sp: stirredReactor.thermo[sp].X[0] for sp in species_to_export}
    tempDependence.loc[temperature] = selected_state

output_dir = r'C:\Users\Hwan\Desktop\RWTH\Projekt\Simulation\JSR'
output_file = output_dir + r'\species_selected.csv'
tempDependence.to_csv(path_or_buf=output_file)
print('Saved selected species to {}'.format(output_file))
