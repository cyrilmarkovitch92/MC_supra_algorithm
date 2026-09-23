# Monte Carlo simulation of suprathermal particles

<p align="center">
  <img src="Image_Atmo_escape.jpg" width="400">
</p>

## Overview

This repository contains a Monte Carlo algorithm for calculating the energy distribution of suprathermal particles and the yield of chemical reactions.

The simulation follows `N_mc` suprathermal particles individually and computes their energy degradation collision by collision. For each collision, the reaction is selected probabilistically according to the corresponding collision frequencies, and the kinetic energy of the suprathermal particle is updated.

The code is designed to simulate energy degradation in a background gas, taking into account the relevant elastic and inelastic collision processes, along with the photochemistry and chemical reactions.

The code requires Python 3 and several scientific Python packages. The required packages are listed in `requirements.txt`

---

## Repository structure

```text
MC_supra_structured/
│
├── data/
│   ├── cross_section_data/
│   │   └── ...
│   ├── CS_dico.pkl
|   ├── tables_CS_theta_inverse.pkl
|   ├── build_CS_dico.py
│   └── build_tables.py
│
├── physics/
|   ├── constants.py
│   ├── reactions.py
│   └── species.py
│
├── utils_functions/
|   ├── cross_section_fct.py
│   ├── mc_function.py
|   └── usefull_fct.py
│
├── examples/
│   └── ...
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Main Files

`utils_functions/mc_function.py` : Contains the main Monte Carlo algorithm, implemented in the function `monte_carlo_numba`.This function performs the particle tracking and calculates the resulting energy distribution and reaction yields.

`physics/reactions.py` : Contains the reactions considered in the simulation.
Reactions are implemented as Python classes containing the relevant physical parameters and properties of each process.

`physics/species.py` : Contains the species present in the background gas, including their masses and other relevant properties.

`data/build_CS_dico.py` : Builds the cross-section dictionary used by the simulation. For elastic collisions, differential cross sections must also be provided in order to calculate the cumulative angular distribution $F_{\theta}(\theta)$

`data/build_tables.py` : Transforms the cross sections and cumulative angular distributions into tables used by the Monte Carlo algorithm.
Interpolation is performed on a predefined energy grid. Outside the range where cross-section data are defined, the corresponding values are set to zero.

`data/cross_section_data` : Folder containing cross sections and differential cross sections data used in the simulation.


---

## How to run the algorithm

### Preparing the cross-section tables

Before running a Monte Carlo simulation, the cross-section data must be processed.
From the root directory of the repository, run

```
python3 -m data.build_CS_dico
```
This builds the cross-section dictionary.

Then run:
```
python3 -m data.build_tables
```
This builds the tables used by the Monte Carlo algorithm.
These two steps should be repeated whenever the reaction list or the corresponding cross-section data are modified.


### Running the Monte Carlo simulation

An example of a complete simulation can be found in `examples/Hydrogen_distribution.py`. To run it from the terminal, go to the path `MC_supra_algorithm/` and type 
```
python3 -m examples.Hydrogen_distribution
```

If you want to run it from a Jupyter notebook cell in VScode, you have to define the working repository as `MC_supra_algorithm/`. For that type in the beginning of the .py file :
```ruby
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
```

The example contains the information required to define a simulation and illustrates how to call the main Monte Carlo function:
```ruby
monte_carlo_numba
```

The general workflow is:
1. Install the required Python packages.
2. Build the cross-section dictionary.
3. Build the cross-section tables.
4. Define the gas composition and the reactions to be included.
5. Define simulation parameters.
6. Run the Monte Carlo simulation.


---

## Description of the Monte Carlo algorithm

The Monte Carlo calculation can be summarized as follows.

### 1. Calculate the total collision frequency

The total collision frequency is calculated as a function of the suprathermal-particle energy, taking into account the reactions selected for the simulation.
The relevant cross sections are obtained from the tables generated during the preprocessing step.

### 2. Follow each suprathermal particle
For each simulated particle, the energy degradation is followed collision by collision. The particle remains in the simulation while its energy is greater than the specified cutoff energy E_cutoff.

### 3. Sample the collision process
At each collision, the probability of each reaction is calculated from its corresponding collision frequency.
A reaction is then selected randomly according to these probabilities.
The kinetic energy of the suprathermal particle is updated according to the selected process.
The particle continues to be followed until either:
- its energy falls below `E_cutoff`, or
- it is removed from the simulation by a process such as photoionization.

### 4. Repeat the calculation
The procedure is repeated for `N_mc` particles.
The resulting trajectories are used to calculate quantities such as:
- the energy distribution of the suprathermal particles;
- the yield of the different reactions.


---
## Adding a new reaction
To add a new reaction to the simulation, the following steps are required.

### 1. Add the reaction
Add a new reaction class to `physics/reactions.py`

> [!CAUTION] 
> At present, elastic collisions must be the first reaction in `REACTION_list`. The table-building procedure in `data/build_tables.py` uses reaction indices based on this ordering. Therefore, changing the position of the elastic collision in `REACTION_list` may lead to incorrect table construction.

### 2. Add the cross-section data

Add the corresponding cross-section information in `data/build_CS_dico.py`. 
> [!IMPORTANT]
> For elastic collisions, the differential cross section must also be provided if the angular distribution is required.
> For rovibrational transitions modeled using a unique cross section together with `Delta_E` and a stopping cross section, a temperature dependence is included in the corresponding treatment.

### 3. Rebuild the cross-section dictionary and the tables

After modifying the reaction list or cross-section data, run:
```
python3 -m data.build_CS_dico
python3 -m data.build_tables
```


---
## Contact

For questions, suggestions, or issues related to the code, please contact: markovitchcyril@gmail.com

