# Monte Carlo simulation of suprathermal particles

Author: Cyril Markovitch

## Overview

This repository contains a Monte Carlo algorithm for calculating the energy distribution of suprathermal particles and the yield of chemical reactions.

The simulation follows `N_mc` suprathermal particles individually and computes their energy degradation collision by collision. For each collision, the reaction is selected probabilistically according to the corresponding collision frequencies, and the kinetic energy of the suprathermal particle is updated.

The code is designed to simulate energy degradation in a background gas, taking into account the relevant elastic and inelastic collision processes.

---

## Repository structure

```text
MC_supra_structured/
│
├── data/
│   ├── cross_section_data/
│   │   └── ...
│   ├── build_CS_dico.py
│   └── build_tables.py
│
├── physics/
│   ├── reactions.py
│   └── species.py
│
├── utils_functions/
│   └── mc_function.py
│
├── examples/
│   └── ...
│
├── README.md
├── requirements.txt
└── .gitignore


## Main Files

`utils_functions/mc_function.py`
Contains the main Monte Carlo algorithm, implemented in the function monte_carlo_numba.This function performs the particle tracking and calculates the resulting energy distribution and reaction yields.

`physics/reactions.py`
Contains the reactions considered in the simulation.
Reactions are implemented as Python classes containing the relevant physical parameters and properties of each process.

`physics/species.py`
contains species of the background gas with their mass.

`data/build_CS_dico.py`
contains cross sections of reactions. For elastic collisions, differential cross section should be also provided to calculate the cumulative distribution function F_theta.

`data/build_tables.py`
transform the cross sections and F_theta into a good format. Interpalotation is done over a given grid. 0 is given outside the range of definition. 

`data/cross_section_data`
Folder containing cross section and differential cross section data.


-------- How to use the code?

You can find a example in examples/Hydrogen_distribution which gathered all the information to compute a distribution and the Yield of reactions. Here are the steps to follow before running the example.
1. Build the cross section dictionnary : run in the terminal python3 -m data.build_CS_dico 
2. Build cross section tables : run in the terminal python3 -m data.build_tables 
3. Go to the file examples/Hydrogen_distribution to see what are the input of the main function monte_carlo_numba.

------- Monte-Carlo algorithm description

Here we provide some details about the main function monte_carlo_numba in utils_functions/mc_function.

1. The algorithm starts by computing the total collision frequency as a function of the energy taking into account the reactions that have been selected for the simulation. For this we use the cross sections previously downloaded.
2. Enter the Monte-Carlo loop. For each particle as much as its energy is higher than E_cutoff, the energy degradation continue.
3. Energy degradation: Commpute the probability of each process to randomly choose one of them for the collision. Compute the kinetic energy loss/gain of the suprathermal particle. Continue until the energy of the suprathermal particle is smaller than E_cutoff or that the suprathermal particle has been removed by a process such as photoionization.
4. Repeat this N_mc times and build the energy distribution and the Yield.

------- Questions

1. How to add a reaction to the algorithm ?

	a. Add a class in the file physics/reactions.py  
    	--> Important remark : Elastic collisions has to be the first reaction because when building tables in the file data/build_tables, 
        indexes are based on the fact that elastic collisions are given in REACTION_list first

	b. Add the cross section in the file data/build_CS_dico.py
    	--> Remark for rovibrational transition : If rovib is modelled with a unique cross section and Delta_E along with the stopping cross section
        Then a temperature dependancy is added.

	c. Run in the terminal " python3 -m data.build_CS_dico " --> Build the dictionnary of cross section

	d. Run in the terminal " python3 -m data.build_tables " --> Build the tables used in the MC code of all the data needed
