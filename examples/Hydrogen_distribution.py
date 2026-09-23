#%% Define the working repository
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

#%% 
from physics.reactions import *
from physics.species import *
from utils_functions.mc_function import *
from utils_functions.usefull_fct import Class_to_list_MC
from utils_functions.cross_section_fct import CS_morton2003
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import pickle


#######################################################
# In this example we compute the energy distribution of suprathermal H atoms in a 
# gas of H, H2 and H2O. Thanks to the distribution we compute the reaction rate of 
# the reaction H + H2O --> H + OH.
#######################################################


# =====================================================
# Simulation parameters
# =====================================================

supra_species = H
m_supra = supra_species.mass
N_mc = int(1e4) # Number of Monte-Carlo simulations
E_cutoff = 0 ## Cut-off energy
Temperature = 300 #K
Eth = kb_cste*Temperature

## Production rate of suprathermal atoms W0
E0_max = 50 # eV 
dE0 = 0.05 #eV
E0 = 10 # eV
W0 = 1 #cm-1.s-1

## Energy bins
E_dist_list = np.logspace(-3.5,np.log10(E0_max+dE0), 500)


# =====================================================
# Species, densities and reactions
# =====================================================

species_names = ['h', 'h2', 'h2o']

n_h2 = 1e3 #cm-3
n_h2o = n_h2/100
n_h = n_h2/1000
n_tot = n_h2 + n_h2o + n_h

densities = np.array([n_h, n_h2, n_h2o]) # !! Densities should be in the same order than species !!

## Reactions involved : H+H-->H+H ; H+H2 --> H+H2 ; H+H2O --> H2+OH ; gamma + H --> e- + H+
wanted_reactions = [H_H_elastic, H_H2_elastic, H_H2O__OH_H2, hnu_h__hp_e] # Those are dictionnaries defined in physics/reactions.py

# =====================================================
# Load cross section data
# =====================================================

with open('data/tables_CS_theta_inverse.pkl', 'rb') as file:
    tables = pickle.load(file)

E_cs_list = tables["E_cs_list"]
CS_table = tables["CS_table"]


# =====================================================
# Optional : if you want to change the cross section of a reaction
# =====================================================

#### For H2
## Define the cross section
Ei = (E_cs_list)*mH/(mH+mH2)
E0_Panarese = 0.5*mH*(3.75*1e3)**2/eV
sigma_Panarese = 17.5e-16*(E0_Panarese /Ei)**0.45

## Choose the reaction by finding it with its name. This will update the local cross section table
for ir,reaction in enumerate(REACTIONS_list):
        if reaction.name =="h+h2-->h+h2":
            tables["CS_table"][ir] = sigma_Panarese


#### For H2O
Eb=0.88 #eV
mu = (mH * 18 * mH) / (mH + 18 * mH) # kg
# Parameters from Table 3 of Morton 2003 : https://linkinghub.elsevier.com/retrieve/pii/S0032063303000473 
A = 1.5e-10*1e-6 # m^3.s^-1.K^-(n-0.5)
n = 0.5
Ea = Eb * eV # J  

CS_HH2O = CS_morton2003(E_cs_list*mu/mH * eV, mu, Ea, A, n) # cm^2

for ir,reaction in enumerate(REACTIONS_list):
        if reaction.name =="h+h2o-->oh+h2":
            tables["CS_table"][ir] = CS_HH2O 

# =====================================================
# Run the Monte-Carlo simulation
# =====================================================

## Random list of inital energies, to uniformly distribute the production ratre W0 into the bin
E0_random_list = np.random.uniform(E0 - dE0 / 2, E0 + dE0 / 2, N_mc)

## Change format to make it compatible with numba
reac_channel_arr, m_projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr, threshold_arr = Class_to_list_MC(wanted_reactions)
tables_list = tuple(tables[key] for key in tables.keys())

## Impose a collision frequency for photoionization
fcoll_ioniz = 1e-9 # s-1

(E_dist_list, H_distrib,
    H_distrib_table,
    Yield,
    SE_distrib_table,
    N_dist_list,
    frequency_tot_distrib_sum,
    Ncoll_th,
    relax_time_list,
    E_last,dEi_list) = monte_carlo_numba(m_supra,N_mc, E_dist_list, Eth, E_cutoff,
        E0_random_list, W0, species_names, densities, tables_list,
        reac_channel_arr, m_projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr ,reaction_type_arr, threshold_arr,
        fcoll_ioniz = fcoll_ioniz)




# =====================================================
# Get the energy distribution and the Yield
# ===================================================== 

## Energy distribution : H_distrib_tot [cm-3.eV-1]

## Yield[i] = Yield of wanted_reactions[i]

# =====================================================
# Calculate the rate coefficient k_i of reaction i
# ===================================================== 

############## EXAMPLE : H + H2O --> H + OH

## Cross section of the process 

Eb=0.88 #eV
mu = (mH * 18 * mH) / (mH + 18 * mH) # kg
# Parameters from Table 3 of Morton 2003 : https://linkinghub.elsevier.com/retrieve/pii/S0032063303000473 
A = 1.5e-10*1e-6 # m^3.s^-1.K^-(n-0.5)
n = 0.5
Ea = Eb * eV # J  

CS_HH2O = CS_morton2003(E_dist_list*mu/mH * eV, mu, Ea, A, n) # cm^2


freq_supra_inte = H_distrib*CS_HH2O*np.sqrt(2*E_dist_list*eV/mH)*1e2 # s^-1.eV^-1
freq_supra = simpson(freq_supra_inte, E_dist_list) # s^-1

## Production rate of H + H2O --> H + OH 
Rate = freq_supra*n_h2o # cm-3.s-1

## Yield calculated from the distribution
Yield_distrib = Rate/W0

## Yield from the simulation
Yield_simu = Yield[2] # Index is 2 corresponds to the index of the reaction H_H2O__OH_H2 in wanted_reactions

print("Yield of H + H2O --> H + OH calculated from the distribution : ", Yield_distrib)
print("Yield of H + H2O --> H + OH calculated in the simulation : ", Yield_simu)

# =====================================================
# Figure of the energy distribution
# ===================================================== 

plt.figure(dpi=200)
plt.plot(E_dist_list, H_distrib)
plt.xlabel(r'$E_{lab}$ [eV]')
plt.ylabel(r'$f(E)$ [cm$^{-3}$.eV$^{-1}$]')
#plt.axvline(x=E_cutoff, color='blue', linestyle='--', label=r'E$_{\rm cutoff}$')
plt.yscale('log')
plt.xscale('log')
plt.grid(which='both', linestyle='--')
plt.title(f"Non-thermal hydrogen distribution in a gas of H, H2, H2O at a temperature of T={Temperature} K ")
plt.show()


