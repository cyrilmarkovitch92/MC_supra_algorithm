from physics.reactions import *
from utils_functions.mc_function import *
from utils_functions.usefull_fct import Class_to_list_MC
from utils_functions.cross_section_fct import CS_morton2003
import matplotlib.pyplot as plt
from scipy.integrate import simpson


#######################################################

#######################################################


# =====================================================
# Simulation parameters
# =====================================================

N_mc = int(1e4) # Number of Monte-Carlo simulations
E_cutoff = 0 ## Cut-off energy

## Production rate of suprathermal atoms W0
E0_max = 50 # eV 
dE0 = 0.05 #eV
E0_list = np.linspace(dE0, E0_max,5)
W0 = 1*np.ones(len(E0_list)) #cm-1.s-1.eV-1

## Energy bins
E_dist_list = np.logspace(-4,np.log10(E0_max+dE0), 500)


# =====================================================
# Species, densities and reactions
# =====================================================

species_names = ['h', 'h2', 'h2o']

n_h2 = 1e21 #cm-3
n_h2o = n_h2/100
n_h = n_h2/2
n_tot = n_h2 + n_h2o + n_h

densities = np.array([n_h, n_h2, n_h2o]) # !! Densities should be in the same order than species !!

## Reactions involved : H+H-->H+H ; H+H2 --> H+H2 ; gamma + H --> e- + H+
wanted_reactions = [H_H_elastic, H_H2_elastic, hnu_h__hp_e] # Those are dictionnaries defined in physics/reactions.py

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

## Define the cross section
Ei = (E_cs_list)*mH/(mH+mH2)
E0_Panarese = 0.5*mH*(3.75*1e3)**2/eV
sigma_Panarese = 17.5e-16*(E0_Panarese /Ei)**0.45

## Choose the reaction to which this has to be applied
for ir,reaction in enumerate(REACTIONS_list):
        if reaction.name =="h+h2-->h+h2":
            tables["CS_table"][ir] = sigma_Panarese


# =====================================================
# Run the Monte-Carlo simulation
# =====================================================


H_distrib_tot = np.zeros(len(E_dist_list)-1)
for iE0, E0 in enumerate(E0_list):
    print("--", iE0, ' / ', len(E0_list))

    ## Random list of inital energies, to uniformly distribute the production ratre W0 into the bin
    E0_random_list = np.random.uniform(E0 - dE0 / 2, E0 + dE0 / 2, N_mc)

    ## Change format to make it compatible with numba
    reac_channel_arr, m_projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr, threshold_arr = Class_to_list_MC(wanted_reactions)
    tables_list = tuple(tables[key] for key in tables.keys())

    ## Impose a collision frequency for photoionization
    fcoll_ioniz = 1e-7 # s-1

    (E_dist_list_ourmodel, H_distrib_ourmodel,
        H_distrib_table_ourmodel,
        Yield_ourmodel,
        SE_distrib_table_ourmodel,
        N_dist_list_ourmodel,
        frequency_tot_distrib_sum_ourmodel,
        Ncoll_th_ourmodel,
        relax_time_list_ourmodel,
        E_last_ourmodel,dEi_list) = monte_carlo_numba(N_mc, E_dist_list, Eth, E_cutoff,
            E0_random_list, W0, species_names, densities, tables_list,
            reac_channel_arr, m_projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr ,reaction_type_arr, threshold_arr,
            fcoll_ioniz = fcoll_ioniz)

    H_distrib_tot += H_distrib_ourmodel


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

CS_HH2O = CS_morton2003(E_dist_list_ourmodel*mu/mH * eV, mu, Ea, A, n) # cm^2

freq_supra_inte = H_distrib_tot*CS_HH2O*np.sqrt(2*E_dist_list_ourmodel*eV/mH)*1e2 # s^-1.eV^-1
freq_supra = simpson(freq_supra_inte, E_dist_list_ourmodel) # s^-1

## Production rate
Rate = freq_supra*n_h2o # cm-3.s-1
