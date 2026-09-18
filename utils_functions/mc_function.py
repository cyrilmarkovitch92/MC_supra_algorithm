import numpy as np
from numba import njit

import pickle

from physics.constants import *

# ------------- Load Cross section and differential cross section tables
# with open('data/tables_CS_theta_inverse.pkl', 'rb') as file:
#     tables = pickle.load(file)



@njit
def interp_linear(x, xgrid, ygrid):
    n = len(xgrid)

    # Below range
    if x <= xgrid[0]:
        return 0 #ygrid[0]

    # Above range
    if x >= xgrid[n-1]:
        return 0 #ygrid[n-1]

    # Find interval
    for i in range(n-1):
        if xgrid[i] <= x <= xgrid[i+1]:
            x0 = xgrid[i]
            x1 = xgrid[i+1]
            y0 = ygrid[i]
            y1 = ygrid[i+1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return 0.0

@njit
def interp_linear_2D(x, y, xgrid, ygrid, zgrid):
    """
    Interpolation bilinéaire.

    Parameters
    ----------
    x, y : float
        Point où interpoler.
    xgrid : 1D array (nx,)
    ygrid : 1D array (ny,)
    zgrid : 2D array (nx, ny)
        zgrid[i, j] = f(xgrid[i], ygrid[j])

    Returns
    -------
    float
    """

    nx = len(xgrid)
    ny = len(ygrid)

    # Saturation sur les bords
    if x <= xgrid[0]:
        ix = 0
    elif x >= xgrid[nx - 1]:
        ix = nx - 2
    else:
        for i in range(nx - 1):
            if xgrid[i] <= x <= xgrid[i + 1]:
                ix = i
                break

    if y <= ygrid[0]:
        iy = 0
    elif y >= ygrid[ny - 1]:
        iy = ny - 2
    else:
        for j in range(ny - 1):
            if ygrid[j] <= y <= ygrid[j + 1]:
                iy = j
                break

    x0 = xgrid[ix]
    x1 = xgrid[ix + 1]
    y0 = ygrid[iy]
    y1 = ygrid[iy + 1]

    # Valeurs aux quatre coins
    z00 = zgrid[ix, iy]
    z10 = zgrid[ix + 1, iy]
    z01 = zgrid[ix, iy + 1]
    z11 = zgrid[ix + 1, iy + 1]

    # Coordonnées réduites
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    # Interpolation bilinéaire
    return (
        (1.0 - tx) * (1.0 - ty) * z00 +
        tx * (1.0 - ty) * z10 +
        (1.0 - tx) * ty * z01 +
        tx * ty * z11
    )


@njit
def Inverse_cdf_theta_numba(E, r, E_grid, r_grid, theta_inverse_table):

    logE = np.log10(E)
    logE_grid = np.log10(E_grid)

    i = np.searchsorted(logE_grid, logE) - 1

    if i < 0:
        i = 0
    if i >= logE_grid.shape[0] - 1:
        i = logE_grid.shape[0] - 2

    logE1 = logE_grid[i]
    logE2 = logE_grid[i+1]

    # Evaluate inverse CDF at both energies
    t1 = interp_linear(r, r_grid, theta_inverse_table[i])
    t2 = interp_linear(r, r_grid, theta_inverse_table[i+1])

    # Log interpolation in energy (unchanged physics)
    theta = t1 + (t2 - t1) * (logE - logE1) / (logE2 - logE1)

    return theta

@njit
def find_species_index(species_names, name_target):
    for i,species in enumerate(species_names):
        if species == name_target:
            return int(i)

@njit
def monte_carlo_numba(N_mc, E_dist_list, Eth, E_cutoff, E0_random_list, W0, species_names, densities, tables_list, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr , threshold_arr, size_max_secondaries=100000, statistics=False, tau_cste=1e20, fcoll_ioniz=0,   Emin_secondary=1e10): #
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - tables_list : list of tables for cross sections and differential cross sections --> Usefull if ones want to change a cross section in its script to test
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - Emin_secondary : minimum energy for secondary H atoms to be tracked [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    - tau_cste = tau/tau0 with (tau=time of reaction to remove H) and (tau0=typicall time between 2 collisions)
    
    """
    # =====================================================
    # compute new E_dist_list and dEi_list = energy distance between each bins
    # =====================================================
    ## E_dist_list :  We take the middle between each Ei : Ei_new = (E_i +E_(i+1))/2
    ## dEi_dist_list : We take the difference between the energy E_i  and E_(i+1), dEi_list[i] = (E_i - E_(i+1))

    dEi_list = E_dist_list[1::] - E_dist_list[0:-1]
    E_dist_list = (E_dist_list[1::] + E_dist_list[0:-1])/2


    T_mc = Eth*eV/kb_cste

    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    
    E_cs_list = tables_list[0]
    CS_table = tables_list[1]
    E_grid_array = tables_list[2]
    r_grid_array = tables_list[3]
    theta_inverse_array = tables_list[4]
    T_rovib_list = tables_list[5]


    # with open("simulation_log_test.txt", "w") as f:

    #     f.write("Monte Carlo Simulation Results\n")
    #     f.write("===============================\n\n")
    #     f.write("E_dist_list = " + str(E_dist_list) + "\n")
    #     f.write("bin_dist = " + str(bin_dist) + "\n\n")
    #     f.write(f"-------------------- precompute frequencies \n")

    
    #tau0 = 1.5e6 #1523.8 ## Test for Panarese
    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):
        #f.write(f"{iE+1}/{bin_dist} \n")

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2

        f_tot = 0.0
        
        for ir in range(len(reac_channel_arr)):

            if reaction_type_arr[ir] == "photoionization":
                # tau = tau_cste*tau0
                # fcoll = 1/tau
                fcoll = fcoll_ioniz
            else:
                reac_chnl = reac_channel_arr[ir]
                
                ## Projectile
                m_projectile = m_projectile_arr[ir]

                ## Target 
                m_target = m_target_arr[ir]
                name_target = target_arr[ir]

                # Find target to get its density
                for i_species,species in enumerate(species_names):
                    if species == name_target:
                        i_target = i_species
                #i_target = 2 #find_species_index(species_names, name_target)
                n_target = densities[i_target]

                ### Argument for isotropy : in average the target will be isotropic so themean tagret velocity is 0
                Ecm = Ei * m_target / (m_target + m_projectile)
                # v_proj_i = np.sqrt(2.0 * Ei * eV / m_projectile) * 1e2 # cm/s
                # v_targ_lab = np.sqrt(2.0 * Eth * eV / m_target ) * 1e2 #cm/s
                
                # reduced_mass =  m_projectile*m_target /(m_projectile + m_target)
                # v_relative = abs(v_proj_i - v_targ_lab) #cm/s
                # Ecm = 0.5*reduced_mass*(v_relative/1e2)**2/ eV


                #CS_reac = CS_table[reac_chnl]
                # f.write(f"{reaction_type_arr[ir]}, channel number = {reac_chnl} \n")
                # f.write(f'Ecm={Ecm} \n')
                # f.write(f'type E_cs_list={type(E_cs_list)} \n')
                
                # if reaction_type_arr[ir] == "rovib":
                #     # f.write("T_rovib_list = " + str(T_rovib_list) + "\n")
                #     # f.write("E_cs_list = " + str(E_cs_list) + "\n")
                #     CS_reac = CS_table[reac_chnl]
                #     sigma = interp_linear_2D(
                #         T_mc,
                #         Ecm,
                #         T_rovib_list,
                #         E_cs_list,
                #         CS_reac
                #     )


                # f.write("E_cs_list = " + str(E_cs_list) + "\n")
                
                CS_reac = CS_table[reac_chnl]
                #f.write("CS_reac = " + str(CS_reac) + "\n")
                # if iE==0:
                #     f.write(f'CS_reac={CS_reac} \n')

                sigma = interp_linear(
                    Ecm, 
                    E_cs_list,
                    CS_reac
                )

                #fcoll = v_relative * sigma * n_target
                fcoll = vi * sigma * n_target

            f_tot += fcoll
        

        frequency_tot_distrib_sum[iE] = f_tot

    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        # f.write(f"---------------- Nmc={i+1} \n")
        
        if i % 1000 == 0:
            print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]


        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0
            time =0 # average time after collisions time += \sum_r (1/nu_r) 

            while E_lab > E_cutoff:

                Ncoll += 1
                
                # #v_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2
                
                v_proj_lab = np.sqrt(2.0 * E_lab * eV / mH) * 1e2 # cm/s
                

                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = np.sqrt(1 - cos_theta_lab**2)

                #----- Sampling the thermal energy from a maxwellian distribution with a direct method
                R1, R2, R3 = np.random.rand(), np.random.rand(), np.random.rand()
                Eth_mc = -kb_cste*T_mc /eV * ( np.log(R1) + np.log(R2) * np.cos(2*np.pi*R3)**2 )

                # f.write(f"----- Ncoll={Ncoll} \n")
                # f.write(f"E_lab before collision={E_lab} \n")
                # f.write(f"E_th ={Eth_mc} \n")


                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    if reaction_type_arr[ir] == "photoionization":
                        # tau = tau_cste*tau0
                        # fcoll = 1/tau
                        fcoll = fcoll_ioniz
                    else:
                        
                        reac_chnl = reac_channel_arr[ir]
                        #f.write(f"{reac_chnl} \n")

                        ## Projectile
                        m_projectile = m_projectile_arr[ir]

                        ## Target 
                        m_target = m_target_arr[ir]
                        name_target = target_arr[ir]

                        # Find target to get its density
                        for i_species,species in enumerate(species_names):
                            if species == name_target:
                                i_target = i_species
                        #i_target = 2 #find_species_index(species_names, name_target)
                        n_target = densities[i_target]

                        ## velocity of the target
                        v_targ_lab_k = np.sqrt(2.0 * Eth_mc * eV / m_target ) * 1e2 #cm/s
                        v_targ_i_k = v_targ_lab_k*np.array([cos_theta_lab, sin_theta_lab, 0])
                        reduced_mass_k =  m_projectile*m_target /(m_projectile + m_target)
                        v_relative_k = (v_proj_i - v_targ_i_k) #cm/s
                            

                        Ecm = 0.5*reduced_mass_k*np.sum((v_relative_k/1e2)**2)/ eV

                        Ecm_target0 = E_lab * m_target / (m_target + m_projectile)
                        # f.write(f" Ecm: {Ecm} eV, Ecm_target0: {Ecm_target0} eV\n")
                        
                        #CS_reac = CS_table[reac_chnl]

                        # if reaction_type_arr[ir] == "rovib":
                        #     CS_reac = CS_table[reac_chnl]
                        #     ### sigma can be precomputed and then no need to do a 2D interpolation
                        #     sigma = interp_linear_2D(
                        #         T_mc,
                        #         Ecm,
                        #         T_rovib_list,
                        #         E_cs_list,
                        #         CS_reac
                        #     )
                        #     #f.write(f" sigma({Ecm}) = {sigma} \n")
                        
                            
                        CS_reac = CS_table[reac_chnl]
                        sigma = interp_linear(
                            Ecm,
                            E_cs_list,
                            CS_reac
                        )

                        fcoll =np.sqrt(np.sum(v_relative_k**2)) * sigma * n_target
                        #fcoll = v_proj_lab * sigma * n_target
                        # f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    
                    
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                #f.write(f"fcoll list : {fcoll_process_list}\n")
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    # f.write(f"f_tot<=0 : {f_tot:.2e}\n")
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot
                #f.write(f"Proba list : {proba_sum_cascade}\n")

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0


                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                # f.write(f"## Process choice : {reac_channel_arr[i_process]}\n")

                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])
                
                ### Define the photoionization time and choose if we should remove the particle
                """
                eta_6 = np.random.rand()
                tau_remove = tau_cste*tau0#tau_cste*1.5e6 #tau_cste/f_tot
                #time += np.log(eta_6/fcoll_process_list[i_process])
                time += eta_6*tau0 #--> tau0 in Panarese. Just to fit with them 
                r_l = np.random.rand() # random number 
                p_l = 1 - np.exp(-time/tau_remove) # probability to remove the particle after time t
                # f.write(f"##### Photoionization \n")
                # f.write(f"f_tot={f_tot:.2e}, tau_remove={tau_remove:.2e} s \n")
                # f.write(f"f_process={fcoll_process_list[i_process] }, time={time:.2e} \n")
                # f.write(f"r_l={r_l}, p_l={p_l} \n")

                if r_l < p_l:
                    # f.write(f"REMOVE H \n\n")
                    break
                """

                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
                    # f.write(f"XXXXXXX removes projectile XXXXXXX\n")
                    break

                if reaction_type_arr[i_process] == 'inelastic':
                    
                    dE_lab = threshold_arr[i_process]
                    #f.write(f"Inelastic collision\n")
    
                
                # elif reaction_type_arr[i_process] == 'rovib':
                #     dE_rovib_table = threshold_arr[i_process]
                #     #dE_lab = interp_linear_2D(T_mc,Ecm,T_rovib_list,E_cs_list,dE_rovib_table)
                #     dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_table)
                #     #f.write(f"!!!!!!!!!! T_mc,Ecm,dE_lab = {T_mc,Ecm,dE_lab} !!!!!!!!!!! \n")
                    
                elif reaction_type_arr[i_process] == 'elastic':
                    # f.write(f"Elastic collision\n")
                    reac_chnl = reac_channel_arr[i_process]
                    mass_projectile = m_projectile_arr[i_process]
                    mass_target = m_target_arr[i_process]
                    reduced_mass = mass_projectile*mass_target /(mass_projectile + mass_target)

                    v_targ_lab = np.sqrt(2.0 * Eth_mc * eV / mass_target ) * 1e2
                    v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])
                    v_relative = v_proj_i - v_targ_i #cm/s

                    V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                    c_proj_i = v_proj_i - V_cm
                    norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                    norm_cproj_f = norm_cproj_i # demonstration

                    # ---- Collision in the CM frame ----

                    ## Define the CM base coordinates in the lab frame
                    e1 = c_proj_i/norm_cproj_i
                    e3 = np.array([0, 0, 1])
                    e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                    ## Angle of the collision
                    r_theta = np.random.rand()
                    #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                    

                    Ecm_choice = 0.5*reduced_mass*np.sum((v_relative/1e2)**2)/ eV # general case
                    #Ecm_choice = E_lab * mass_target / (mass_target + mH) # Case of target is at rest
                    
                    #, {E_grid_array[reac_chnl]}, {r_grid_array[reac_chnl]}, {theta_inverse_array[reac_chnl]}")
                    theta_CM = Inverse_cdf_theta_numba(
                        Ecm_choice,
                        r_theta,
                        E_grid_array[reac_chnl], # we need to use the reaction channel index because those lists are created from the full reaction list
                        r_grid_array[reac_chnl],
                        theta_inverse_array[reac_chnl]
                    )

                    # f.write(f"Ecm_choice,r_theta, theta_CM = {Ecm_choice,r_theta, theta_CM*180/np.pi }\n")

                    phi_CM = 2*np.pi*np.random.rand()
                    
                    c_proj_f = norm_cproj_f*( np.cos(theta_CM)*e1 + np.sin(theta_CM)*np.cos(phi_CM)*e2 + np.sin(theta_CM)*np.sin(phi_CM)*e3) # lab frame

                    # ---- energy loss ----
                    #f.write(f"## Energy loss\n")
                    # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                    # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                    dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV
                    # f.write(f"E_lab={E_lab} \n")
                    # f.write(f"dE_lab= {dE_lab} \n")

                    # v_proj_f = c_proj_f + V_cm
                    # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                    #print(dE_lab)
                    
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth_mc + dE_lab
                        if E_secondary > Emin_secondary:
                            
                            E_Hf_list[n_active] = E_secondary
                            # f.write(f"## Secondary H produced : E_secondary={E_secondary} \n")
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                # f.write(f"## Energy loss : dE_lab = {dE_lab}\n")
                # f.write(f"E_lab after collision={E_lab} \n")
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                #f.write(f"idx_bin={idx_bin} \n")
                N_dist_list[idx_bin] += 1
                count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        if statistics:
            
            N_mean_list = np.zeros(bin_dist)
            N_var_list = np.zeros(bin_dist)

            n_sample = i + 1

            for b in range(bin_dist):

                # ---- compute mean ----
                s = 0.0
                for k in range(n_sample):
                    s += count_table[k, b]

                mean = s / n_sample
                N_mean_list[b] = mean

                # ---- compute variance ----
                if n_sample > 1:
                    s2 = 0.0
                    for k in range(n_sample):
                        diff = count_table[k, b] - mean
                        s2 += diff * diff

                    N_var_list[b] = s2 / n_sample
                else:
                    N_var_list[b] = 0.0

            for b in range(bin_dist):
                freq = frequency_tot_distrib_sum[b]
                if freq > 0.0 and np.isfinite(freq):
                    H_distrib_table[i, b] = (
                        W0 * N_mean_list[b] / (freq * dEi_list[b])
                    )
                    SE_distrib_table[i, b] = (
                        W0 / (freq * dEi_list[b])
                    ) * np.sqrt(N_var_list[b] / (i + 1))
            
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        freq = frequency_tot_distrib_sum[b]
        if freq > 0.0 and np.isfinite(freq):
            H_distrib[b] = (
                W0 * (N_dist_list[b] / N_mc) / (freq * dEi_list[b])
            )

    Yield = proba_sum_cascade/N_mc

    return ( E_dist_list,
        H_distrib,
        H_distrib_table,
        Yield,
        SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last,
        dEi_list
        #count_table
    )



@njit
def monte_carlo_testRovib_numba(dE_rovib_list, N_mc, E_dist_list, Eth, E_cutoff, dEi, E0_random_list, W0, species_names, densities, tables_list, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr , threshold_arr, size_max_secondaries=100000, statistics=False,  ):
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - tables_list : list of tables for cross sections and differential cross sections --> Usefull if ones want to change a cross section in its script to test
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    """

    T_mc = Eth*eV/kb_cste

    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    
    E_cs_list = tables_list[0]
    CS_table = tables_list[1]
    #CS_table = np.asarray(CS_table, dtype=np.float64)
    E_grid_array = tables_list[2]
    r_grid_array = tables_list[3]
    theta_inverse_array = tables_list[4]
    T_rovib_list = tables_list[5]


    # with open("simulation_log_test.txt", "w") as f:

    #     f.write("Monte Carlo Simulation Results\n")
    #     f.write("===============================\n\n")
    #     f.write("E_dist_list = " + str(E_dist_list) + "\n")
    #     f.write("bin_dist = " + str(bin_dist) + "\n\n")


    #     f.write(f"-------------------- precompute frequencies \n")
    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):
        #f.write(f"{iE+1}/{bin_dist} \n")

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2

        f_tot = 0.0
        
        for ir in range(len(reac_channel_arr)):
            reac_chnl = reac_channel_arr[ir]

            ## Projectile
            m_projectile = m_projectile_arr[ir]

            ## Target 
            m_target = m_target_arr[ir]
            name_target = target_arr[ir]

            # Find target to get its density
            for i_species,species in enumerate(species_names):
                if species == name_target:
                    i_target = i_species
            #i_target = 2 #find_species_index(species_names, name_target)
            n_target = densities[i_target]
            
            Ecm = Ei * m_target / (m_target + m_projectile)
            #CS_reac = CS_table[reac_chnl]
            # f.write(f"{reaction_type_arr[ir]}, channel number = {reac_chnl} \n")
            # f.write(f'Ecm={Ecm} \n')
            # f.write(f'type E_cs_list={type(E_cs_list)} \n')
            
            # if reaction_type_arr[ir] == "rovib":
            #     # f.write("T_rovib_list = " + str(T_rovib_list) + "\n")
            #     # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            #     CS_reac = CS_table[reac_chnl]
            #     sigma = interp_linear_2D(
            #         T_mc,
            #         Ecm,
            #         T_rovib_list,
            #         E_cs_list,
            #         CS_reac
            #     )

            # else:
            # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            
            CS_reac = CS_table[reac_chnl]
            #f.write("CS_reac = " + str(CS_reac) + "\n")
            # if iE==0:
            #     f.write(f'CS_reac={CS_reac} \n')

            sigma = interp_linear(
                Ecm,
                E_cs_list,
                CS_reac
            )


            fcoll = vi * sigma * n_target
            f_tot += fcoll
        

        frequency_tot_distrib_sum[iE] = f_tot

    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        # f.write(f"---------------- Nmc={i+1} \n")
        
        if i % 1000 == 0:
            print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]


        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0

            while E_lab > E_cutoff:

                Ncoll += 1
                # f.write(f"----- Ncoll={Ncoll} \n")
                # f.write(f"E_lab before collision={E_lab} \n")
                #v_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2
                
                v_proj_lab = np.sqrt(2.0 * E_lab * eV / mH) * 1e2 # cm/s
                

                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = np.sqrt(1 - cos_theta_lab**2)


                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    
                    reac_chnl = reac_channel_arr[ir]
                    #f.write(f"{reac_chnl} \n")

                    ## Projectile
                    m_projectile = m_projectile_arr[ir]

                    ## Target 
                    m_target = m_target_arr[ir]
                    name_target = target_arr[ir]

                    # Find target to get its density
                    for i_species,species in enumerate(species_names):
                        if species == name_target:
                            i_target = i_species
                    #i_target = 2 #find_species_index(species_names, name_target)
                    n_target = densities[i_target]

                    ## velocity of the target
                    v_targ_lab_k = np.sqrt(2.0 * Eth * eV / m_target ) * 1e2 #cm/s
                    v_targ_i_k = v_targ_lab_k*np.array([cos_theta_lab, sin_theta_lab, 0])
                    reduced_mass_k =  m_projectile*m_target /(m_projectile + m_target)
                    v_relative_k = (v_proj_i - v_targ_i_k) #cm/s
                        

                    Ecm = 0.5*reduced_mass_k*np.sum((v_relative_k/1e2)**2)/ eV

                    Ecm_target0 = E_lab * m_target / (m_target + m_projectile)
                    #f.write(f" Ecm: {Ecm} eV, Ecm_target0: {Ecm_target0} eV\n")
                    
                    #CS_reac = CS_table[reac_chnl]


                    # if reaction_type_arr[ir] == "rovib":
                    #     if len(dE_rovib_list) > 1:
                    #         CS_reac = CS_table[reac_chnl]
                    #         ### sigma can be precomputed and then no need to do a 2D interpolation
                    #         dE_ = interp_linear(
                    #                 Ecm,
                    #                 E_cs_list,
                    #                 CS_reac
                    #             )
                    #         #f.write(f" sigma({Ecm}) = {sigma} \n")

                    # else:
                    CS_reac = CS_table[reac_chnl]
                    sigma = interp_linear(
                        Ecm,
                        E_cs_list,
                        CS_reac
                    )
                    

                    fcoll =np.sqrt(np.sum(v_relative_k**2)) * sigma * n_target
                    #fcoll = v_proj_lab * sigma * n_target
                    # f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                #f.write(f"fcoll list : {fcoll_process_list}\n")
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    #f.write(f"f_tot<=0\n")
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot
                #f.write(f"Proba list : {proba_sum_cascade}\n")

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0


                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                #f.write(f"## Process choice : {reac_channel_arr[i_process]}\n")

                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])
                

                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
                    #f.write(f"removes projectile\n")
                    break

                if reaction_type_arr[i_process] == 'inelastic':
                    #f.write(f"Inelastic collision\n")
                    dE_lab = threshold_arr[i_process]
                
                elif reaction_type_arr[i_process] == 'rovib':
                    dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_list)
                    # if type(dE_rovib_list) == np.ndarray:
                    #     dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_list)
                    # else:
                    #     dE_lab = threshold_arr[i_process]
                    
                    # dE_rovib_table = threshold_arr[i_process]
                    # #dE_lab = interp_linear_2D(T_mc,Ecm,T_rovib_list,E_cs_list,dE_rovib_table)
                    # dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_table)
                    # f.write(f"!!!!!!!!!! T_mc,Ecm,dE_lab = {T_mc,Ecm,dE_lab} !!!!!!!!!!! \n")
                    
                elif reaction_type_arr[i_process] == 'elastic':
                    #f.write(f"Elastic collision\n")
                    reac_chnl = reac_channel_arr[i_process]
                    mass_projectile = m_projectile_arr[i_process]
                    mass_target = m_target_arr[i_process]
                    reduced_mass = mass_projectile*mass_target /(mass_projectile + mass_target)

                    v_targ_lab = np.sqrt(2.0 * Eth * eV / mass_target ) * 1e2 #cm/s
                    
                    v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])
                    v_relative = v_proj_i - v_targ_i #cm/s

                    V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                    c_proj_i = v_proj_i - V_cm
                    norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                    norm_cproj_f = norm_cproj_i # demonstration

                    # ---- Collision in the CM frame ----

                    ## Define the CM base coordinates in the lab frame
                    e1 = c_proj_i/norm_cproj_i
                    e3 = np.array([0, 0, 1])
                    e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                    ## Angle of the collision
                    r_theta = np.random.rand()
                    #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                    

                    Ecm_choice = 0.5*reduced_mass*np.sum((v_relative/1e2)**2)/ eV # general case
                    #Ecm_choice = E_lab * mass_target / (mass_target + mH) # Case of target is at rest
                    
                    #f.write(f"{Ecm_choice,r_theta}, {E_grid_array[reac_chnl]}, {r_grid_array[reac_chnl]}, {theta_inverse_array[reac_chnl]}")
                    theta_CM = Inverse_cdf_theta_numba(
                        Ecm_choice,
                        r_theta,
                        E_grid_array[reac_chnl], # we need to use the reaction channel index because those lists are created from the full reaction list
                        r_grid_array[reac_chnl],
                        theta_inverse_array[reac_chnl]
                    )

                    phi_CM = 2*np.pi*np.random.rand()
                    
                    c_proj_f = norm_cproj_f*( np.cos(theta_CM)*e1 + np.sin(theta_CM)*np.cos(phi_CM)*e2 + np.sin(theta_CM)*np.sin(phi_CM)*e3) # lab frame

                    #v_proj_f = c_proj_f + V_cm


                    # ---- energy loss ----
                    #f.write(f"## Energy loss\n")
                    # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                    # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                    dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV
                    
                    #dE_lab_test = 0.5*mass_projectile*(np.sum(v_proj_i**2) - np.sum(v_proj_f**2))*1e-4/eV
                    # f.write(f"e1={e1}, e2={e2}, e3={e3} \n")
                    # f.write(f"e1={np.sqrt(np.sum(e1**2))}, e2={np.sqrt(np.sum(e2**2))}, e3={np.sqrt(np.sum(e3**2))} \n")         
                    
                    # f.write(f"v_proj_i= {np.sqrt(2*E_lab*eV/m_projectile)*1e2} cm/s \n")
                    # f.write(f"||v_targ_i||= {v_targ_lab} cm/s \n")
                    # f.write(f"v_relative={v_relative} \n")
                    
                    # #f.write(f"E_lab={E_lab} \n")
                    # f.write(f"c_proj_i = {c_proj_i} \n")
                    # f.write(f"c_proj_f= {c_proj_f} \n")
                    # f.write(f"||c_proj_f||= {np.sqrt(np.sum(c_proj_f**2))} = ||c_proj_i||= {np.sqrt(np.sum(c_proj_i**2))} \n")
                    # f.write(f"V_cm = {V_cm} \n")
                    # f.write(f"{(c_proj_i-c_proj_f)*V_cm} \n")
                    # f.write(f"dE_lab_test= {dE_lab_test} \n")
                    
                    # f.write(f"dE_lab_Eth0= {dE_lab_Eth0} \n")

                    # f.write(f"dE_lab= {dE_lab} \n")

                    # v_proj_f = c_proj_f + V_cm
                    # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                    #print(dE_lab)
                    
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth + dE_lab
                        if E_secondary > E_cutoff:
                            
                            E_Hf_list[n_active] = E_secondary
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                # f.write(f"## Energy loss : dE_lab = {dE_lab}\n")
                # f.write(f"E_lab after collision={E_lab} \n")
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                #f.write(f"idx_bin={idx_bin} \n")
                N_dist_list[idx_bin] += 1
                count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        if statistics:
            
            N_mean_list = np.zeros(bin_dist)
            N_var_list = np.zeros(bin_dist)

            n_sample = i + 1

            for b in range(bin_dist):

                # ---- compute mean ----
                s = 0.0
                for k in range(n_sample):
                    s += count_table[k, b]

                mean = s / n_sample
                N_mean_list[b] = mean

                # ---- compute variance ----
                if n_sample > 1:
                    s2 = 0.0
                    for k in range(n_sample):
                        diff = count_table[k, b] - mean
                        s2 += diff * diff

                    N_var_list[b] = s2 / n_sample
                else:
                    N_var_list[b] = 0.0

            for b in range(bin_dist):
                freq = frequency_tot_distrib_sum[b]
                if freq > 0.0 and np.isfinite(freq):
                    H_distrib_table[i, b] = (
                        W0 * N_mean_list[b] / (freq * dEi)
                    )
                    SE_distrib_table[i, b] = (
                        W0 / (freq * dEi)
                    ) * np.sqrt(N_var_list[b] / (i + 1))
            
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        freq = frequency_tot_distrib_sum[b]
        if freq > 0.0 and np.isfinite(freq):
            H_distrib[b] = (
                W0 * (N_dist_list[b] / N_mc) / (freq * dEi)
            )

    Yield = proba_sum_cascade/N_mc

    return (
        H_distrib,
        H_distrib_table,
        Yield,
        SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last,
        #count_table
    )


## This Monte-Carlo stop tracking a particle when it is removed by photo-ionization for instance
#@njit
def monte_carlo_remove_numba(N_mc, E_dist_list, Eth, E_cutoff, dEi, E0_random_list, W0, species_names, densities, tables_list, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr , threshold_arr, size_max_secondaries=100000, statistics=False, tau_cste=482):
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - tables_list : list of tables for cross sections and differential cross sections --> Usefull if ones want to change a cross section in its script to test
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    - tau_cste = tau/tau0 with (tau=time of reaction to remove H) and (tau0=typicall time between 2 collisions)
    """

    T_mc = Eth*eV/kb_cste

    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    
    E_cs_list = tables_list[0]
    CS_table = tables_list[1]
    #CS_table = np.asarray(CS_table, dtype=np.float64)
    E_grid_array = tables_list[2]
    r_grid_array = tables_list[3]
    theta_inverse_array = tables_list[4]
    T_rovib_list = tables_list[5]


    # with open("simulation_log_test.txt", "w") as f:

    #     f.write("Monte Carlo Simulation Results\n")
    #     f.write("===============================\n\n")
    #     f.write("E_dist_list = " + str(E_dist_list) + "\n")
    #     f.write("bin_dist = " + str(bin_dist) + "\n\n")
    #     f.write(f"-------------------- precompute frequencies \n")


    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):
        #f.write(f"{iE+1}/{bin_dist} \n")

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2

        f_tot = 0.0
        
        for ir in range(len(reac_channel_arr)):
            reac_chnl = reac_channel_arr[ir]

            ## Projectile
            m_projectile = m_projectile_arr[ir]

            ## Target 
            m_target = m_target_arr[ir]
            name_target = target_arr[ir]

            # Find target to get its density
            for i_species,species in enumerate(species_names):
                if species == name_target:
                    i_target = i_species
            #i_target = 2 #find_species_index(species_names, name_target)
            n_target = densities[i_target]
            
            Ecm = Ei * m_target / (m_target + m_projectile)
            #CS_reac = CS_table[reac_chnl]
            # f.write(f"{reaction_type_arr[ir]}, channel number = {reac_chnl} \n")
            # f.write(f'Ecm={Ecm} \n')
            # f.write(f'type E_cs_list={type(E_cs_list)} \n')
            
            # if reaction_type_arr[ir] == "rovib":
            #     # f.write("T_rovib_list = " + str(T_rovib_list) + "\n")
            #     # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            #     CS_reac = CS_table[reac_chnl]
            #     sigma = interp_linear_2D(
            #         T_mc,
            #         Ecm,
            #         T_rovib_list,
            #         E_cs_list,
            #         CS_reac
            #     )

            # else:
            # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            
            CS_reac = CS_table[reac_chnl]
            #f.write("CS_reac = " + str(CS_reac) + "\n")
            # if iE==0:
            #     f.write(f'CS_reac={CS_reac} \n')

            sigma = interp_linear(
                Ecm,
                E_cs_list,
                CS_reac
            )


            fcoll = vi * sigma * n_target
            f_tot += fcoll
        

        frequency_tot_distrib_sum[iE] = f_tot

    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        # f.write(f"---------------- Nmc={i+1} \n")
        
        if i % 1000 == 0:
            print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]


        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0
            time =0 # average time after collisions time += \sum_r (1/nu_r)

            while E_lab > E_cutoff:

                Ncoll += 1
                # f.write(f"----- Ncoll={Ncoll} \n")
                # f.write(f"E_lab before collision={E_lab} \n")
                #v_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2
                
                v_proj_lab = np.sqrt(2.0 * E_lab * eV / mH) * 1e2 # cm/s
                

                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = np.sqrt(1 - cos_theta_lab**2)


                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    
                    reac_chnl = reac_channel_arr[ir]
                    #f.write(f"{reac_chnl} \n")

                    ## Projectile
                    m_projectile = m_projectile_arr[ir]

                    ## Target 
                    m_target = m_target_arr[ir]
                    name_target = target_arr[ir]

                    # Find target to get its density
                    for i_species,species in enumerate(species_names):
                        if species == name_target:
                            i_target = i_species
                    #i_target = 2 #find_species_index(species_names, name_target)
                    n_target = densities[i_target]

                    ## velocity of the target
                    v_targ_lab_k = np.sqrt(2.0 * Eth * eV / m_target ) * 1e2 #cm/s
                    v_targ_i_k = v_targ_lab_k*np.array([cos_theta_lab, sin_theta_lab, 0])
                    reduced_mass_k =  m_projectile*m_target /(m_projectile + m_target)
                    v_relative_k = (v_proj_i - v_targ_i_k) #cm/s
                        

                    Ecm = 0.5*reduced_mass_k*np.sum((v_relative_k/1e2)**2)/ eV

                    Ecm_target0 = E_lab * m_target / (m_target + m_projectile)
                    #f.write(f" Ecm: {Ecm} eV, Ecm_target0: {Ecm_target0} eV\n")
                    
                    #CS_reac = CS_table[reac_chnl]

                    # if reaction_type_arr[ir] == "rovib":
                    #     CS_reac = CS_table[reac_chnl]
                    #     ### sigma can be precomputed and then no need to do a 2D interpolation
                    #     sigma = interp_linear_2D(
                    #         T_mc,
                    #         Ecm,
                    #         T_rovib_list,
                    #         E_cs_list,
                    #         CS_reac
                    #     )
                    #     #f.write(f" sigma({Ecm}) = {sigma} \n")

                    # else:
                    CS_reac = CS_table[reac_chnl]
                    sigma = interp_linear(
                        Ecm,
                        E_cs_list,
                        CS_reac
                    )

                    fcoll =np.sqrt(np.sum(v_relative_k**2)) * sigma * n_target
                    #fcoll = v_proj_lab * sigma * n_target
                    # f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                #f.write(f"fcoll list : {fcoll_process_list}\n")
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    #f.write(f"f_tot<=0\n")
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot
                #f.write(f"Proba list : {proba_sum_cascade}\n")

                

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0

                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                
                # f.write(f"## Process choice : {reac_channel_arr[i_process]}\n")

                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])
                
                ### Define the photoionization time and choose if we should remove the particle
                tau_remove = tau_cste/f_tot
                time += 1/fcoll_process_list[i_process] 
                r_l = np.random.rand() # random number 
                p_l = 1 - np.exp(-time/tau_remove) # probability to remove the particle after time t
                
                if r_l < p_l:
                    break


                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
                    #f.write(f"removes projectile\n")
                    break

                if reaction_type_arr[i_process] == 'inelastic':
                    
                    dE_lab = threshold_arr[i_process]
                    #f.write(f"Inelastic collision\n")
    
                
                # elif reaction_type_arr[i_process] == 'rovib':
                #     dE_rovib_table = threshold_arr[i_process]
                #     #dE_lab = interp_linear_2D(T_mc,Ecm,T_rovib_list,E_cs_list,dE_rovib_table)
                #     dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_table)
                #     #f.write(f"!!!!!!!!!! T_mc,Ecm,dE_lab = {T_mc,Ecm,dE_lab} !!!!!!!!!!! \n")
                    
                elif reaction_type_arr[i_process] == 'elastic':
                    #f.write(f"Elastic collision\n")
                    reac_chnl = reac_channel_arr[i_process]
                    mass_projectile = m_projectile_arr[i_process]
                    mass_target = m_target_arr[i_process]
                    reduced_mass = mass_projectile*mass_target /(mass_projectile + mass_target)

                    v_targ_lab = np.sqrt(2.0 * Eth * eV / mass_target ) * 1e2
                    v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])
                    v_relative = v_proj_i - v_targ_i #cm/s

                    V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                    c_proj_i = v_proj_i - V_cm
                    norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                    norm_cproj_f = norm_cproj_i # demonstration

                    # ---- Collision in the CM frame ----

                    ## Define the CM base coordinates in the lab frame
                    e1 = c_proj_i/norm_cproj_i
                    e3 = np.array([0, 0, 1])
                    e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                    ## Angle of the collision
                    r_theta = np.random.rand()
                    #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                    

                    Ecm_choice = 0.5*reduced_mass*np.sum((v_relative/1e2)**2)/ eV # general case
                    #Ecm_choice = E_lab * mass_target / (mass_target + mH) # Case of target is at rest
                    
                    #f.write(f"{Ecm_choice,r_theta}, {E_grid_array[reac_chnl]}, {r_grid_array[reac_chnl]}, {theta_inverse_array[reac_chnl]}")
                    theta_CM = Inverse_cdf_theta_numba(
                        Ecm_choice,
                        r_theta,
                        E_grid_array[reac_chnl], # we need to use the reaction channel index because those lists are created from the full reaction list
                        r_grid_array[reac_chnl],
                        theta_inverse_array[reac_chnl]
                    )

                    phi_CM = 2*np.pi*np.random.rand()
                    
                    c_proj_f = norm_cproj_f*( np.cos(theta_CM)*e1 + np.sin(theta_CM)*np.cos(phi_CM)*e2 + np.sin(theta_CM)*np.sin(phi_CM)*e3) # lab frame

                    # ---- energy loss ----
                    #f.write(f"## Energy loss\n")
                    # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                    # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                    dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV
                    #f.write(f"E_lab={E_lab} \n")
                    #f.write(f"dE_lab= {dE_lab} \n")

                    # v_proj_f = c_proj_f + V_cm
                    # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                    #print(dE_lab)
                    
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth + dE_lab
                        if E_secondary > E_cutoff:
                            
                            E_Hf_list[n_active] = E_secondary
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                # f.write(f"## Energy loss : dE_lab = {dE_lab}\n")
                # f.write(f"E_lab after collision={E_lab} \n")
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                #f.write(f"idx_bin={idx_bin} \n")
                N_dist_list[idx_bin] += 1
                count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        if statistics:
            
            N_mean_list = np.zeros(bin_dist)
            N_var_list = np.zeros(bin_dist)

            n_sample = i + 1

            for b in range(bin_dist):

                # ---- compute mean ----
                s = 0.0
                for k in range(n_sample):
                    s += count_table[k, b]

                mean = s / n_sample
                N_mean_list[b] = mean

                # ---- compute variance ----
                if n_sample > 1:
                    s2 = 0.0
                    for k in range(n_sample):
                        diff = count_table[k, b] - mean
                        s2 += diff * diff

                    N_var_list[b] = s2 / n_sample
                else:
                    N_var_list[b] = 0.0

            for b in range(bin_dist):
                freq = frequency_tot_distrib_sum[b]
                if freq > 0.0 and np.isfinite(freq):
                    H_distrib_table[i, b] = (
                        W0 * N_mean_list[b] / (freq * dEi)
                    )
                    SE_distrib_table[i, b] = (
                        W0 / (freq * dEi)
                    ) * np.sqrt(N_var_list[b] / (i + 1))
            
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        freq = frequency_tot_distrib_sum[b]
        if freq > 0.0 and np.isfinite(freq):
            H_distrib[b] = (
                W0 * (N_dist_list[b] / N_mc) / (freq * dEi)
            )

    Yield = proba_sum_cascade/N_mc

    return (
        H_distrib,
        H_distrib_table,
        Yield,
        SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last,
        #count_table
    )



@njit
def monte_carlo_test_numba(N_mc, E_dist_list, Eth, E_cutoff, dEi, E0_random_list, W0, species_names, densities, tables_list, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr , threshold_arr, size_max_secondaries=100000, statistics=False, tau_cste=482):
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - tables_list : list of tables for cross sections and differential cross sections --> Usefull if ones want to change a cross section in its script to test
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    - tau_cste = tau/tau0 with (tau=time of reaction to remove H) and (tau0=typicall time between 2 collisions)
    
    """

    T_mc = Eth*eV/kb_cste

    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    
    E_cs_list = tables_list[0]
    CS_table = tables_list[1]
    #CS_table = np.asarray(CS_table, dtype=np.float64)
    E_grid_array = tables_list[2]
    r_grid_array = tables_list[3]
    theta_inverse_array = tables_list[4]
    T_rovib_list = tables_list[5]


    # with open("simulation_log_test.txt", "w") as f:

    #     f.write("Monte Carlo Simulation Results\n")
    #     f.write("===============================\n\n")
    #     f.write("E_dist_list = " + str(E_dist_list) + "\n")
    #     f.write("bin_dist = " + str(bin_dist) + "\n\n")
    #     f.write(f"-------------------- precompute frequencies \n")

    

    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):
        #f.write(f"{iE+1}/{bin_dist} \n")

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2

        f_tot = 0.0
        
        for ir in range(len(reac_channel_arr)):
            reac_chnl = reac_channel_arr[ir]

            ## Projectile
            m_projectile = m_projectile_arr[ir]

            ## Target 
            m_target = m_target_arr[ir]
            name_target = target_arr[ir]

            # Find target to get its density
            for i_species,species in enumerate(species_names):
                if species == name_target:
                    i_target = i_species
            #i_target = 2 #find_species_index(species_names, name_target)
            n_target = densities[i_target]
            
            Ecm = Ei * m_target / (m_target + m_projectile)
            #CS_reac = CS_table[reac_chnl]
            # f.write(f"{reaction_type_arr[ir]}, channel number = {reac_chnl} \n")
            # f.write(f'Ecm={Ecm} \n')
            # f.write(f'type E_cs_list={type(E_cs_list)} \n')
            
            # if reaction_type_arr[ir] == "rovib":
            #     # f.write("T_rovib_list = " + str(T_rovib_list) + "\n")
            #     # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            #     CS_reac = CS_table[reac_chnl]
            #     sigma = interp_linear_2D(
            #         T_mc,
            #         Ecm,
            #         T_rovib_list,
            #         E_cs_list,
            #         CS_reac
            #     )

            # else:
            # f.write("E_cs_list = " + str(E_cs_list) + "\n")
            
            CS_reac = CS_table[reac_chnl]
            #f.write("CS_reac = " + str(CS_reac) + "\n")
            # if iE==0:
            #     f.write(f'CS_reac={CS_reac} \n')

            sigma = interp_linear(
                Ecm,
                E_cs_list,
                CS_reac
            )


            fcoll = vi * sigma * n_target
            f_tot += fcoll
        

        frequency_tot_distrib_sum[iE] = f_tot

    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        # f.write(f"---------------- Nmc={i+1} \n")
        
        if i % 1000 == 0:
            print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]


        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0
            time =0 # average time after collisions time += \sum_r (1/nu_r) 

            while E_lab > E_cutoff:

                Ncoll += 1
                # f.write(f"----- Ncoll={Ncoll} \n")
                # f.write(f"E_lab before collision={E_lab} \n")
                #v_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2
                
                v_proj_lab = np.sqrt(2.0 * E_lab * eV / mH) * 1e2 # cm/s
                

                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = np.sqrt(1 - cos_theta_lab**2)


                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    
                    reac_chnl = reac_channel_arr[ir]
                    #f.write(f"{reac_chnl} \n")

                    ## Projectile
                    m_projectile = m_projectile_arr[ir]

                    ## Target 
                    m_target = m_target_arr[ir]
                    name_target = target_arr[ir]

                    # Find target to get its density
                    for i_species,species in enumerate(species_names):
                        if species == name_target:
                            i_target = i_species
                    #i_target = 2 #find_species_index(species_names, name_target)
                    n_target = densities[i_target]

                    ## velocity of the target
                    v_targ_lab_k = np.sqrt(2.0 * Eth * eV / m_target ) * 1e2 #cm/s
                    v_targ_i_k = v_targ_lab_k*np.array([cos_theta_lab, sin_theta_lab, 0])
                    reduced_mass_k =  m_projectile*m_target /(m_projectile + m_target)
                    v_relative_k = (v_proj_i - v_targ_i_k) #cm/s
                        

                    Ecm = 0.5*reduced_mass_k*np.sum((v_relative_k/1e2)**2)/ eV

                    Ecm_target0 = E_lab * m_target / (m_target + m_projectile)
                    #f.write(f" Ecm: {Ecm} eV, Ecm_target0: {Ecm_target0} eV\n")
                    
                    #CS_reac = CS_table[reac_chnl]

                    # if reaction_type_arr[ir] == "rovib":
                    #     CS_reac = CS_table[reac_chnl]
                    #     ### sigma can be precomputed and then no need to do a 2D interpolation
                    #     sigma = interp_linear_2D(
                    #         T_mc,
                    #         Ecm,
                    #         T_rovib_list,
                    #         E_cs_list,
                    #         CS_reac
                    #     )
                    #     #f.write(f" sigma({Ecm}) = {sigma} \n")

                    # else:
                    CS_reac = CS_table[reac_chnl]
                    sigma = interp_linear(
                        Ecm,
                        E_cs_list,
                        CS_reac
                    )

                    fcoll =np.sqrt(np.sum(v_relative_k**2)) * sigma * n_target
                    #fcoll = v_proj_lab * sigma * n_target
                    # f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                #f.write(f"fcoll list : {fcoll_process_list}\n")
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    #f.write(f"f_tot<=0\n")
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot
                #f.write(f"Proba list : {proba_sum_cascade}\n")

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0


                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                # f.write(f"## Process choice : {reac_channel_arr[i_process]}\n")

                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])

                ### Define the photoionization time and choose if we should remove the particle
                tau_remove = tau_cste/f_tot
                time += 1/fcoll_process_list[i_process] 
                r_l = np.random.rand() # random number 
                p_l = 1 - np.exp(-time/tau_remove) # probability to remove the particle after time t
                
                if r_l < p_l:
                    break

                
                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
                    #f.write(f"removes projectile\n")
                    break

                if reaction_type_arr[i_process] == 'inelastic':
                    
                    dE_lab = threshold_arr[i_process]
                    #f.write(f"Inelastic collision\n")
    
                
                # elif reaction_type_arr[i_process] == 'rovib':
                #     dE_rovib_table = threshold_arr[i_process]
                #     #dE_lab = interp_linear_2D(T_mc,Ecm,T_rovib_list,E_cs_list,dE_rovib_table)
                #     dE_lab = interp_linear(Ecm,E_cs_list,dE_rovib_table)
                #     #f.write(f"!!!!!!!!!! T_mc,Ecm,dE_lab = {T_mc,Ecm,dE_lab} !!!!!!!!!!! \n")
                    
                elif reaction_type_arr[i_process] == 'elastic':
                    #f.write(f"Elastic collision\n")
                    reac_chnl = reac_channel_arr[i_process]
                    mass_projectile = m_projectile_arr[i_process]
                    mass_target = m_target_arr[i_process]
                    reduced_mass = mass_projectile*mass_target /(mass_projectile + mass_target)

                    v_targ_lab = np.sqrt(2.0 * Eth * eV / mass_target ) * 1e2
                    v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])
                    v_relative = v_proj_i - v_targ_i #cm/s

                    V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                    c_proj_i = v_proj_i - V_cm
                    norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                    norm_cproj_f = norm_cproj_i # demonstration

                    # ---- Collision in the CM frame ----

                    ## Define the CM base coordinates in the lab frame
                    e1 = c_proj_i/norm_cproj_i
                    e3 = np.array([0, 0, 1])
                    e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                    ## Angle of the collision
                    r_theta = np.random.rand()
                    #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                    

                    Ecm_choice = 0.5*reduced_mass*np.sum((v_relative/1e2)**2)/ eV # general case
                    #Ecm_choice = E_lab * mass_target / (mass_target + mH) # Case of target is at rest
                    
                    #f.write(f"{Ecm_choice,r_theta}, {E_grid_array[reac_chnl]}, {r_grid_array[reac_chnl]}, {theta_inverse_array[reac_chnl]}")
                    theta_CM = Inverse_cdf_theta_numba(
                        Ecm_choice,
                        r_theta,
                        E_grid_array[reac_chnl], # we need to use the reaction channel index because those lists are created from the full reaction list
                        r_grid_array[reac_chnl],
                        theta_inverse_array[reac_chnl]
                    )

                    phi_CM = 2*np.pi*np.random.rand()
                    
                    c_proj_f = norm_cproj_f*( np.cos(theta_CM)*e1 + np.sin(theta_CM)*np.cos(phi_CM)*e2 + np.sin(theta_CM)*np.sin(phi_CM)*e3) # lab frame

                    # ---- energy loss ----
                    #f.write(f"## Energy loss\n")
                    # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                    # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                    dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV
                    #f.write(f"E_lab={E_lab} \n")
                    #f.write(f"dE_lab= {dE_lab} \n")

                    # v_proj_f = c_proj_f + V_cm
                    # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                    #print(dE_lab)
                    
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth + dE_lab
                        if E_secondary > E_cutoff:
                            
                            E_Hf_list[n_active] = E_secondary
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                # f.write(f"## Energy loss : dE_lab = {dE_lab}\n")
                # f.write(f"E_lab after collision={E_lab} \n")
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                #f.write(f"idx_bin={idx_bin} \n")
                N_dist_list[idx_bin] += 1
                count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        if statistics:
            
            N_mean_list = np.zeros(bin_dist)
            N_var_list = np.zeros(bin_dist)

            n_sample = i + 1

            for b in range(bin_dist):

                # ---- compute mean ----
                s = 0.0
                for k in range(n_sample):
                    s += count_table[k, b]

                mean = s / n_sample
                N_mean_list[b] = mean

                # ---- compute variance ----
                if n_sample > 1:
                    s2 = 0.0
                    for k in range(n_sample):
                        diff = count_table[k, b] - mean
                        s2 += diff * diff

                    N_var_list[b] = s2 / n_sample
                else:
                    N_var_list[b] = 0.0

            for b in range(bin_dist):
                freq = frequency_tot_distrib_sum[b]
                if freq > 0.0 and np.isfinite(freq):
                    H_distrib_table[i, b] = (
                        W0 * N_mean_list[b] / (freq * dEi)
                    )
                    SE_distrib_table[i, b] = (
                        W0 / (freq * dEi)
                    ) * np.sqrt(N_var_list[b] / (i + 1))
            
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        freq = frequency_tot_distrib_sum[b]
        if freq > 0.0 and np.isfinite(freq):
            H_distrib[b] = (
                W0 * (N_dist_list[b] / N_mc) / (freq * dEi)
            )

    Yield = proba_sum_cascade/N_mc

    return ( 
        H_distrib,
        H_distrib_table,
        Yield,
        SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last
        #count_table
    )


@njit
def monte_carlo_numba_neutrons(N_mc, E_dist_list, Eth, E_cutoff, dEi, E0_random_list, W0, species_names, densities, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, size_max_secondaries=100000):
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    """
    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2
        
    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        
        if i % 1000 == 0:
            print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]

        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0

            while E_lab > E_cutoff:
                
                Ncoll += 1
                v_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2

                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    reac_chnl = reac_channel_arr[ir]

                    ## Projectile
                    m_projectile = m_projectile_arr[ir]

                    ## Target 
                    m_target = m_target_arr[ir]
                    name_target = target_arr[ir]
                    # Find target to get its density
                    for i_species,species in enumerate(species_names):
                        if species == name_target:
                            i_target = i_species
                    #i_target = 2 #find_species_index(species_names, name_target)
                    n_target = densities[i_target]
                    
                    Ecm = E_lab * m_target / (m_target + m_projectile)
                    
                    sigma = 1
                    fcoll = v_lab * sigma * n_target
                    #f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0

                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])
                

                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
    
                    break

                
                reac_chnl = reac_channel_arr[i_process]
                mass_projectile = m_projectile_arr[i_process]
                mass_target = m_target_arr[i_process]

                v_proj_lab = np.sqrt(2.0 * E_lab * eV / mass_projectile) * 1e2 # cm/s
                v_targ_lab = np.sqrt(2.0 * Eth * eV / mass_target) * 1e2

                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = 1 - cos_theta_lab**2

                v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])

                V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                c_proj_i = v_proj_i - V_cm
                norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                norm_cproj_f = norm_cproj_i # demonstration

                # ---- Collision in the CM frame ----

                ## Define the CM base coordinates in the lab frame
                e1 = c_proj_i/norm_cproj_i
                e3 = np.array([0, 0, 1])
                e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                # Angle of the collision
                r_theta = np.random.rand()
                #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                Ecm_choice = E_lab * mass_target / (mass_target + mH)

                # theta_CM = Inverse_cdf_theta_numba(
                #     Ecm_choice,
                #     r_theta,
                #     E_grid_array[reac_chnl], # we need to use the reaction channel index because those lists are created from the full reaction list
                #     r_grid_array[reac_chnl],
                #     theta_inverse_array[reac_chnl]
                # )

                cos_theta_CM = 2.0 * np.random.rand() - 1.0
                sin_theta_CM = 1 - cos_theta_CM**2

                phi_CM = 2*np.pi*np.random.rand()
                
                c_proj_f = norm_cproj_f*( cos_theta_CM*e1 + sin_theta_CM*np.cos(phi_CM)*e2 + sin_theta_CM*np.sin(phi_CM)*e3) # lab frame

                # ---- energy loss ----
                # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV
                # v_proj_f = c_proj_f + V_cm
                # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                #print(dE_lab)
                
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth + dE_lab
                        if E_secondary > E_cutoff:
                            
                            E_Hf_list[n_active] = E_secondary
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                N_dist_list[idx_bin] += 1
                #count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot
                

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        """
        N_mean_list = np.zeros(bin_dist)
        N_var_list = np.zeros(bin_dist)

        n_sample = i + 1

        for b in range(bin_dist):

            # ---- compute mean ----
            s = 0.0
            for k in range(n_sample):
                s += count_table[k, b]

            mean = s / n_sample
            N_mean_list[b] = mean

            # ---- compute variance ----
            if n_sample > 1:
                s2 = 0.0
                for k in range(n_sample):
                    diff = count_table[k, b] - mean
                    s2 += diff * diff

                N_var_list[b] = s2 / n_sample
            else:
                N_var_list[b] = 0.0

        for b in range(bin_dist):
            freq = frequency_tot_distrib_sum[b]
            if freq > 0.0 and np.isfinite(freq):
                H_distrib_table[i, b] = (
                    W0 * N_mean_list[b] / (freq * dEi)
                )
                SE_distrib_table[i, b] = (
                    W0 / (freq * dEi)
                ) * np.sqrt(N_var_list[b] / (i + 1))
        """
        
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        H_distrib[b] = ((N_dist_list[b] / N_mc) / ( dEi)
        )

    Yield = proba_sum_cascade/N_mc

    return (
        H_distrib,
        #H_distrib_table,
        Yield,
        #SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last,
        #count_table
    )



@njit
def XS_cste_fct_numba(E_cm, sigma_cste, Ea):
    if E_cm >= Ea:
        return sigma_cste
    else:
        return 0.0

@njit
def monte_carlo_numba_HG_XSelascste_XSchemcste(g, sigma_cste, Ea,N_mc, E_dist_list, Eth, E_cutoff, dEi, E0_random_list, W0, species_names, densities, tables_list, reac_channel_arr, m_projectile_arr , target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, size_max_secondaries=100000, statistics=False):
    """
    Monte Carlo simulation of H atom energy distribution.
    Inputs:
    - N_mc : number of Monte Carlo simulations = Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!! Number of H atoms launch at the energy E0 (initial energy of the H atom) !!!!
    - E_dist_list (len(E_dist_list)) : list of energy bins for the distribution [eV]
    - E0_random_list (len(N_mc)) : initial energy of the H atom [eV] !!!! List of N_mc values !!!!
    - W0 : production rate of H atoms [cm^-3 s^-1]
    - process_list : list of processes (reactions)
    - densities : list of densities for each target species [cm^-3]
    - reac_channel_arr, projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr : arrays of reaction (size = Nreaction) parameters corresponding to the processes we want to include in the simulation
    - Eth : threshold energy for thermalization [eV]
    - E_cutoff : energy below which the H atom is considered thermalized [eV]
    - dEi : width of the energy bins [eV]
    - size_max_secondaries : maximum number of secondary H atoms to track
    """
    nb_process = len(reac_channel_arr)
    bin_dist = len(E_dist_list)

    # -------------------------
    # Storage arrays
    # -------------------------
    Ncoll_th = np.zeros(N_mc)
    relax_time_list = np.zeros(N_mc)
    E_last = np.zeros(N_mc)

    N_dist_list = np.zeros(bin_dist)
    frequency_tot_distrib_sum = np.zeros(bin_dist)

    count_table = np.zeros((N_mc, bin_dist))
    H_distrib_table = np.zeros((N_mc, bin_dist))
    SE_distrib_table = np.zeros((N_mc, bin_dist))

    
    E_cs_list = tables_list[0]
    CS_table = tables_list[1]
    #CS_table = np.asarray(CS_table, dtype=np.float64)
    E_grid_array = tables_list[2]
    r_grid_array = tables_list[3]
    theta_inverse_array = tables_list[4]
    

    # with open("simulation_log_test.txt", "w") as f:
    #     f.write("Monte Carlo Simulation Results\n")
    #     f.write("===============================\n\n")
    #     f.write("E_dist_list = " + str(E_dist_list) + "\n")
    #     f.write("bin_dist = " + str(bin_dist) + "\n\n")

    
    #     # =====================================================
    #     # Precompute total frequency per energy bin
    #     # =====================================================
    #     f.write(f"-------------------- precompute frequencies \n")
    
    # =====================================================
    # Precompute total frequency per energy bin
    # =====================================================
    for iE in range(bin_dist):

        Ei = E_dist_list[iE]
        
        vi = np.sqrt(2.0 * Ei * eV / mH) * 1e2

        f_tot = 0.0
        
        for ir in range(len(reac_channel_arr)):
            reac_chnl = reac_channel_arr[ir]

            ## Projectile
            m_projectile = m_projectile_arr[ir]

            ## Target 
            m_target = m_target_arr[ir]
            name_target = target_arr[ir]
            

            # Find target to get its density
            for i_species,species in enumerate(species_names):
                if species == name_target:
                    i_target = i_species
            #i_target = 2 #find_species_index(species_names, name_target)
            n_target = densities[i_target]
            
            Ecm = Ei * m_target / (m_target + m_projectile)
            #CS_reac = CS_table[reac_chnl, :]
            if name_target=="h2o":
                CS_reac = CS_table[reac_chnl, :]
                sigma = interp_linear(
                    Ecm,
                    E_cs_list,
                    CS_reac
                )
                #sigma = XS_cste_fct_numba(Ecm, sigma_cste, Ea,)
                #f.write(f"sigma_h2o ({Ei})= {sigma}\n")
            if name_target=="x_test":
                #sigma =  4e-15
                v0=3.75*1e5 # cm/s
                sigma = 17.5e-16*(vi/v0)**0.9

            fcoll = vi * sigma * n_target
            f_tot += fcoll
        

        frequency_tot_distrib_sum[iE] = f_tot

    # =====================================================
    # Monte Carlo loop
    # =====================================================
    proba_sum_cascade = np.zeros(nb_process) # Sum of proba for each channel when introducing an H atom with energy E0 (secondaries taken into account)
    
    for i in range(N_mc):
        #f.write(f"\n ---------------- Nmc = {i} / {N_mc}")
        
        # if i % 1000 == 0:
        #     print(f"Step={i} / {N_mc}")

        E_Hf_list = np.zeros(size_max_secondaries)
        n_active = 1
        E_Hf_list[0] = E0_random_list[i]


        while n_active > 0:

            # find lowest energy particle
            idx_min = 0
            Emin = E_Hf_list[0]
            for k in range(1, n_active):
                if E_Hf_list[k] < Emin:
                    Emin = E_Hf_list[k]
                    idx_min = k

            E_lab = Emin

            # remove it
            E_Hf_list[idx_min] = E_Hf_list[n_active - 1]
            n_active -= 1

            Ncoll = 0
            relax_time = 0.0

            while E_lab > E_cutoff:

                Ncoll += 1
                #f.write(f"\n --- Ncoll = {Ncoll},  Elab = {E_lab}")
                v_proj_lab = np.sqrt(2.0 * E_lab * 1.6e-19 / mH) * 1e2
                # ---- Velocities in lab frame before collision
                v_proj_i = v_proj_lab*np.array([1, 0, 0])
                cos_theta_lab = 2.0 * np.random.rand() - 1.0 # isotropic 
                sin_theta_lab = np.sqrt(1 - cos_theta_lab**2)
                #f.write(f"\n - cos_theta_lab = {cos_theta_lab}, sin_theta_lab = {sin_theta_lab}")


                # ---- collision frequencies ----
                f_tot = 0.0
                fcoll_process_list = np.zeros(nb_process)

                for ir in range(len(reac_channel_arr)):
                    reac_chnl = reac_channel_arr[ir]

                    ## Projectile
                    m_projectile = m_projectile_arr[ir]

                    ## Target 
                    m_target = m_target_arr[ir]
                    name_target = target_arr[ir]
                    #f.write(f"\n - {name_target}, m_target = {m_target}")
                    
                    # Find target to get its density
                    for i_species,species in enumerate(species_names):
                        if species == name_target:
                            i_target = i_species
                    #i_target = 2 #find_species_index(species_names, name_target)
                    n_target = densities[i_target]

                    ## velocity of the target
                    v_targ_lab_k = np.sqrt(2.0 * Eth * eV / m_target ) * 1e2
                    v_targ_i_k = v_targ_lab_k*np.array([cos_theta_lab, sin_theta_lab, 0])
                    reduced_mass_k =  m_projectile*m_target /(m_projectile + m_target)
                    #f.write(f"\n v_proj_i = {v_proj_i}, v_targ_i_k = {v_targ_i_k}")
                    v_relative_k = v_proj_i - v_targ_i_k

                    Ecm = 0.5*reduced_mass_k*np.sum((v_relative_k/1e2)**2)/ eV
                    

                    #f.write(f"\n E_lab = {E_lab}, Ecm = {Ecm}")

                    #Ecm = E_lab * m_target / (m_target + m_projectile)
                    #CS_reac = CS_table[reac_chnl, :]
                    if name_target=="h2o":
                        CS_reac = CS_table[reac_chnl, :]
                        sigma = interp_linear(
                            Ecm,
                            E_cs_list,
                            CS_reac
                        )
                        #sigma = XS_cste_fct_numba(Ecm, sigma_cste, Ea,)
                    if name_target=="x_test":
                        #sigma =  4e-15
                        v0=3.75*1e5 # cm/s
                        sigma = 17.5e-16*(v_proj_lab/v0)**0.9
                    
                    #f.write(f"\n process {name_target},  sigma = {sigma}, v_relative_k = {v_relative_k}, n_target = {n_target}")

                    fcoll =np.sqrt(np.sum(v_relative_k**2)) * sigma * n_target
                    #fcoll = v_proj_lab * sigma * n_target
                    #f.write(f"Reaction channel: {reac_chnl}, Ecm: {Ecm} eV, sigma: {sigma} cm^2, n_target: {n_target} cm^-3, fcoll: {fcoll} s^-1\n")
                    fcoll_process_list[ir] = fcoll
                    f_tot += fcoll

                
                if f_tot <= 0.0 or not np.isfinite(f_tot):
                    break
                
                proba_sum_cascade += fcoll_process_list / f_tot

                # ---- choose process ----
                r_process = np.random.rand()
                cumulative = 0.0
                i_process = 0

                for j in range(nb_process):
                    cumulative += fcoll_process_list[j] / f_tot
                    if r_process <= cumulative:
                        i_process = j
                        break
                
                proba_list = fcoll_process_list / f_tot
                list_choice = np.ones(nb_process)
                for k in range(nb_process):
                    list_choice[k] = np.sum(proba_list[:k + 1])
                

                if removes_projectile_arr[i_process]: # if the reaction is not elastic, we loose the hydrogen atom in the cascade
    
                    break

                
                reac_chnl = reac_channel_arr[i_process]
                mass_projectile = m_projectile_arr[i_process]
                mass_target = m_target_arr[i_process]
                reduced_mass = mass_projectile*mass_target /(mass_projectile + mass_target)

                v_targ_lab = np.sqrt(2.0 * Eth * eV / mass_target ) * 1e2
                v_targ_i = v_targ_lab*np.array([cos_theta_lab, sin_theta_lab, 0])
                v_relative = v_proj_i - v_targ_i

                V_cm = (mass_projectile*v_proj_i + mass_target*v_targ_i)/(mass_projectile + mass_target)

                c_proj_i = v_proj_i - V_cm
                norm_cproj_i = np.sqrt(np.sum(c_proj_i**2))
                norm_cproj_f = norm_cproj_i # demonstration

                # ---- Collision in the CM frame ----

                ## Define the CM base coordinates in the lab frame
                e1 = c_proj_i/norm_cproj_i
                e3 = np.array([0, 0, 1])
                e2 = np.array([ e3[1]*e1[2]-e3[2]*e1[1] , e3[2]*e1[0]-e3[0]*e1[2], e3[0]*e1[1]-e3[1]*e1[0]] )

                ## Angle of the collision
                r_theta = np.random.rand()
                #theta_CM = Inverse_cdf_theta(E_lab, r_theta, E_grid_list[i_process], F_theta_inverse_list[i_process])
                

                Ecm_choice = 0.5*reduced_mass*np.sum(v_relative**2) / eV # general case
                #Ecm_choice = E_lab * mass_target / (mass_target + mH) # Case of target is at rest
                #theta_CM = theta_hg_numba(r_theta,g)
                if abs(g) < 1e-12:
                    # isotropic case
                    cos_thetaCM = 2.0 * r_theta - 1.0
                else:
                    # anisotropic HG case
                    term = (1.0 - g * g) / (1.0 + g - 2.0 * g * r_theta) # Change of the sign ==> Antonio comments
                    cos_thetaCM  = (1.0 + g * g - term * term) / (2.0 * g)
                sin_thetaCM = np.sqrt(1-cos_thetaCM**2)
                phi_CM = 2*np.pi*np.random.rand()
                
                c_proj_f = norm_cproj_f*( cos_thetaCM*e1 + sin_thetaCM*np.cos(phi_CM)*e2 + sin_thetaCM*np.sin(phi_CM)*e3) # lab frame

                # ---- energy loss ----
                # factor = 2.0 * (mH * mass_target) / ((mH + mass_target) ** 2)
                # dE_lab_Eth0 = factor * E_lab * (1.0 - np.cos(theta_CM))

                dE_lab = mass_projectile * np.sum( (c_proj_i-c_proj_f)*V_cm )/(1e4*eV) # eV

                # v_proj_f = c_proj_f + V_cm
                # dE_lab = E_lab - 0.5*m_projectile*np.sum(v_proj_f**2)
                #print(dE_lab)
                
                # ---- secondary H (H-H only) ----
                if produces_secondary_arr[i_process]: # if the reaction produces secondary particles
                    if n_active < size_max_secondaries:
                        E_secondary = Eth + dE_lab
                        if E_secondary > E_cutoff:
                            
                            E_Hf_list[n_active] = E_secondary
                            n_active += 1

                            # bin update
                            idx_bin = np.argmin(np.abs(E_dist_list - E_secondary))
                            N_dist_list[idx_bin] += 1

                # ---- update energy ----
                E_lab -= dE_lab
                idx_bin = np.argmin(np.abs(E_dist_list - E_lab))
                N_dist_list[idx_bin] += 1
                count_table[i, idx_bin] += 1

                relax_time += 1.0 / f_tot

            Ncoll_th[i] = Ncoll
            relax_time_list[i] = relax_time
            E_last[i] = E_lab

        # ---- statistics ----
        if statistics:
            
            N_mean_list = np.zeros(bin_dist)
            N_var_list = np.zeros(bin_dist)

            n_sample = i + 1

            for b in range(bin_dist):

                # ---- compute mean ----
                s = 0.0
                for k in range(n_sample):
                    s += count_table[k, b]

                mean = s / n_sample
                N_mean_list[b] = mean

                # ---- compute variance ----
                if n_sample > 1:
                    s2 = 0.0
                    for k in range(n_sample):
                        diff = count_table[k, b] - mean
                        s2 += diff * diff

                    N_var_list[b] = s2 / n_sample
                else:
                    N_var_list[b] = 0.0

            for b in range(bin_dist):
                freq = frequency_tot_distrib_sum[b]
                if freq > 0.0 and np.isfinite(freq):
                    H_distrib_table[i, b] = (
                        W0 * N_mean_list[b] / (freq * dEi)
                    )
                    SE_distrib_table[i, b] = (
                        W0 / (freq * dEi)
                    ) * np.sqrt(N_var_list[b] / (i + 1))
            
    # =====================================================
    # Final distribution & yield
    # =====================================================
    H_distrib = np.zeros(bin_dist)

    for b in range(bin_dist):
        freq = frequency_tot_distrib_sum[b]
        if freq > 0.0 and np.isfinite(freq):
            H_distrib[b] = (
                W0 * (N_dist_list[b] / N_mc) / (freq * dEi)
            )

    Yield = proba_sum_cascade/N_mc

    return (
        H_distrib,
        H_distrib_table,
        Yield,
        SE_distrib_table,
        N_dist_list,
        frequency_tot_distrib_sum,
        Ncoll_th,
        relax_time_list,
        E_last,
        #count_table
    )
