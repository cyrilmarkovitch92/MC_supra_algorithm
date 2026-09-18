

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from data.load_cs_dcs import load_cross_sections
from physics.reactions import REACTIONS_list
import pickle

from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_inverse_cdf_table(E_cm, theta_cm, F_theta, N_grid):

    # 2D interpolation of F(E,theta)
    Ftheta_interp = RegularGridInterpolator(
        (E_cm, theta_cm),
        F_theta,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Log energy grid (same physics as before)
    E_grid = np.logspace(np.log10(np.min(E_cm)),
                         np.log10(np.max(E_cm)),
                         N_grid)

    theta_grid = np.linspace(np.min(theta_cm),
                             np.max(theta_cm),
                             N_grid)

    # Build F table on dense grid
    F_table = np.zeros((N_grid, N_grid))
    for i, E in enumerate(E_grid):
        F_table[i, :] = Ftheta_interp((E, theta_grid))

    # Now build inverse CDF table
    # We invert F_table[i,:] numerically
    r_grid = np.linspace(0.0, 1.0, N_grid)
    theta_inverse_table = np.zeros((N_grid, N_grid))

    for i in range(N_grid):
        theta_inverse_table[i, :] = np.interp(
            r_grid,
            F_table[i, :],
            theta_grid
        )

    return E_grid, r_grid, theta_inverse_table



def preprocess_reactions(E_cs_list, cs_data, reaction_list,T_rovib_list, Ngrid=1000):

    CS_table = []

    E_grid_list = []
    r_grid_list = []
    theta_inverse_list = []
    
    for reac in reaction_list:
        print(reac.name)
        
        data = cs_data[reac.name]

        if reac.reaction_type == "rovib":
            Twanted = 1000 #K

            T_rovib = data["CS"]["T_rovib"]
            iT = np.argmin(abs(Twanted-T_rovib))
            E_cs = data["CS"]["energy"]
            cs_table = data["CS"]["cross_section"][iT]
            
            cs_interp1D = interp1d(E_cs, cs_table,kind='linear', bounds_error=False, fill_value=0)
            CS_table.append( np.array(cs_interp1D(E_cs_list ),dtype=float ) )
            
            #cs_interp2D = RegularGridInterpolator((T_rovib, E_cs), cs_table,method="linear",bounds_error=False,fill_value=0 )
            #CS_table.append(cs_interp2D((T_rovib_list,E_cs_list )) )
        else:

            E_cs = data["CS"]["energy"]
            cs_list = data["CS"]["cross_section"]
            cs_interp = interp1d(E_cs, cs_list, kind='linear', bounds_error=False, fill_value=0)

            CS_table.append(cs_interp(E_cs_list))

            E_dcs_list = data["DCS"]["E_cm"]

            if reac.reaction_type == "elastic": #!!! F_theta function only for elastic reactions

                # angular distribution handling
                theta_list = data["DCS"]["theta_cm"]
                F_list = data["DCS"]["Ftheta_table"]

                Egrid, rgrid, theta_inv = build_inverse_cdf_table(
                    E_dcs_list, theta_list, F_list, Ngrid
                )

                E_grid_list.append(Egrid)
                r_grid_list.append(rgrid)
                theta_inverse_list.append(theta_inv)

        
        ## The grid will contain only the elastic reactions. Those reactions has to be defined in the begining of the physics/reaction file
        # else:
        #     E_grid_list.append(np.zeros(Ngrid))
        #     r_grid_list.append(np.zeros(Ngrid))
        #     theta_inverse_list.append(np.zeros((Ngrid, Ngrid)))
        #     # E_grid_list.append(None)
        #     # r_grid_list.append(None)
        #     # theta_inverse_list.append(None)

    return (
        CS_table, #np.array(CS_table, dtype='f'), # size (N_reactions, N_grid)
        np.array(E_grid_list, dtype='f'), # size (N_elastic_reactions, N_grid)
        np.array(r_grid_list, dtype='f'), # size (N_elastic_reactions, N_grid)
        np.array(theta_inverse_list, dtype='f') # size (N_elastic_reactions, N_grid, N_grid)
    )


# =====================================================
# Load the Xsec and diff xsec & F_theta function in a table
# =====================================================

cs_data_load = load_cross_sections()
Ngrid = 1000
E_cs_list = np.logspace(-4, 2, Ngrid) # energy grid for cross sections (same as before)
T_rovib_list = np.logspace(np.log10(100), np.log10(5000), Ngrid)
(CS_table, E_grid_array, r_grid_array, theta_inverse_array) = preprocess_reactions(E_cs_list, cs_data_load, REACTIONS_list,T_rovib_list=T_rovib_list, Ngrid= Ngrid)

tables_dico = {
    "E_cs_list": E_cs_list,
    "CS_table": CS_table,
    "E_grid_array": E_grid_array,
    "r_grid_array": r_grid_array,
    "theta_inverse_array": theta_inverse_array,
    "T_rovib_list":T_rovib_list
}

path_table = HERE
#path_table = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/"
with open(path_table / "tables_CS_theta_inverse.pkl", "wb") as f:
    pickle.dump(tables_dico, f)


