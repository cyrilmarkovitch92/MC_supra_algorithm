#%% 
## In the terminal run python3 -m data.build_CS_dico
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent

from physics.constants import *
from utils_functions.usefull_fct import *
from utils_functions.cross_section_fct import XML_array_CR_diff, CS_morton2003

from physics.reactions import REACTIONS_list

def convert_txt_array(text):
    # Split the text by spaces
    array_list = text.split()
    # Convert strings to floats (optional)
    array_list = [float(num) for num in array_list]

    return array_list

def empty_process():
    return {
        "m2": 0,
        "CS": {"energy": np.array([]), "cross_section": np.array([])},
        "DCS": {"E_cm": np.array([]), "theta_cm": np.array([]), "DCS_table": np.array([]), "Ftheta_table": np.array([])},
    }


process_list = []
for reac in REACTIONS_list:
    process_list.append(reac.name)

CS_dico = {p: empty_process() for p in process_list }

print(CS_dico.keys())

for reac in REACTIONS_list:
    reac_name = reac.name
    print(reac_name)

###---------------------- Elastic reactions ----------------------###
    
    ########################### H-H elastic collisions ############################### 
    if reac_name == "h+h-->h+h":
        print("Processing H-H elastic collision...")
        
        #---------------------------- H-H Cross section
        ### ALLADIN : Krstic 1999 https://doi.org/10.1088/0953-4075/32/14/317
        text_E = "1.0000E-01 1.2589E-01 1.5849E-01 1.9953E-01 2.5119E-01 3.1623E-01 3.9811E-01 5.0119E-01 6.3096E-01 7.9433E-01 1.0000E+00 1.2589E+00 1.5849E+00 1.9953E+00 2.5119E+00 3.1623E+00 3.9811E+00 5.0119E+00 6.3096E+00 7.9433E+00 1.0000E+01 1.2589E+01 1.5849E+01 1.9953E+01 2.5119E+01 3.1623E+01 3.9811E+01 5.0119E+01 6.3096E+01 7.9433E+01 1.0000E+02"
        E_HH = convert_txt_array(text_E)
        text_CS ="5.4364E-15 5.7175E-15 5.8286E-15 5.2021E-15 5.7113E-15 5.7468E-15 4.9585E-15 5.1808E-15 5.1758E-15 4.9118E-15 4.7443E-15 4.6900E-15 4.5017E-15 4.4801E-15 4.2864E-15 4.2548E-15 4.0348E-15 4.0082E-15 3.7930E-15 3.7415E-15 3.5890E-15 3.4494E-15 3.4085E-15 3.1948E-15 3.1182E-15 3.0633E-15 2.8726E-15 2.7166E-15 2.6627E-15 2.6056E-15 2.4784E-15"
        CS_HH = convert_txt_array(text_CS)

        CS_dico[reac_name]["CS"]["energy"] = E_HH
        CS_dico[reac_name]["CS"]["cross_section"] = CS_HH
        sig_HH = interp1d(E_HH, CS_HH, kind='linear', bounds_error=False, fill_value=0)

    
        # Ovchinnikov 2017 : fig 7 of https://doi.org/10.1088/1361-6455/aa64ac 

        # filename="cross_section_data/H-H_CR_distinguishable_Ovchinnikov.csv"
        # columns=import_csv_column(filename)
        # columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        # columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        # E_cm_distin_Ovchinnikov, CS_HH_distin_Ovchinnikov= columns[:,0], columns[:,1]
        # CS_HH_distin_Ovchinnikov = CS_HH_distin_Ovchinnikov*a0**2 #Conversion from a.u to cm^2

        # filename="cross_section_data/H-H_CR_undistinguishable_Ovchinnikov.csv"
        # columns=import_csv_column(filename)
        # columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        # columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        # E_cm_undistin_Ovchinnikov, CS_HH_undistin_Ovchinnikov= columns[:,0], columns[:,1]
        # CS_HH_undistin_Ovchinnikov = CS_HH_undistin_Ovchinnikov*a0**2 #Conversion from a.u to cm^2


        #---------------------------- H-H Differential cross section 
        # Krstic 1999 https://doi.org/10.1088/0953-4075/32/14/317
        # Note that the differential cross section is 2*pi*sin(theta)*dsig/dOmega

        filename = HERE / "H-H" / "H-H_CR_diff.xml"
        #filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H/H-H_CR_diff.xml"
        #filename = "/cross_section_data/H-H_elastic/H-H_CR_diff.xml"
        energy_CM_list_HH = np.array([0.1, 1, 1.25, 1.58, 1.99, 2.51, 3.16, 3.98, 5.01, 
                                6.3, 7.94, 0.125, 10, 12.5, 15.8, 19.9, 25.1, 31.6,
                                39.8, 50.1, 63, 79.4, 0.158, 100, 0.199, 0.251, 0.316,
                                0.398, 0.501, 0.631, 0.794])

        CS_diff_HH_data = XML_array_CR_diff(filename, energy_CM_list_HH)

        # Sort energies and corresponding data
        unsorted_energies = energy_CM_list_HH
        sorted_indices = np.argsort(unsorted_energies)
        E_cm = unsorted_energies[sorted_indices]

        # Assuming each entry in CS_diff_HH_data[i] has:
        # CS_diff_HH_data[i][1] -> theta array
        # CS_diff_HH_data[i][2] -> DCS array
        # !!! all theta arrays are identical !!!
        theta_cm = np.array(CS_diff_HH_data[sorted_indices[0]][1])

        # Initialize 2D tables
        DCS_table = np.zeros((len(E_cm), len(theta_cm)))
        Ftheta_table = np.zeros_like(DCS_table)

        for i, i_sorted in enumerate(sorted_indices): 
            # i = 0, 1, 2, ...
            # i_sorted = 0, 11, 12, 8, ... which correspond to the energy sorted
            energy = energy_CM_list_HH[i_sorted]
            theta = np.array(CS_diff_HH_data[i_sorted][1])
            DCS = np.array(CS_diff_HH_data[i_sorted][2])

            DCS_table[i, :] = DCS  # fill one row per energy

            # Compute cumulative normalized distribution F_theta
            F_theta = np.zeros_like(theta)
            for j in range(1, len(theta)):
                F_theta[j] = simpson(DCS[:j], theta[:j]) / sig_HH(energy)
            
            Ftheta_table[i, :] = F_theta

        # Store into dictionary
        CS_dico[reac_name]["DCS"] = {
            "E_cm": E_cm,
            "theta_cm": theta_cm[:-1], # 2 times 3.14 so we remove one
            "DCS_table": DCS_table[:, :-1],
            "Ftheta_table": Ftheta_table[:, :-1]
        }



    ################################# H-H2 elastic collision  #################################
    if reac_name == "h+h2-->h+h2":
        print("Processing H-H2 elastic collision...")
        #---------------------------- Cross section
        text_E = "1.0000E-01 1.2589E-01 1.5849E-01 1.9953E-01 2.5119E-01 3.1623E-01 3.9811E-01 5.0119E-01 6.3096E-01 7.9433E-01 1.0000E+00 1.2589E+00 1.5849E+00 1.9953E+00 2.5119E+00 3.1623E+00 3.9811E+00 5.0119E+00 6.3096E+00 7.9433E+00 1.0000E+01 1.9953E+01 5.0119E+01 1.0000E+02"
        E_HH2 = convert_txt_array(text_E)
        text_CS ="5.7640E-15 5.7141E-15 5.6615E-15 5.6056E-15 5.5466E-15 5.4842E-15 5.4219E-15 5.3427E-15 5.3020E-15 5.1922E-15 5.0739E-15 5.0824E-15 4.7109E-15 4.6692E-15 4.6431E-15 4.4978E-15 4.2593E-15 4.0646E-15 4.0063E-15 4.0241E-15 4.0476E-15 3.7059E-15 2.7675E-15 2.1170E-15"
        CS_HH2 = convert_txt_array(text_CS)

        CS_dico[reac_name]["CS"]["energy"] = E_HH2
        CS_dico[reac_name]["CS"]["cross_section"] = CS_HH2

        sig_HH2 = interp1d(E_HH2, CS_HH2, kind='linear', bounds_error=False, fill_value=0) # One could use fill_value="extrapolate" + heaviside function when it's negative.


        #---------------------------- H-H2 differential cross section
        filename = HERE / "H-H2" / "H-H2_CR_diff.xml"
        #filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/H-H2_CR_diff.xml"
        energy_CM_list_HH2 = np.array([0.1, 1, 1.25, 1.58, 1.99, 2.51, 3.16, 3.98, 5.01, 
                                6.3, 7.94, 0.125, 10, 19.9, 50.1, 0.158,
                                100, 0.199, 0.251, 0.316, 0.398, 0.501, 0.631, 0.794])

        CS_diff_HH2_data = XML_array_CR_diff(filename, energy_CM_list_HH2)


        # Sort energies and corresponding data
        unsorted_energies = energy_CM_list_HH2
        sorted_indices = np.argsort(unsorted_energies)
        E_cm = unsorted_energies[sorted_indices]

        # !!! all theta arrays are identical !!!
        theta_cm = np.array(CS_diff_HH2_data[sorted_indices[0]][1])

        # Initialize 2D tables
        DCS_table = np.zeros((len(E_cm), len(theta_cm)))
        Ftheta_table = np.zeros_like(DCS_table)

        for i, i_sorted in enumerate(sorted_indices): 
            # i = 0, 1, 2, ...
            # i_sorted = 0, 11, 12, 8, ... which correspond to the energy sorted
            energy = energy_CM_list_HH2[i_sorted]
            theta = np.array(CS_diff_HH2_data[i_sorted][1])
            DCS = np.array(CS_diff_HH2_data[i_sorted][2])

            DCS_table[i, :] = DCS  # fill one row per energy

            # Compute cumulative normalized distribution F_theta
            F_theta = np.zeros_like(theta)
            for j in range(1, len(theta)):
                F_theta[j] = simpson(DCS[:j], theta[:j]) / sig_HH2(energy)
            
            Ftheta_table[i, :] = F_theta

        # Store into dictionary
        CS_dico[reac_name]["DCS"] = {
            "E_cm": E_cm,
            "theta_cm": theta_cm[:-1], # 2 times 3.14 so we remove one
            "DCS_table": DCS_table[:, :-1], # 2 times 3.14 so we remove one
            "Ftheta_table": Ftheta_table[:, :-1] # 2 times 3.14 so we remove one
        }


    ################################# H-He elastic collision  #################################
    # From Swaczyna 2021 : https://doi.org/10.3847/2041-8213/abf436

    if reac_name == "h+he-->h+he":
        print("Processing H-He elastic collision...")
        #---------------------------- Cross section

        filename = HERE /"H-He"/"H_He_elas_CS_DCS_Ftheta.pkl"
        #filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-He/H_He_elas_CS_DCS_Ftheta.pkl"
        with open(filename, "rb") as f:
            h_he_elas_dico = pickle.load(f)

        CS_dico[reac_name]["CS"] = {
            "energy": h_he_elas_dico["CS"]["energy"],
            "cross_section": h_he_elas_dico["CS"]["cross_section"]
        }

        CS_dico[reac_name]["DCS"] = {
            "E_cm": h_he_elas_dico["DCS"]["E_cm"],
            "theta_cm": h_he_elas_dico["DCS"]["theta_cm"],
            "DCS_table": h_he_elas_dico["DCS"]["DCS_table"],
            "Ftheta_table": h_he_elas_dico["DCS"]["Ftheta_table"]
        }

        """
        #### Extract H-He data from txt file
        # Directory containing the files
        folder = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-He/Swaczyna_2021-diff_cross_sections/H0He0el/"

        # Initialize a list to store the data from all files
        all_data = []

        # Loop through all files in the folder
        for filename in os.listdir(folder):
            if filename.endswith(".dat"):  # Process only .dat files
                filepath = os.path.join(folder, filename)

                with open(filepath, 'r') as file:
                    lines = file.readlines()

                # Extract the CM energy
                for line in lines:
                    if line.startswith("# CM energy:"):
                        cm_energy = float(line.split(":")[1].strip().split()[0])
                        break

                # Find the starting line of the numerical data
                data_start_line = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("# the\tdcs\twei"):
                        data_start_line = i + 1
                        break

                if data_start_line is not None:
                    # Extract the numerical data
                    numerical_data = np.loadtxt(lines[data_start_line:], usecols=(0, 1))
                    theta = numerical_data[:, 0]  # First column (the)
                    dcs = numerical_data[:, 1]    # Second column (dcs)

                    # Append data to the list as [CM energy, theta array, dcs array]
                    all_data.append([cm_energy, theta, dcs])

        # Convert the list to a structured numpy matrix/object array
        CS_diff_HHe_data = np.array(all_data, dtype=object)

        size_HHe = len(CS_diff_HHe_data )
        # Save the data for later use (optional)
        #np.save("cross_section_data/all_DCS_He.npy", CS_diff_HHe_data)

        ### Example of accessing data from the matrix
        # CM energy of the first file : data_DCS_HHe[0][0]
        # Theta array of the first file : data_DCS_HHe[0][1]
        # DCS=dsigma/dOmega*2*pi*sin(the) array of the first file : data_DCS_HHe[0][2]

        energy_CM_list_HHe = np.zeros(size_HHe)
        for i in range(size_HHe):
            energy_CM_list_HHe[i] = CS_diff_HHe_data[i][0]

        # Sort energies and corresponding data
        unsorted_energies = energy_CM_list_HHe
        sorted_indices = np.argsort(unsorted_energies)
        E_cm = unsorted_energies[sorted_indices]


        # !!! all theta arrays are identical !!!
        theta_cm = CS_diff_HHe_data[sorted_indices[0]][1]

        ### Cross section
        CS_list = np.zeros(len(E_cm))
        for i, i_sorted in enumerate(sorted_indices): 
            # i = 0, 1, 2, ...
            # i_sorted = 0, 11, 12, 8, ... which correspond to the energy sorted
            energy = energy_CM_list_HHe[i_sorted]
            theta = CS_diff_HHe_data[i_sorted][1]
            DCS = CS_diff_HHe_data[i_sorted][2]*a0**2

            # Compute cross section
            CS = simpson(DCS, theta)
            CS_list[i] = CS

        sig_HHe = interp1d(E_cm, CS_list, kind='linear', bounds_error=False, fill_value=0) # One could use fill_value="extrapolate" + heaviside function when it's negative.

        CS_dico[reac_name]["CS"] = {
            "energy": E_cm,
            "cross_section": CS_list
        }

        ### Differential cross section
        DCS_table = np.zeros((len(E_cm), len(theta_cm)))
        Ftheta_table = np.zeros_like(DCS_table)

        for i, i_sorted in enumerate(sorted_indices): 
            # i = 0, 1, 2, ...
            # i_sorted = 0, 11, 12, 8, ... which correspond to the energy sorted
            energy = energy_CM_list_HHe[i_sorted]
            theta = CS_diff_HHe_data[i_sorted][1]
            DCS = CS_diff_HHe_data[i_sorted][2]*a0**2

            DCS_table[i, :] = DCS  # fill one row per energy

            # Compute cumulative normalized distribution F_theta
            F_theta = np.zeros_like(theta)
            for j in range(1, len(theta)):
                F_theta[j] = simpson(DCS[:j], theta[:j]) / sig_HHe(energy)
            
            Ftheta_table[i, :] = F_theta

            # Compute cross section
            CS = simpson(DCS, theta)
            CS_list[i] = CS
        
        # Store into dictionary
        CS_dico[reac_name]["DCS"] = {
            "E_cm": E_cm,
            "theta_cm": theta_cm,
            "DCS_table": DCS_table,
            "Ftheta_table": Ftheta_table
        }
        """




###---------------------- Inelastic reactions ----------------------###


    ################################# VIBRATION : H + H2(v_low) --> H + H2(v_up) #################################
    initial_vstate = 0
    final_vstate = [1, 2, 3, 4, 5]

    filename = HERE / "H-H2"/"Vib_dico_v0vp_Lique_Krstic.pkl"
    #filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Vib_dico_v0vp_Lique_Krstic.pkl"
    with open(filename, "rb") as f:
        vib_h2_dico = pickle.load(f)
        
    Ecm_vib = vib_h2_dico["Ecm"]


    for ivf, vf in enumerate(final_vstate ):
        if reac_name == f"h+h2v{initial_vstate}-->h+h2v{vf}":
            #print(f"h+h2v{initial_vstate}-->h+h2v{vf}", reac.reac_channel)
            CS_vib_lique_krstic = vib_h2_dico[vf]

            CS_dico[reac_name]["CS"] = {
                "energy": np.array(Ecm_vib, dtype=float), #lab frame
                "cross_section": np.array(CS_vib_lique_krstic,dtype=float)
            }
    
    """
    #------ v_low = 0 --> v_up = 1
    if reac_name == "h+h2v0-->h+h2v1":
        vlow, vup = 0, 1
        print("Processing h+h2v0-->h+h2v1 collision...")
        
        ## From Table 8 of Phelps 1990 https://srd.nist.gov/jpcrdreprint/1.555858.pdf
        # E_h2vlow_vup = np.array([1.000, 1.334, 1.778, 2.371, 3.162, 4.217, 5.623, 7.499, 10.00, 13.34, 17.78, 23.11, 31.62, 42.17, 56.23, 74.99, 100.0, 133.4,
        #     177.8, 237.1, 316.2, 421.7, 562.3, 749.9, 1000, 1334, 1778, 2371, 3162, 4217, 5623, 7499, 10000])
        
        # CS__h2vlow_vup = np.array([0.076, 0.365, 0.77, 1.23, 1.7, 2.23, 2.8, 3.35, 3.85, 4.3, 4.7, 4.95, 5.13, 5.23, 5.2, 5.05, 4.75, 4.47, 4.08, 3.65, 
        #     3.25, 2.84, 2.45, 2.05, 1.7, 1.4, 1.12, 0.89, 0.71, 0.53, 0.4, 0.297, 0.225])*1e-16

        ## From Figure 5 a of Krstic and Schultz 2000 : 10.1088/0953-4075/32/10/310
        filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/CS_h2v{vlow}_v{vup}-Fig5a_Krstic2000.csv"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_h2vlow_vup, CS_h2vlow_vup= columns[:,0], columns[:,1]*(a0)**2 ## Energy in the center of mass

        CS_dico[reac_name]["CS"] = {
            "energy": np.array(E_h2vlow_vup, dtype=float), #lab frame
            "cross_section": np.array(CS_h2vlow_vup,dtype=float)
        }

    #------ v_low = 0 --> v_up = 2
    if reac_name == "h+h2v0-->h+h2v2":
        vlow, vup = 0, 2
        print("Processing h+h2v0-->h+h2v2 collision...")
        
        ## From Figure 5 of Krstic and Schultz 2000 : 10.1088/0953-4075/32/10/310
        filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/CS_h2v{vlow}_v{vup}-Fig5a_Krstic2000.csv"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_h2vlow_vup, CS_h2vlow_vup= columns[:,0], columns[:,1]*(a0)**2

        CS_dico[reac_name]["CS"] = {
            "energy": np.array(E_h2vlow_vup, dtype=float), #lab frame
            "cross_section": np.array(CS_h2vlow_vup,dtype=float)
        }

    #------ v_low = 0 --> v_up = 3
    if reac_name == "h+h2v0-->h+h2v3":
        vlow, vup = 0, 3
        print("Processing h+h2v0-->h+h2v3 collision...")
        
        ## From Figure 5 of Krstic and Schultz 2000 : 10.1088/0953-4075/32/10/310
        filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/CS_h2v{vlow}_v{vup}-Fig5a_Krstic2000.csv"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_h2vlow_vup, CS_h2vlow_vup= columns[:,0], columns[:,1]*(a0)**2

        CS_dico[reac_name]["CS"] = {
            "energy": np.array(E_h2vlow_vup, dtype=float), #lab frame
            "cross_section": np.array(CS_h2vlow_vup,dtype=float)
        }

    #------ v_low = 0 --> v_up = 4
    if reac_name == "h+h2v0-->h+h2v4":
        vlow, vup = 0, 4
        print("Processing h+h2v0-->h+h2v4 collision...")
        
        ## From Figure 5 of Krstic and Schultz 2000 : 10.1088/0953-4075/32/10/310
        filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/CS_h2v{vlow}_v{vup}-Fig5a_Krstic2000.csv"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_h2vlow_vup, CS_h2vlow_vup= columns[:,0], columns[:,1]*(a0)**2

        ## Correction --> Some energy values are the same, then when doing interpolation --> problem
        # for ie, E_h2v4 in enumerate(E_h2vlow_vup):
        #     for E_up in E_h2vlow_vup[E_h2vlow_vup>E_h2v4]:
        #         if E_up ==E_h2v4:
        #             E_h2vlow_vup[ie] = E_h2vlow_vup[ie] - E_h2vlow_vup[ie]/1e6

        CS_dico[reac_name]["CS"] = {
            "energy": np.array(E_h2vlow_vup, dtype=float), #lab frame
            "cross_section": np.array(CS_h2vlow_vup,dtype=float)
        }

    #------ v_low = 0 --> v_up = 5
    if reac_name == "h+h2v0-->h+h2v5":
        vlow, vup = 0, 5
        print("Processing h+h2v0-->h+h2v5 collision...")
        
        ## From Figure 5 of Krstic and Schultz 2000 : 10.1088/0953-4075/32/10/310
        filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/CS_h2v{vlow}_v{vup}-Fig5a_Krstic2000.csv"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_h2vlow_vup, CS_h2vlow_vup= columns[:,0], columns[:,1]*(a0)**2

        CS_dico[reac_name]["CS"] = {
            "energy": np.array(E_h2vlow_vup, dtype=float), #lab frame
            "cross_section": np.array(CS_h2vlow_vup,dtype=float)
        }
    """

    ################################# ROVIBRATION : H + H2(v_low, J_low) --> H + H2(v_up, J_up) #################################
    
    ### Pure rotation of v=0 : Lique 2016 : doi:10.1093/mnras/stv1683
    # Include an average contribution of the pure rotation v=0 crontribution
    filename = HERE / "H-H2" / "Rot_dico_v0_Lique.pkl"
    #filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rot_dico_v0_Lique.pkl"
    with open(filename, "rb") as f:
        rot_dico = pickle.load(f)

    T_rot_h2 = rot_dico["T"]
    E_rot_h2 = rot_dico["E"]
    CS_rot_h2_table = rot_dico["CS"]
    Stopping_CS_rot_h2_table = rot_dico["Stopping_CS"]
    #Delta_E_rot= np.where(CS_rot_table != 0, Stopping_CS_rot_table/ CS_rot_table, 0)

    if reac_name == f"h+h2v0J-->h+h2v0Jp":
        CS_dico[reac_name]["CS"] = {
            "T_rovib":T_rot_h2,
            "energy": E_rot_h2 , #lab frame
            "cross_section": CS_rot_h2_table,
            "Stopping_CS": Stopping_CS_rot_h2_table
        }
    """

    ### Pure rotation of v=0 : Lique 2016 : doi:10.1093/mnras/stv1683
    # State-to-state approach
    

    T_build_CS = 1000 #K --> Test with this temperature. In reality, we should define a matrix with different temperatures
    
    filename_levels_L = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rovib_H_H2_Lique/levels_Lique.pkl"
    with open(filename_levels_L, "rb") as f:
        levels_L = pickle.load(f)

    filename_CS_L = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rovib_H_H2_Lique/CS_rovib_Lique.pkl"
    with open(filename_CS_L, "rb") as f:
        CS_rovib_L = pickle.load(f)

    # Partition function
    Q_T_L=0
    for initial_level, ini in levels_L.items():
        g_i = ini["g"]
        E_i = ini['energy_cm-1'] #cm-1
        ## Q(T)
        Q_T_L += g_i*np.exp(-(E_i*cm_TO_EV/K_TO_EV)/T_build_CS)

    for initial_level, ini in levels_L.items():
        if ini["v"]==0:
            for final_level, fin in levels_L.items():
                if fin["v"]==0:
    
                    # Skip elastic collisions
                    if initial_level == final_level:
                        continue

                    vi, Ji = ini["v"], ini["J"]
                    vf, Jf = fin["v"], fin["J"]

                    if reac_name == f"h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf}":
                        print(f"Processing h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf} collision...")
                        g_i = ini["g"]
                        E_i = ini['energy_cm-1']*cm_TO_EV*eV #J
                        E_h2_rot = CS_rovib_L[final_level][initial_level][:,0] * cm_TO_EV # Convert to eV
                        CS_h2_rot = (g_i*np.exp(-E_i/(kb_cste*T_build_CS))/Q_T_L)*CS_rovib_L[final_level][initial_level][:,1] * 1e-16 # Convert to cm^2

                        CS_dico[reac_name]["CS"] = {
                            "energy": E_h2_rot, #lab frame
                            "cross_section": CS_h2_rot
                        }
    """
    

    ### Total Rovibration Lique 2016 : doi:10.1093/mnras/stv1683
    # Include an average contribution of the rovibrational crontribution
    """
    filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rovib_dico_Lique.pkl"
    with open(filename, "rb") as f:
        rovib_dico = pickle.load(f)

    T_rovib_h2 = rovib_dico["T"]
    E_rovib_h2 = rovib_dico["E"]
    CS_rovib_h2_table = rovib_dico["CS"]
    Stopping_CS_rovib_h2_table = rovib_dico["Stopping_CS"]


    if reac_name == f"h+h2vJ-->h+h2vpJp":
        CS_dico[reac_name]["CS"] = {
            "T_rovib":T_rovib_h2,
            "energy": E_rovib_h2, #lab frame
            "cross_section": CS_rovib_h2_table,
            "Stopping_CS": Stopping_CS_rovib_h2_table
        }
    """
    ### Writhmall : http://dx.doi.org/10.1088/0953-4075/40/16/003
    # Including each level one by one
    """
    K_TO_EV = 8.617333262145e-5

    filename_CS = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rotation_H2_para_ortho/ortho/CS_dico_H2_rot_ortho.pkl"
    with open(filename_CS, "rb") as f:
        cross_sections = pickle.load(f)

    filename_level = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rotation_H2_para_ortho/ortho/levels_ortho.pkl"
    with open(filename_level, "rb") as f:
        levels = pickle.load(f)

    for initial_level, ini in levels.items():

        for final_level, fin in levels.items():

            # Skip elastic collisions
            if initial_level == final_level:
                continue

            vi, Ji = ini["v"], ini["J"]
            vf, Jf = fin["v"], fin["J"]

            if reac_name == f"h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf}":
                print(f"Processing h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf} collision...")

                E_h2_rot = cross_sections[final_level][initial_level][:,0] * K_TO_EV # Convert to eV
                CS_h2_rot = cross_sections[final_level][initial_level][:,1] * 1e-16 # Convert to cm^2

                CS_dico[reac_name]["CS"] = {
                    "energy": E_h2_rot, #lab frame
                    "cross_section": CS_h2_rot
                }
    """




###---------------------- Chemical reactions ----------------------###

    ################################# H + H2O --> OH + H2 #################################
    if reac_name == "h+h2o-->oh+h2":
        print("Processing h+h2o-->oh+h2 collision...")

        #------------ Morton 2003 (model): https://doi.org/10.1016%2FS0032-0633(03)00047-3
        mu = (mH * 18 * mH) / (mH + 18 * mH) # kg
        # Parameters from Table 3 of Morton 2003 : https://linkinghub.elsevier.com/retrieve/pii/S0032063303000473 
        A = 1.5e-10*1e-6 # m^3.s^-1.K^-(n-0.5)
        n = 0.5
        Ea = 0.88 * eV # J    

        E_chemical_list = np.logspace(-1, 3, 1000) # eV

        CS_HH2O = CS_morton2003(E_chemical_list * eV, mu, Ea, A, n) # cm^2
        
        #---------- Brouard 2004 (experiment)
        ## From Figure 4 of Brouard 2004
        filename = HERE / "H-H2O" / "CS_fullline-Fig4_Brouard2004.csv"
        #filename = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2O/c"
        columns=import_csv_column(filename)
        columns[:,0] =np.array([s.replace(',', '.') for s in columns[:,0]], dtype=float)
        columns[:,1] =np.array([s.replace(',', '.') for s in columns[:,1]], dtype=float)
        E_Brouard, CS_Brouard= np.array(columns[:,0], dtype=float), np.array(columns[:,1]*1e-16, dtype=float)

        CS_dico[reac_name]["CS"] = {
            "energy": E_Brouard, #E_Brouard, E_chemical_list
            "cross_section": CS_Brouard, #CS_Brouard , CS_HH2O
        }

###---------------------- Photochemical reactions ----------------------###
    if reac_name == "h+hnu-->hp+e":
        print("Processing h+hnu-->hp+e collision...")
        ### Put the cross section to 0 at the moment just to fit with Panarese 
        E_photoioniz = np.logspace(-4,2,1000)
        CS_photo_ioniz = np.zeros(len(E_photoioniz))
        CS_dico[reac_name]["CS"] = {
                    "energy": np.array(E_photoioniz, dtype=float),
                    "cross_section": np.array(CS_photo_ioniz, dtype=float) , #
                }
    




###. !!!!!!! Store data in a dictionnary !!!!!!!

# Save dictionary to a file
#CS_dico = convert_defaultdict_to_dict(CS_dico)
DIR_DATA = HERE 
#DIR_DATA = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/"
with open(DIR_DATA/'CS_dico.pkl', 'wb') as file:
    pickle.dump(CS_dico, file)


