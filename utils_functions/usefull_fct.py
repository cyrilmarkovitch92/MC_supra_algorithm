import numpy as np 
import pandas as pd
from physics.constants import *


def convert_txt_array(text):
    # Split the text by spaces
    array_list = text.split()
    # Convert strings to floats (optional)
    array_list = [float(num) for num in array_list]

    return array_list

def fmt(val):
    """
    Format a number for filenames:
    - powers of 10 → '1eX'
    - integers → 'int'
    - otherwise → scientific with no '+'
    """
    if isinstance(val, (int, np.integer)):
        return str(val)

    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))

        exp = int(np.round(np.log10(val)))
        if np.isclose(val, 10**exp, rtol=1e-12):
            return f"1e{exp}"

        return f"{val:.2e}".replace("+", "")

    raise TypeError("Unsupported type")

def import_csv_column(filename):
    """
    return columns from csv file as an array columns =[column1,column2]
    To get the first column as an array : column1=columns[:,0]
    """
    try:
        data=pd.read_csv(filename, delimiter=";")
        columns=data.to_numpy()
        return columns
    except Exception as e:
        print("Error occurred while importing CSV:", e)
        return None, None
    
def save_to_csv(energy_list, spectrum_list, file_name):

    """
    Save two numpy arrays into a CSV file with two columns: energy and spectrum.

    Args:
        energy_list (numpy.ndarray): Array of energy values.
        spectrum_list (numpy.ndarray): Array of spectrum values.
        file_name (str): Name of the output CSV file.
    """
    # Ensure both lists are numpy arrays
    energy_list = np.array(energy_list)
    spectrum_list = np.array(spectrum_list)
    
    # Check if they have the same length
    if len(energy_list) != len(spectrum_list):
        raise ValueError("Energy and spectrum lists must have the same length.")
    
    # Combine the arrays into a single array for writing to CSV
    data = np.column_stack((energy_list, spectrum_list))
    
    # Save to CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')  # You can switch to `delimiter=';'` if needed
        # Write the header
        writer.writerow(["Energy", "Spectrum"])
        # Write the data
        writer.writerows(data)
    print(f"Data saved to {file_name}")


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
n_colors = 40

def create_colormap_list(n_colors):

    """
    Returns a list of colors compatible with plt.plot(..., color=...)
    First 10 colors are fixed, additional ones are generated from a colormap.
    """

    base_colors = [
        "red", "green", "blue", "orange", "purple",
        "brown", "pink", "gray", "cyan", "magenta"
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # Generate additional distinct colors
    extra_needed = n_colors - len(base_colors)
    cmap = plt.cm.tab20  # good qualitative map

    extra_colors = [mcolors.to_hex(cmap(i / extra_needed))
                    for i in range(extra_needed)]

    return base_colors + extra_colors


### Return the class into lists
def Class_to_list_MC( wanted_reactions):

    reac_channel_arr = np.array(
        [r.reac_channel for r in wanted_reactions],
        dtype=np.int32
    )

    m_projectile_arr = np.array(
        [r.m_projectile for r in wanted_reactions],
        dtype=np.float64
    )

    target_arr = np.array(
        [r.target for r in wanted_reactions],
        dtype=np.str_
    )

    m_target_arr = np.array(
        [r.m_target for r in wanted_reactions],
        dtype=np.float64
    )

    produces_secondary_arr = np.array(
        [r.produces_secondary for r in wanted_reactions],
        dtype=np.bool_
    )

    removes_projectile_arr = np.array(
        [r.removes_projectile for r in wanted_reactions],
        dtype=np.bool_
    )

    reaction_type_arr = np.array(
        [r.reaction_type for r in wanted_reactions],
        dtype=np.str_
    )

    threshold_arr = [r.threshold for r in wanted_reactions]#,
        #dtype=np.float64
    

    return reac_channel_arr, m_projectile_arr, target_arr, m_target_arr, produces_secondary_arr, removes_projectile_arr, reaction_type_arr, threshold_arr


def maxwellian(E,T):
    # E [eV]
    # T [K]
    #1 eV = 1.6e-19 J, donc 1/J=1.6e-19 1/eV
    # return f_maxwellian[eV^-1]
    return 2*np.sqrt(E*eV/np.pi)*((kb_cste*T)**(-3/2))*np.exp(-E*eV/(kb_cste*T))*eV
    #return 2*np.sqrt(E/np.pi)*((kb*T)**(-3/2))*np.exp(-E/(kb*T))



