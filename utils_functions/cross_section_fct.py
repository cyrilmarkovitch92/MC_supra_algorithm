import numpy as np
import scipy.integrate as inte
import xml.etree.ElementTree as ET
from scipy.special import gamma as gamma_fct

from physics.constants import kb_cste, a0
import pickle


def CS_morton2003(E_list,mu, Ea, A, n):
    # equation (9) and (11) of Morton 2003 https://doi.org/10.1016/S0032-0633(03)00047-3 
    # E_list : Center of mass energy in [J]
    # Ea [J] : activation energy of the reaction in [J]
    # mu : reduced mass [kg]
    # A [m^3.s^-1.K^-(n-0.5)]

    Gamma_n = gamma_fct(n+1)
    C = A * ( (2**(3/2)) * (np.pi *mu)**(-1/2) * (kb_cste)**(n-0.5) * Gamma_n )**(-1) # C in m^(4-2n).kg^(1-n).s^(2n-2)
    
    sigma=np.zeros(len(E_list))
    for i, E in enumerate(E_list):
        T=E/kb_cste
        E0 = Ea - (n-0.5)*(kb_cste*T) # J
        if E > E0:
            sigma[i] = C*((E - E0)**n)/E # sigma in m^2
    return sigma*1e4 # cm^2

def DCS_Lewkow2014(Ecm, theta_cm, mu, gamma):
    # Ecm[eV], theta_cm[deg], mu[amu]
    # gamma = 1 for atom-atom collision
    # gamma = 1.4 for atom-molecule collision
    x0 = 50.12
    C1, C2, C3, C4, C5, C6, C7 = -0.13, 1.00, 2.70, 10.0, 2.04, -0.03, 32.3

    x = Ecm*theta_cm/mu
    y = gamma/(theta_cm*np.sin(theta_cm*np.pi/180))
    if x > x0:
        DCS =  y * np.exp( C1*(np.log(x))**2 + C2*np.log(x) + C3 ) 
    else:
        DCS =  y * ( C4 * np.exp( C5 + C6*np.log(x) ) + C7 ) # a0^-2
    
    return DCS*a0**2 # cm2


def integration_log(function, E_min, E_max):
    def fct_log(t):
        E=np.exp(t)
        return function(E)*E
    integration= inte.quad(fct_log, np.log(E_min), np.log(E_max), limit=10000 )[0]
    return integration



def XML_array_CR_diff(filename, energy_CM_list):
    """
    This function take the differential cross section for each energies from the XML file.
    
    Input : file name (txt), Energy list of the differential cross sections (array)
    Output : np.array [ [[E0],[theta00, theta01,...],[CR_diff00, CR_diff01,...]], 
                        [[E1],[theta10, theta11,...],[CR_diff10, CR_diff11,...]],
                         ... ]
    
    Remark : energy_CM_list is not sorted!!
    """
    # Parse the XML file (replace with the path to your XML file)
    tree = ET.parse(filename)
    root = tree.getroot()

    # Parse the XML data
    namespace = {'xsams': 'http://vamdc.org/xml/xsams/1.0'}  # Define namespace

    # Initialize a list to store the energy data
    size = len(root.findall('.//xsams:CollisionalTransition', namespace))
    CR_diff_data = []
    # Loop through each CollisionalTransition
    for i in range(size):
        #energy_id = collisional_transition.get('id')
        collisional_transition = root.findall('.//xsams:CollisionalTransition', namespace)[i]
        # Find X and Y values in TabulatedData
        x_data = collisional_transition.findall('.//xsams:X/xsams:DataList', namespace)
        y_data = collisional_transition.findall('.//xsams:Y/xsams:DataList', namespace)
        
        # Extract the theta and CR_diff values
        theta_values = list(map(float, x_data[0].text.split())) if x_data else []
        CR_diff_values = list(map(float, y_data[0].text.split())) if y_data else []
        
        # Append the formatted data
        CR_diff_data += [[energy_CM_list[i], theta_values, CR_diff_values]]

    return CR_diff_data


