from dataclasses import dataclass
from physics.constants import *
# import pickle
# import numpy as np

REACTIONS_list = []


@dataclass
class Reaction:
    reac_channel : int
    name: str
    projectile: str
    m_projectile: float
    target: str
    m_target: float
    reaction_type: str
    threshold: float
    produces_secondary: bool
    removes_projectile: bool
    # sigma_interp: callable
    # differential_model: callable


# !!!!!!!!!!!!!!! ELASTIC REACTIOSN CHANNELS MUST BE DEFINED FIRST IN THE REACTIONS_list !!!!!!!!!

###---------------------- Elastic reactions ----------------------###
chnl = 0
H_H_elastic = Reaction(
    reac_channel = chnl,
    name="h+h-->h+h",
    projectile="h",
    m_projectile = mH,
    target="h",
    m_target= mH,
    reaction_type="elastic",
    threshold=0.0,
    produces_secondary=True,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(H_H_elastic)

chnl +=1
H_He_elastic = Reaction(
    reac_channel = chnl,
    name="h+he-->h+he",
    projectile="h",
    m_projectile = mH,
    target="he",
    m_target= mHe,
    reaction_type="elastic",
    threshold=0.0,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(H_He_elastic)

chnl +=1
H_H2_elastic = Reaction(
    reac_channel = chnl,
    name="h+h2-->h+h2",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="elastic",
    threshold=0.0,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(H_H2_elastic)


n_p_elastic = Reaction(
    reac_channel = -1,
    name="n+p-->n+p",
    projectile="neutron",
    m_projectile = m_neutron,
    target="proton",
    m_target= m_proton,
    reaction_type="elastic",
    threshold=0.0,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)

H_Xtest_elastic = Reaction(
    reac_channel = -2,
    name="h+x_test-->h+x_test",
    projectile="h",
    m_projectile = mH,
    target="x_test",
    m_target= m_test,
    reaction_type="elastic",
    threshold=0.0,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)

###---------------------- Inelastic reactions ----------------------###

################################# VIBRATION : H + H2(v_low) --> H + H2(v_up) #################################

#------ v_low = 0 --> v_up = 1
chnl +=1
h_h2v01_inel = Reaction(
    reac_channel = chnl,
    name="h+h2v0-->h+h2v1",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="inelastic",
    threshold=0.52,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v01_inel)

#------ v_low = 0 --> v_up = 2
chnl +=1
h_h2v02_inel = Reaction(
    reac_channel = chnl,
    name="h+h2v0-->h+h2v2",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="inelastic",
    threshold=1.00305,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v02_inel)

#------ v_low = 0 --> v_up = 3
chnl +=1
h_h2v03_inel = Reaction(
    reac_channel = chnl,
    name="h+h2v0-->h+h2v3",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="inelastic",
    threshold = 1.4616,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v03_inel)

#------ v_low = 0 --> v_up = 4
chnl +=1
h_h2v04_inel = Reaction(
    reac_channel = chnl,
    name="h+h2v0-->h+h2v4",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="inelastic",
    threshold = 1.89181,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v04_inel)


#------ v_low = 0 --> v_up = 5
chnl +=1
h_h2v05_inel = Reaction(
    reac_channel = chnl,
    name="h+h2v0-->h+h2v5",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="inelastic",
    threshold = 2.29369,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v05_inel)

################################# ROVIBRATION : H + H2(v_low, J_low) --> H + H2(v_up, J_up) #################################

### Pure rotation v=0 of Lique 2016 : doi:10.1093/mnras/stv1683
# Include an average contribution for the pure rotation for v=0

# filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rot_dico_v0_Lique.pkl"
# with open(filename, "rb") as f:
#     rot_dico = pickle.load(f)

# CS_rot_table = rot_dico["CS"]
# Stopping_CS_rot_table = rot_dico["Stopping_CS"]
# Delta_E_rot= np.where(CS_rot_table != 0, Stopping_CS_rot_table/ CS_rot_table, 0)

dE_rot = 0.1
#------ (0, J) --> (0, J') : Excitation + De-excitation (super elastic)
chnl +=1
h_h2v0J_rot = Reaction(
    reac_channel = chnl,
    name="h+h2v0J-->h+h2v0Jp",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="rovib",
    threshold = dE_rot,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2v0J_rot)

"""

### Pure rotation v=0 of Lique 2016 : doi:10.1093/mnras/stv1683
# State-to-State approach

filename_levels_L = f"/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rovib_H_H2_Lique/levels_Lique.pkl"
with open(filename_levels_L, "rb") as f:
    levels_L = pickle.load(f)


for initial_level, ini in levels_L.items():
    if ini["v"]==0:
        for final_level, fin in levels_L.items():
            if fin["v"]==0:
                # Skip elastic collisions
                if initial_level == final_level:
                    continue

                chnl += 1

                vi, Ji = ini["v"], ini["J"]
                vf, Jf = fin["v"], fin["J"]

                # Threshold energy (only positive for excitation)
                Eth = (fin["energy_cm-1"] - ini["energy_cm-1"]) * cm_TO_EV

                #if Eth>0: # Only consider excitation
                reaction = Reaction(
                    reac_channel=chnl,
                    name=f"h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf}",
                    projectile="h",
                    m_projectile=mH,
                    target="h2",
                    m_target=mH2,
                    reaction_type="inelastic",
                    threshold=Eth,
                    produces_secondary=False,
                    removes_projectile=False,
                )

                # Create a variable with the desired name
                globals()[f"h_h2_{initial_level}_{final_level}"] = reaction

                REACTIONS_list.append(reaction)

"""


### Total rovibrational of Lique 2016 : doi:10.1093/mnras/stv1683
# Include an average contribution of the rovibrational crontribution
"""
filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rovib_dico_Lique.pkl"
with open(filename, "rb") as f:
    rovib_dico = pickle.load(f)

CS_rovib_table = rovib_dico["CS"]
Stopping_CS_rovib_table = rovib_dico["Stopping_CS"]
Delta_E_rovib= np.where(CS_rovib_table != 0, Stopping_CS_rovib_table/ CS_rovib_table, 0)


#------ (v, J) --> (v', J') : Excitation + De-excitation (super elastic)
chnl +=1
h_h2vJ_rovib = Reaction(
    reac_channel = chnl,
    name="h+h2vJ-->h+h2vpJp",
    projectile="h",
    m_projectile = mH,
    target="h2",
    m_target= mH2,
    reaction_type="rovib",
    threshold = Delta_E_rovib,
    produces_secondary=False,
    removes_projectile=False,
    # sigma_interp=sigma_HH,
    # differential_model=dcs_HH
)
REACTIONS_list.append(h_h2vJ_rovib)


"""
### Writhmall : http://dx.doi.org/10.1088/0953-4075/40/16/003
# Including each level one by one
"""
filename = "/Users/cm285277/Desktop/PhD_CEA/Code/H-suprathermal/MC_supra_structured/data/cross_section_data/H-H2/Rotation_H2_para_ortho/ortho/levels_ortho.pkl"


with open(filename, "rb") as f:
    levels = pickle.load(f)

for initial_level, ini in levels.items():

    for final_level, fin in levels.items():

        # Skip elastic collisions
        if initial_level == final_level:
            continue

        chnl += 1

        vi, Ji = ini["v"], ini["J"]
        vf, Jf = fin["v"], fin["J"]

        # Threshold energy (only positive for excitation)
        Eth = (fin["energy_K"] - ini["energy_K"]) * K_TO_EV

        if Eth>0: # Only consider excitation
            reaction = Reaction(
                reac_channel=chnl,
                name=f"h+h2v{vi}J{Ji}-->h+h2v{vf}J{Jf}",
                projectile="h",
                m_projectile=mH,
                target="h2",
                m_target=mH2,
                reaction_type="inelastic",
                threshold=Eth,
                produces_secondary=False,
                removes_projectile=False,
            )

        # Create a variable with the desired name
        globals()[f"h_h2_{initial_level}_{final_level}"] = reaction

        REACTIONS_list.append(reaction)
"""


###---------------------- Chemical reactions ----------------------###

chnl +=1
H_H2O__OH_H2 = Reaction(
    reac_channel = chnl,
    name="h+h2o-->oh+h2",
    projectile="h",
    m_projectile = mH,
    target="h2o",
    m_target= mH2O,
    reaction_type="dissociation",
    threshold=0.88,
    produces_secondary=False,
    removes_projectile=True,
    #sigma_interp=sigma_H2O
    #differential_model=None,
)
REACTIONS_list.append(H_H2O__OH_H2)

######## Test : elastic reactions defined first
type_test0 = "elastic"
for r in REACTIONS_list:
    type_test1 = r.reaction_type
    if type_test1 == "elastic" and type_test0 != "elastic":
        print(f"Error: Elastic reaction channel {r.reac_channel} is defined after a non-elastic reaction channel: {r.name}")
    type_test0 = type_test1


###---------------------- Photochemical reactions ----------------------###chnl +=1
hnu_h__hp_e = Reaction(
    reac_channel = chnl,
    name="h+hnu-->hp+e",
    projectile="hnu",
    m_projectile = 0.0, #has to be a float
    target="h",
    m_target= mH,
    reaction_type="photoionization",
    threshold=0.0, #has to be a float
    produces_secondary=False,
    removes_projectile=True,
    #sigma_interp=sigma_H2O
    #differential_model=None,
)
REACTIONS_list.append(hnu_h__hp_e)
