#%%
from dataclasses import dataclass

from physics.constants import *


@dataclass
class Species:

    name: str
    mass: float


H = Species("h", mH)
H2 = Species("h2", mH2)
He = Species("he", mHe)
H2O = Species("h2o", mH2O)

X_test = Species("x_test", m_test)