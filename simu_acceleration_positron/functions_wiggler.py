import numpy as np
from scipy.constants import e, c
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation

def update_rad_int(ring, wiggler, E):
    I1w = 0
    I2w = wiggler.DI2_woE / (E * e / c) ** 2
    I3w = wiggler.DI3_woE / (E * e / c) ** 3
    I4w = wiggler.DI4_woE / (E * e / c) ** 3
    I5w = 0
    return np.array([ring.I1+I1w, ring.I2+I2w, ring.I3+I3w, ring.I4+I4w, ring.I5+I5w])

def update_SRtracker_and_track(ring, rfstation, beam, wiggler, E):
    SR = [SynchrotronRadiation(ring, rfstation, beam, rad_int = update_rad_int(ring, wiggler,E), quantum_excitation=True, python=True, shift_beam=False)]
    SR[0].print_SR_params()
    SR[0].track()

class wiggler:
    def __init__(self, nw=2, np = 43, Lp = 0.095, B = 1):
        self.nw = nw
        self.np = np
        self.Lp = Lp #m
        self.Lw = self.Lp * self.np #2 * (self.np - 2) * self.Lp
        self.B = B #T
        self.DI2_woE = None
        self.DI3_woE = None
        self.DI4_woE = None
        self.DI5_woE = None

        self.update_DI2()
        self.__str__()

    def update_DI2(self):
        #E typically in eV

        self.DI2_woE = 1/2 * self.nw * self.Lw * (e * self.B)** 2
        self.DI1_woE = -self.DI2_woE * (self.Lw / 2 / np.pi) ** 2
        self.DI3_woE = 4  / (3 * np.pi) * self.nw * self.Lw * (e * self.B) ** 3
        self.DI4_woE = 0 # dispersion created neglected
        self.DI5_woE = 0 # dispersion created neglected, self.nw * self.Lp**2 * self.Lw / (15 * np.pi **3 )* (e * self.B ) ** 5
