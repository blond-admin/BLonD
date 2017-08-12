
from beam.beam import Beam
from input_parameters.ring import Ring
import beam.profile as slicesModule
import numpy as np
import matplotlib.pyplot as plt


n_turns = 1

ring_length = 125

alpha = 0.001

general_params = Ring(n_turns, ring_length, alpha, 1e9)

n_macroparticles = 1000000
intensity = 1e10

my_beam = Beam(general_params, n_macroparticles, intensity)
np.random.seed(134213434)
my_beam.dt = np.random.randn(n_macroparticles)*general_params.t_rev[0]/25+general_params.t_rev[0]/2
my_beam.dE = np.random.randn(n_macroparticles)

plt.figure(1)
plt.plot(my_beam.dt, my_beam.dE, '.')

n_slices = 200

slices = slicesModule.Profile(my_beam)
slices.track()
plt.figure(2)
plt.plot(slices.bin_centers, slices.n_macroparticles)

CutOptions = slicesModule.CutOptions(cut_left=0, cut_right=general_params.t_rev[0], n_slices = n_slices, cuts_unit='s', omega_RF=2*np.pi/general_params.t_rev[0])
FitOptions = slicesModule.FitOptions(fitMethod='fwhm', fitExtraOptions=None)
FilterOptions = slicesModule.FilterOptions(filterMethod=None, filterExtraOptions=None)
OtherSlicesOptions = slicesModule.OtherSlicesOptions(smooth=False, direct_slicing = True)

slices_complex = slicesModule.Profile(my_beam, CutOptions = CutOptions,
                 FitOptions= FitOptions,
                 FilterOptions=FilterOptions, 
                 OtherSlicesOptions = OtherSlicesOptions)

plt.figure(3)
plt.plot(slices_complex.bin_centers, slices_complex.n_macroparticles)

print(slices_complex.bunchPosition, slices_complex.bunchLength)
plt.show()

