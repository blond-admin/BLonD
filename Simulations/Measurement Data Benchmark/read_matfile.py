import scipy.io
import matplotlib.pyplot as plt
import numpy as np

mat = scipy.io.loadmat('dataTimeDomainSim.mat')

V3section = mat['V3section']
V4section = mat['V4section']
beamPhase = mat['beamPhase']
hCav3Section = mat['hCav3Section']
hCav4Section = mat['hCav4Section']
transmitter3Section = mat['transmitter3Section']
transmitter4Section = mat['transmitter4Section']

def mat_converter(data):
    new_data = np.zeros(data.shape[0], dtype=complex)

    for i in range(data.shape[0]):
        new_data[i] = data[i]

    return new_data

plt.figure(1)
plt.title('hCav3Section')
plt.plot(hCav3Section[0])

plt.figure(2)
plt.title('hCav4Section')
plt.plot(hCav4Section[0])
plt.show()