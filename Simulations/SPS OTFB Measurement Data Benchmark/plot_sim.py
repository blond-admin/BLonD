import numpy as np
import matplotlib.pyplot as plt


dir = 'sim_data/1000turns_fl/'

V_ANT = np.load(dir + '3sec_Vant.npy')

plt.plot(np.abs(V_ANT) / 4)
plt.show()