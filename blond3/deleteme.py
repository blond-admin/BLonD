import matplotlib.pyplot as plt
import numpy as np

# series 1 before 20:20
_xs = [-164,-161, -167, -168]
_ys = [[25.4],[24.7, 28.2, 39], [38,10, 9], [10, 5, 6.7, 5.3], [6.5, 2.8, 0.4]]
# series 1 20:28
_xs = [-169, -165, -174, -172, -171]

_ys = [[15,24,16], [27, 40, 55], [58, -12, -11, -2, -12, -3, 6, -12],
       [-5,-1, 2], [15, 20, 18]]


xs = []
ys = []
for i, x in enumerate(_xs):
    for y in _ys[i]:
        xs.append(x)
        ys.append(y)
xs,ys = np.array(xs), np.array(ys)
plt.plot(xs,ys, "o", color="C0")
z = np.polyfit(xs, ys, 1)
range_ = xs.max() - xs.min()
xs_fit = np.linspace(xs.min()-range_, xs.max()+range_)
ys_fit = np.poly1d(z)(xs_fit)
idx = np.argmin(np.abs(ys_fit))
plt.axvline(xs_fit[idx], color="k", label=f"{xs_fit[idx]}")
plt.plot(xs_fit, ys_fit)
plt.xlabel("Rotor phase a")
plt.ylabel("Symmetry factor (%)")
plt.legend()
plt.show()