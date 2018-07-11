import numpy as np
import matplotlib.pyplot as plt

T = np.arange(0, 5000)/100.
# F = np.sin(8*t)

def sanity_check(I, aff, Zold, Wold, Oold, dt=0.01):
    """
        To find oscillatory regimes for single neuron
    """
    v = Zold
    w = Wold
    o = Oold

    ext =(I + I[::-1]) + (aff + I[::-1])

    fv   = v*(0.5 - v)*(v - 1)
    vdot = (fv - w + ext)/2.
    wdot = 0.1*v - 0.1*w

    # vdot = v*(v+12)*(1-v) - w + 5*(I - I[::-1])
    # wdot = o*(v-0.01*w)
    # omegadot = -5*I*w/(np.sqrt(v**2 + w**2))

    v = v + vdot*dt
    w = w + wdot*dt
    o = o + omegadot*dt

    # v1[v1 < 0] = 0
    return v, w, o

Z, OO = [], []
v = np.array([1, 05])
w = np.array([1, 05])
o = np.array([10, 10])

for t in T:
	Z.append(v)
	OO.append(o)

	I  = np.array([np.sin(1*t)/4., np.sin(2*t)/4.])
	aff = np.array([np.sin(3*t)/4., np.sin(3*t)/4.])

	v, w, o = sanity_check(I, aff, v, w, o)

Z = np.array(Z)
OO = np.array(OO)

plt.plot(np.sin(3*T)/4)
plt.plot(Z[:, 0])
plt.plot(Z[:, 1])
plt.show()

# plt.plot(np.sin(8*T))
# plt.plot(OO[:, 0])
# plt.plot(OO[:, 1])
# plt.show()
