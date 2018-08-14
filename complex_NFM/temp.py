import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.001
T = 10

omega = 20
phi = np.pi
mu  = 1
eps = 10

Z1s, Z2s = [], []

Z1 = 0.1 + 0.2*1j
Z2 = 0.1 + 0.2*1j

for i in range(int(T/dt)):

	F2 = np.cos(omega*i)
	F1 = np.cos(omega*i + phi)

	Z1s.append(Z1)
	Z2s.append(Z2)

	Z1dot = Z1*(mu - np.abs(Z1)**2) + omega*Z1 *1j + eps*F1
	Z2dot = Z2*(mu - np.abs(Z2)**2) + omega*Z2 *1j + eps*F2

	Z1 = Z1 + Z1dot*dt
	Z2 = Z2 + Z2dot*dt

plt.figure('Comp')
plt.plot(np.real(Z1s), np.imag(Z1s))
plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
plt.plot(np.real(Z2s), np.imag(Z2s))
plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')

plt.figure('Real')
plt.plot(np.real(Z1s), 'g')
plt.plot(np.real(Z2s), 'r')
plt.show()