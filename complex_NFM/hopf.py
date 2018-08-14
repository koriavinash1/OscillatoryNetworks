import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01

a = 2 + 0.01j  # inc img part for depolarization 
b = 2 + 0.01j  # inc img part for depolarization
c = -0.1 - 100j  

d = -0.01 # + 0.001j # i ext
Z = 0.05 + 0.01j

Zs = []
omega = 0.002*np.pi/500

for i in range(5000):
	Zs.append(Z)
	Zstar = np.conjugate(Z)
	F = np.sin(omega*i)

	Zdot = Z + 1*(np.pi/6.)*Z *1j - Z*Z*Zstar + .001*F 
	
	# if np.abs(Zdot) > 100: Zdot = 10 + 10j
	# Zdot = (c/fr) + x*((-1*a /fr) + eps*1j) - y*(1./fr + eps*gamma) + (x**2 / fr) * (a + 1) + x**3 / fr 
	Z = Z + Zdot*dt


#plt.subplot(5, 1, 1)
plt.plot(np.real(Zs))
plt.show()

# plt.subplot(5, 1, 2)
plt.plot(np.imag(Zs))
plt.show()

# plt.subplot(5, 1, 3)
plt.plot(np.real(Zs), np.imag(Zs))
plt.plot(np.real(Zs[0]), np.imag(Zs[0]), '*r')
plt.show()

# plt.subplot(5, 1, 4)
plt.plot(np.abs(Zs))
plt.show()

# plt.subplot(5, 1, 5)
plt.plot(np.angle(Zs))

plt.show()