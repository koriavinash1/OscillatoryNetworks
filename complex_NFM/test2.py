import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01

a = 2 + 0.01j  # inc img part for depolarization 
b = 2 + 0.01j  # inc img part for depolarization
c = 0.01 + 1j  
cp = 0.001 + 0.001j  




Z1, Z2 = 0.05 + 0.01j, 0.05 + 0.01j

Z1s, Z2s = [], []

for i in range(5000):
	d1 = 0.01*np.exp((2*np.pi*5*0.0002*i )*1j) # i ext
	d2 = 0.01*np.exp((2*np.pi*5*0.0002*i + np.pi/2)*1j) # i ext
	Z1s.append(Z1)
	Z2s.append(Z2)

	Z1star = np.conjugate(Z1)
	Z2star = np.conjugate(Z2)

	Z1dot = a * Z1star + b* Z1 *1j + c * ((5.0/24.0)*Z1 + (1.0/8.0) * Z1star)*Z1*Z1star + cp*(Z1 - Z2) + d1
	Z2dot = a * Z2star + b* Z2 *1j + c * ((5.0/24.0)*Z2 + (1.0/8.0) * Z2star)*Z2*Z2star + cp*(Z2 - Z1) + d2
	
	Z1 = Z1 + Z1dot*dt
	Z2 = Z2 + Z2dot*dt


# plt.subplot(2, 1, 1)
plt.plot(np.real(Z1s))
# plt.subplot(2, 1, 2)
plt.plot(np.real(Z2s))
plt.show()

# plt.subplot(2, 1, 1)
plt.plot(np.imag(Z1s))
# plt.subplot(2, 1, 2)
plt.plot(np.imag(Z2s))
plt.show()

# plt.subplot(2, 1, 1)
plt.plot(np.real(Z1s), np.imag(Z1s))
plt.plot(np.real(Z1s[0]), np.imag(Z1s[0]), '*r')
# plt.subplot(2, 1, 2)
plt.plot(np.real(Z2s), np.imag(Z2s))
plt.plot(np.real(Z2s[0]), np.imag(Z2s[0]), '*r')
plt.show()