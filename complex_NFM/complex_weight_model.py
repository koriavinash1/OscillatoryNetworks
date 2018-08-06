import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01
T = 500

omega = 0.002*np.pi/T
phi = 10* np.pi/180.0

a = 0.5 + 0.008j  # inc img part for depolarization 
b = 0.5 + 0.008j  # inc img part for depolarization
c = 0.05 + 0.05j  
cp = 0*(+0.01 + 0.01j  )




Z1, Z2 = 0.05 + 0.01j, 0.05 + 0.01j

Z1s, Z2s = [], []

for i in range(int(T/dt)):
	I1 = 0.005*np.exp(1j*omega*i) 
	I2 = 0.005*np.exp(1j*(omega*i + phi))

	# I2 = np.real(I1) + np.imag(I2)*1j

	Z1s.append(Z1)
	Z2s.append(Z2)

	Z1star = np.conjugate(Z1)
	Z2star = np.conjugate(Z2)

	Z1dot = a * Z1star + b* Z1 *1j + c * ((5.0/24.0)*Z1)*Z1*Z1star + I1 + cp*np.sin(np.angle(Z2) - np.angle(Z1))
	Z2dot = a * Z2star + b* Z2 *1j + c * ((5.0/24.0)*Z2)*Z2*Z2star + I2 + cp*np.sin(np.angle(Z1) - np.angle(Z2))
	
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