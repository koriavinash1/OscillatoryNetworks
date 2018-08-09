import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01
T = 50

omega = 0.002*np.pi/T
phi = 180.* np.pi/180.0

a = 0.02 + 0.01j  # inc img part for depolarization 
b = a
ita =1e-4
c = 0.01 + 1j  
np.random.seed(2018)
cp1 = 0.01 *(+0.01*np.random.randn() + np.random.randn()*0.01j )
cp2 = 0.01 *(+0.01*np.random.randn() + np.random.randn()*0.01j )
Z1, Z2 = 0.05*np.random.randn() + np.random.randn()*0.01j, 0.05*np.random.randn() + np.random.randn()*0.01j

def weight_update(Z1, Z2, w, phi = phi):
	# dw = ita *(Z1*np.conjugate(Z2) + Z2*np.conjugate(Z1) - 1)
	dw = ita *(np.abs(Z1) * np.abs(Z2) * (1 + np.array([np.cos(np.angle(Z1) - np.angle(Z2) + phi), np.sin(np.angle(Z2) - np.angle(Z1) + phi)]))) + 0j
	w  = w + dw

	return w / np.abs(w), dw

# np.random.seed(2018)
# Z1 = Z2

dws = []	
for ep in range(50):
	Z1s, Z2s = [], []
	plt.ion()
	plt.clf()
	for i in range(int(T/dt)):
		
		Z1star = np.conjugate(Z1)
		Z2star = np.conjugate(Z2)

		if ep < 30:
			I1 = 0.05 + 0.01*np.sin(omega*i) + 0j  + np.conjugate(cp1)*(Z2 * Z1star -1)# constant < 0.18
			I2 = 0.05 + 0.01*np.sin(omega*i + phi) + 0j + np.conjugate(cp2)*(Z1 * Z2star -1)
		else:
			I1 = 0.15 + np.conjugate(cp1)*(Z2 * Z1star - 1)
			I2 = 0.15 + np.conjugate(cp2)*(Z1 * Z2star - 1)

		# if i % 500 == 499:
			# exit()
			# print ("I1 : {}".format(np.abs(I1)) + "  I2 : {}".format(np.abs(I2)) + "  CP1 : {}".format(np.abs(cp1)) +"  CP2 : {}".format(np.abs(cp2)))
		# print (Z1, Z2, Z1star, Z2star)
		# I2 = np.real(I1) + np.imag(I2)*1j

		Z1s.append(Z1)
		Z2s.append(Z2)

		Z1dot = a * Z1star - b* Z1 *1j + c * ((5.0/24.0)*Z1)*Z1*Z1star + I1  
		Z2dot = a * Z2star - b* Z2 *1j + c * ((5.0/24.0)*Z2)*Z2*Z2star + I2 
		
		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt

		# Z1 = Z1/np.abs(Z1)
		# Z2 = Z2/np.abs(Z2)
		
		if ep < 30:
			cp1, dw = weight_update(Z1, Z2, cp1) 
			cp2, _ = weight_update(Z2, Z1, cp2) 

		dws.append(dw)

	# plt.subplot(2, 1, 1)
	plt.plot(np.real(Z1s))
	# plt.subplot(2, 1, 2)
	plt.plot(np.real(Z2s))
	plt.title("Epoch : {}".format(ep))
	plt.pause(0.5)
	# plt.show()
	# # plt.subplot(2, 1, 1)
	# plt.plot(np.imag(Z1s))
	# # plt.subplot(2, 1, 2)
	# plt.plot(np.imag(Z2s))
	# plt.show()

	# # plt.subplot(2, 1, 1)
	# plt.plot(np.real(Z1s), np.imag(Z1s))
	# plt.plot(np.real(Z1s[0]), np.imag(Z1s[0]), '*r')
	# # plt.subplot(2, 1, 2)
	# plt.plot(np.real(Z2s), np.imag(Z2s))
	# plt.plot(np.real(Z2s[0]), np.imag(Z2s[0]), '*r')
	# plt.show()