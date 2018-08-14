import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01
T = 50

omega = 0.002*np.pi/T
phi = 180.* np.pi/180.0
c = -0.1 - 10j  
train_epoch = 30

ita =1e-3


np.random.seed(2018)
cp1 = 0.001*(np.random.randn() + np.random.randn()*1j )
cp2 = 0.001*(np.random.randn() + np.random.randn()*1j )


def weight_update(Z1, Z2, w, phi = phi):
	dw = ita *(Z1*np.conjugate(Z2) - w)# + np.abs(Z1) * np.abs(Z2) *(1 + 1j))
	w  = w + dw
	return w, dw

dws = []	
for ep in range(50):
	Z1s, Z2s = [], []
	plt.ion()
	
	np.random.seed(2018)
	Z1, Z2 = 0.01*(np.random.randn() + np.random.randn()*1j), 0.01*(np.random.randn() + np.random.randn()*1j)

	for i in range(int(T/dt)):
		
		Z1star = np.conjugate(Z1)
		Z2star = np.conjugate(Z2)


		if ep < train_epoch:
			I1 = -0.01 + 0.1*np.sin(omega*i)
			I2 = -0.01 + 0.1*np.sin(omega*i + phi)
		else:
			I1 = -0.1 + 0.1*np.sin(omega*i)
			I2 = -0.1

		Z1s.append(Z1)
		Z2s.append(Z2)

		Z1dot = Z1star + Z1 *1j - c*Z1*Z1*Z1star + I1 + cp1 * (Z2-Z1star)
		Z2dot = Z2star + Z2 *1j - c*Z2*Z2*Z2star + I2 + np.conjugate(cp1) * (Z1-Z2star)

		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt
		
		if ep < train_epoch:
			cp1, dw = weight_update(Z1, Z2, cp1) 
			cp1     = cp1 / np.abs(cp1)


		dws.append(dw)

	# plt.subplot(2, 1, 1)
	plt.figure('Comp')
	plt.clf()
	plt.plot(np.real(Z1s), np.imag(Z1s))
	plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
	plt.plot(np.real(Z2s), np.imag(Z2s))
	plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')
	plt.title("Epoch : {}".format(ep))

	plt.figure('Real')
	plt.clf()
	plt.plot(np.abs(Z1s))
	plt.plot(np.abs(Z2s))

	plt.figure('dw')
	plt.clf()
	plt.plot(np.real(dws))
	# plt.plot(np.imag(dws))

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