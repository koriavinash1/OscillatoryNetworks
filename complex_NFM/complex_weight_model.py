import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.001
T = 50

omega = 0.002*np.pi/T
phi = 180.* np.pi/180.0
c = -0.1 - 10j  

ita =1e-3


np.random.seed(2018)
cp1 = (+0.01*np.random.randn() + np.random.randn()*0.01j )
cp2 = (+0.01*np.random.randn() + np.random.randn()*0.01j )


def weight_update(Z1, Z2, w, phi = phi):
	dw = ita *(Z1*np.conjugate(Z2) + Z2*np.conjugate(Z1) + 1+1j)
	w  = w + dw
	return w / np.abs(w), dw

# np.random.seed(2018)
# Z1 = Z2
# np.random.seed(2018)
Z1, Z2 = (0.05 + 0.01j), (0.05 + 0.01j)

dws = []	
for ep in range(50):
	Z1s, Z2s = [], []
	plt.ion()
	
	for i in range(int(T/dt)):
		
		Z1star = np.conjugate(Z1)
		Z2star = np.conjugate(Z2)

		if ep < 10:
			I1 = 0.002 + (0.001*np.sin(omega*i)) + 0.01*(cp1)*(Z2 - Z1)
			I2 = 0.002 + (0.001*np.sin(omega*i + phi)) + 0.01*(cp2)*(Z1 - Z2)
		else:
			I1 = 0.001 + 0.01*(cp1)*(Z2 - Z1)#/(np.abs(Z1)*np.abs(Z2))
			I2 = 0.001 + 0.01*(cp2)*(Z1 - Z2)#/(np.abs(Z1)*np.abs(Z2))

		# if i % 100 == 99:
		# 	exit()
		# print ("sinI1: {}".format(np.sin(omega*i + phi)) + "  I1 : {}".format(np.abs(I1)) + "  I2 : {}".format(np.abs(I2)) + "  CP1 : {}".format(np.abs(cp1)) +"  CP2 : {}".format(np.abs(cp2)))

		Z1s.append(Z1)
		Z2s.append(Z2)

		Z1dot = (Z1star + Z1 *1j + c*Z1*Z1*Z1star - I1)
		Z2dot = (Z2star + Z2 *1j + c*Z2*Z2*Z2star - I2)

		# print ("Z1 : {}".format(Z1) + "  Z1star : {}".format(Z1star) + " Z2 : {}".format(Z2) + "Z2star : {}".format(Z2star))
		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt

		# Z1 = Z1/np.abs(Z1)
		# Z2 = Z2/np.abs(Z2)
		
		if ep < 10:
			cp1, dw = weight_update(Z1, Z2, cp1) 
			cp2, _  = weight_update(Z2, Z1, cp2) 

		dws.append(dw)

	# plt.subplot(2, 1, 1)
	plt.figure('Comp')
	plt.clf()
	plt.plot(np.real(Z1s), np.imag(Z1s))
	plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
	# plt.subplot(2, 1, 2)
	plt.plot(np.real(Z2s), np.imag(Z2s))
	plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')
	plt.title("Epoch : {}".format(ep))

	plt.figure('Real')
	plt.clf()
	plt.plot(np.abs(Z1s))
	plt.plot(np.abs(Z2s))

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