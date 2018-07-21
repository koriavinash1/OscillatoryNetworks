from Oscillators import *
from configure import Config
from tqdm import tqdm
from GaussianStatistics import *
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
import pdb

# load Orientation Bars
OrientationBars = np.load('./Orientation_bars.npy')
print ("Orientation Bars Loaded: Shape {}".format(OrientationBars.shape))
weights_path = './nfm_weights_4_Lat0.05.npy'


Gstat = GaussianStatistics()
config= Config()

# for 3D plot...
X = np.arange(0, config.N)
Y = np.arange(0, config.N)
X, Y = np.meshgrid(X, Y)

# ----------------------------------------------

# dynamic to dynamic
class NFM(object):
	def __init__(self, ci=0.3, test=None):
		self.ci    = ci
		self.ita   = 5e-4
		self.s2dnfm = CoupledNFM(size  =(config.N, config.N),
							exe_rad    = config.eRad,
							inhb_rad   = config.N,
							exe_ampli  = config.eA,
							inhb_ampli = config.iA)

	def display(self, Z, i, fig,  _type = '2d'):
		plt.clf()
		plt.ion()
		if _type == '3d':
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(X, Y, np.array(Z, dtype='float') , cmap=cm.coolwarm,
			               linewidth=0, antialiased=False)
			ax.set_title(np.max(Z))
			# Customize the z axis.
			ax.set_zlim(-1.01, 1.01)
			ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
			fig.colorbar(surf, aspect=10)
		else:
			plt.imshow(Z)

		plt.draw()
		plt.xlabel(str(i))
		plt.pause(0.5)
		pass

	# Statis input to dynamic output
	def staticToDynamic(self, image):
		"Statis to Dynamic needs one to one connections,\
		Between static image and Oscillator"

		
		s2d_nsheets = []
		for i in range(0, int(config.T/config.dt)):
			self.s2dnfm.lateralDynamics(verbose = True, ci = self.ci)
			s2d_nsheets.append(self.s2dnfm.Z)
			if i %100 == 99:
				fig2D = plt.figure('2D Mag')
				self.display(np.abs(self.s2dnfm.Z), i, fig2D)
				fig3D = plt.figure('2D Phase')
				self.display(np.angle(self.s2dnfm.Z), i, fig3D)
		return np.array(s2d_nsheets)





if __name__ == '__main__':
	nfm = NFM(test=None)
	images = OrientationBars.reshape(-1, 10, 10)
	print(np.max(images[0]), np.min(images[0]))


	nfm.staticToDynamic(images[23, :, :])

	# nfm.response(images, simulations=4)

	# weights = np.load('./nfm_weights.npy').reshape(10,10,10,10)
	# weights = np.load(weights_path).reshape(10,10,10,10)
	# a = np.empty((100,100))
	# for i in range(10):
	# 	for j in range(10):
	# 		a[i*10:(i+1)*10, j*10:(j+1)*10] = weights[i,j,:,:]
	# plt.imshow(a)
	# plt.show()

	# plt.ion()
	# for i in range(0, len(images)-4, 22):
	# 	plt.imshow(nfm.check_resp(images[i]))
	# 	plt.xlabel(str(i))
	# 	plt.pause(1)
	# 	plt.show()