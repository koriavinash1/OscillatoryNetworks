import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from Gratings import *
from Oscillator import CoupledNFM
from configure import Config
from SOM import SOM
import pdb

Gstat = GaussianStatistics()
config= Config()
grts  = Gratings()

rand = lambda N: np.random.randn(N, N)

class Main(object):
	def __init__(self, deltaT):
		self.deltaT = deltaT
		self.Zs     = np.zeros((config.N, config.N, int(config.T/condig.dt)), dtype='complex64')
		self.Flag   = False
		self.Flagcount  = 0
		self.oscillator = CoupledNFM()
		pass


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


	def runNFM(self, _display = False, blink=True):
		"""
		"""

		for t in range(0, int(config.T/config.dt), 1):
			temp_aff = rand(config.N)

			if blink:
				if np.random.randint(1) and t > 3000:
					self.Flag = True

				if self.Flag and self.Flagcount < config.deltaT:
					temp_aff = grts.fixedGrating(theta = 45) # [45, 135]
					self.Flagcount += 1
				else:
					self.Flag = False

			self.oscillator.lateralDynamics(temp_aff)
			self.oscillator.updateLatWeights()
			self.Zs[:,:, t] = self.oscillator.Z

			if _display and t % 100 == 0: self.display(np.real(self.oscillator.Z))


		return self.Zs

	def filteringSignal(self):
		"""
			Delta wave filtering...
			(0 - 4 Hz)
		"""

		pass


	def classifier(self):
		"""
			delta Signal classifier 
		"""

		pass