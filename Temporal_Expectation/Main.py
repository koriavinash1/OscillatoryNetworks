import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from Gratings import *
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
		pass