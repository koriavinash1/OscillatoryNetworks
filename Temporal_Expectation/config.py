import numpy as np

class Config():
	def __init__(self):
		self.ita = 1e-3
		self.dt  = 0.01
		self.T   = 100
		self.N   = 10

		# time for projection ...
		self.deltaT = 500
		self.omega  = 2.*np.pi/ 30.0
		self.mu     = 1
		self.eps    = 1

		self.Thresh = 0.003
	pass