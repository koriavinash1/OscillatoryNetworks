import numpy as np

class Config():
	def __init__(self):
		self.ita = 1e-3
		self.dt  = 0.01
		self.T   = 100
		self.N   = 10

		# time for projection ...
		self.deltaT = 500
		self.omega  = 2.*np.pi/ 25
		self.mu     = 2
		self.eps    = 0.01

		self.Thresh = 0.5
	pass