import numpy as np

class Config():
	def __init__(self):
		self.ita = 1e-3
		self.dt  = 0.01
		self.T   = 70
		self.N   = 10
		self.TrainingTime = 5000

		# time for projection ...
		self.rdeltaT = 1000
		self.gdeltaT = 1000
		self.omega  = 2.*np.pi/ 25.0
		self.mu     = 1
		self.eps    = 1

		self.Thresh = 0.0015
		self.generate = True
		self.random   = False
		self.saveLat  = False
	pass