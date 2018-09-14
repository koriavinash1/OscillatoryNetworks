import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from config import Config
# from SOM import SOM
import pdb

Gstat = GaussianStatistics()
config= Config()


# ===================================================================================
# NFM 2D implementation
class CoupledNFM(object):
    " D2D is Dynamic input to Dynamic system\
    With all to all connections usage: >> "

    def __init__(self, 
                size       = (10,10), 
                exe_rad    = 2,
                inhb_rad   = 10, 
                exe_ampli  = 2., 
                inhb_ampli = 1., 
                test       = None):

        # ===================================================================
        # init
        self.eRad = exe_rad
        self.iRad = inhb_rad

        # gaussian statistics...
        self.eA   = exe_ampli
        self.iA   = inhb_ampli
        
        # complex neural field parameters ...
        self.Z    = 0.05*np.random.randn(size[0], size[1]) + 0.05*np.random.randn(size[0], size[1])*1j
       
        # lateral connection defination ...
        self.cf      = np.zeros((size[0], size[1], size[0], size[1]), dtype='float64')
        self.cftemp  = np.zeros((size[0], size[1], size[0]+2*self.iRad, size[1]+2*self.iRad), dtype='float64')

        self.Wlat = 0.05*(np.random.randn(size[0], size[1], size[0], size[1]) +\
                               np.random.randn(size[0], size[1], size[0], size[1]))*1j


        # ===================================================================
        for i in range(self.iRad, size[0]+self.iRad):
            for j in range(self.iRad, size[1]+self.iRad):
                # lateral weights
                gauss1, _   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
                self.kernel = gauss1
                self.cftemp[i-self.iRad, j-self.iRad, i-self.iRad:i+self.iRad, j-self.iRad:j+self.iRad] = self.kernel

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                self.cf[i,j,:,:] = self.cftemp[i,j,self.iRad:self.iRad + size[0], self.iRad:self.iRad + size[1]]

        
        print ("MXH After normalization,  max lateral: ", np.max(self.Wlat), "  min lateral: ", np.min(self.Wlat))
        


    def Normalize(self, mat, type_='L1'):
        "performs different types of normalization"

        if type_ == 'L1':
            mat = mat/ np.sum(np.abs(mat))
        elif type_ == 'MinMax':
            mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        elif type_ == 'Zscore':
            mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        elif type_ == 'tanh':
            mat = np.tanh(mat)
        else:
            raise ValueError("Invalid Type found")
        return mat


    def updateLatWeights(self):
        """
        """
        size = self.Z.shape
        deltaw = (np.zeros((size[0], size[1], size[0], size[1])) +\
                        np.zeros((size[0], size[1], size[0], size[1])))*1j
        for i in range(self.Z.shape[0]):
            for j in range(self.Z.shape[1]):
                deltaw[i, j] = config.ita*(self.Z[i,j]*np.conjugate(self.Z) - self.Wlat[i, j])

        self.Wlat += deltaw 
        self.Wlat = self.Normalize(self.Wlat)
        
        pass


    def lateralDynamics(self, temp_aff = 0.2):
        "Dynamics of NFM sheet..."
        temp_lat = np.zeros((self.Z.shape[0], self.Z.shape[1])) + 1j*np.zeros((self.Z.shape[0], self.Z.shape[1])) 

        for i in range(self.Z.shape[0]):
            for j in range(self.Z.shape[1]):
                temp_lat[i, j] = np.mean(self.cf[i,j] * (self.Z*np.conjugate(self.Wlat[i,j]) - self.Z[i,j]*(np.conjugate(self.Wlat[i,j]) + self.Wlat[i,j])))

        Zdot = self.Z*(config.mu - np.abs(self.Z)**2) + config.omega*self.Z *1j + config.eps*temp_aff + temp_lat
        self.Z = self.Z + Zdot*config.dt


    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)
 