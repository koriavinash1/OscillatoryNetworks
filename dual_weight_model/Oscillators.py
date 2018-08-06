import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from SOM import SOM
import pdb
Gstat = GaussianStatistics()
config= Config()


OrientationBars = np.load('./Orientation_bars.npy')
images = OrientationBars.reshape(-1, 10, 10)

# twts = np.zeros((10, 10, 10, 10))
# for i in range(10):
#     for j in range(10):
#         ai = np.random.randint(0, 4)
#         twts[i, j, :, :] = images[ai*22]

# b = np.empty((100, 100))
# for i in range(10):
#     for j in range(10):
#         b[i*10:(i+1)*10, j*10:(j+1)*10] = twts[i,j,:,:]

# plt.imshow(b)
# plt.show()


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
        self.Wlat     = np.zeros((size[0], size[1], size[0], size[1]), dtype='complex64')
        self.Wlattemp = np.zeros((size[0], size[1], size[0]+2*self.iRad, size[1]+2*self.iRad), dtype='complex64')
        
        # ===================================================================
        # weight initialization with real and imaginary maxican hat
        for i in range(self.iRad, size[0]+self.iRad):
            for j in range(self.iRad, size[1]+self.iRad):
                # lateral weights
                gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
                # Gstat.Visualize(gauss1+gauss2, _type = '2d')
                self.kernel= (gauss1 - gauss2) * np.exp(( - gauss1 + gauss2)*1j)
                self.Wlattemp[i-self.iRad, j-self.iRad, i-self.iRad:i+self.iRad, j-self.iRad:j+self.iRad] = self.kernel

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                self.Wlat[i,j,:,:] = self.Wlattemp[i,j,self.iRad:self.iRad + size[0], self.iRad:self.iRad + size[1]]
        
        # normalization L1
        for ii in range(10):
            for jj in range(10):
                # plt.imshow(self.Wlat[ii,jj,:,:])
                # plt.show()
                self.Wlat[ii, jj, :, :]  = self.Normalize(self.Wlat[ii, jj, :, :])
        
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


    def lateralDynamics(self, aff = 2+2j, verbose = True, ci=config.ci):
        "Dynamics of NFM sheet..."
        aff = np.zeros((10,10)) + 1j*np.zeros((10,10)) + 0.1 + 0.1j
        aff[3,3] = 1 + 1j
        # temp_lat = np.zeros_like(self.Z)
        # for m in range(self.Z.shape[0]):
        #     for n in range(self.Z.shape[1]):
        #         temp_lat[m,n] = np.sum((np.angl(self.Z) - np.angle(self.Z[m,n]))*self.Wlat[m, n, :, :])
        
        # high self weights ...
        # temp_lat = temp_lat + np.real(self.Z)
        temp_lat = np.sum(self.Z.reshape(self.Z.shape[0], self.Z.shape[1], 1, 1)*self.Wlat, axis=(2, 3))
        temp_lat = temp_lat if not np.isnan(temp_lat).any() else 0.0
        # temp_lat = self.Normalize(temp_lat)

        temp_aff = 0.5*aff
        temp_lat = 0.05*temp_lat
        I = temp_lat + temp_aff
        
        a = 0.2 + 0.01j  # inc img part for depolarization 
        c = 0.01 + 1j 

        Z = self.Z
        Zstar = np.conjugate(Z)

        x = (Z + Zstar)/2.
        y = (Z - Zstar)/2.

        Zdot = a * Zstar + a * Z *1j + c * ((5.0/24.0)*Z + (1.0/8.0) * Zstar)*Z*Zstar + I
        

        Z = Z + Zdot*config.dt
        # Z[np.abs(Z) > 1e1] = 1. + 1.j
        
        self.Z = Z


    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)