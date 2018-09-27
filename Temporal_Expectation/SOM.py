import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from tqdm import tqdm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from config import Config
from Gratings import *
# from keras.datasets import mnist

# Gstat  = GaussianStatistics()
Gstat  = Gratings()

config = Config()


class SOM():
    """
    """
    def __init__(self,  output_size=(10,10), data=np.random.rand(100,100), epochs=1, learning_rate=1e-3):
        self.data = data
        self.iterations = len(data)*epochs
        self.output_size = output_size
        self.input_num = data.shape[1]
        self.nrad  = 2 #int...
        self.sig   = 0.5
        self.alpha = learning_rate
        self.weights = np.random.rand(self.output_size[0], self.output_size[1], self.input_num)

    def Normalize(self, mat):
        # mat = mat/ np.max(mat)
        # mat = mat / np.sum(abs(mat))
        mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        # mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        return mat

    def nstnbridx(self, row_data):
        initial_dis = float("inf")
        index_bmu   = [0, 0]
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                dist_neuron = np.linalg.norm(row_data-self.weights[i, j, :])
                if dist_neuron < initial_dis:
                    initial_dis = dist_neuron
                    index_bmu = [i, j]# Best matching unit (BMU)
        return np.array(index_bmu)

    def nbridxdis(self, index_bmu):
        nbrind = np.zeros(((2*self.nrad + 1)**2, 2))
        for i in range(2*self.nrad + 1):
            for j in range(2*self.nrad + 1):
                ix = i*(2*self.nrad+1) + j
                nbrind[ix,:] = [i - self.nrad, j- self.nrad] + index_bmu

        # print (nbrind, (nbrind[:,1] >= 0))
        nbrind = nbrind[np.where((nbrind[:,0] >= 0) * (nbrind[:,1] >= 0))]
        nbrind = nbrind[np.where((nbrind[:,0] < self.output_size[0]) * (nbrind[:,1] < self.output_size[1]))]

        mm, _  = nbrind.shape
        nbrdist = np.zeros(mm)
        for i in range(mm):
            diff = nbrind[i,:] - index_bmu
            nbrdist[i] = diff.dot(diff)
        return nbrind, nbrdist

    def fit(self):
        """
        """
        for itter in tqdm(range(self.iterations)):
            initial_dis = float("inf")
            row_index = np.random.randint(len(self.data))
            learning_rate = self.alpha*np.exp(-itter/self.iterations)
            row_data = self.data[row_index]
            bmu_idx  = self.nstnbridx(row_data)
            nbrind, nbrdist = self.nbridxdis(bmu_idx)
            mx, _ = nbrind.shape
            for i in range(mx):
                idx = nbrind[i,:]
                wt  = self.weights[int(idx[0]), int(idx[1]), :]
                diff = row_data - wt
                dis  = nbrdist[i]/self.sig **2
                delta = learning_rate*np.exp(-dis)*diff
                self.weights[int(idx[0]), int(idx[1]), :] = delta + wt
        print ("SOM Training done!!...")
        pass

    def response(self, X, wt, sig = 1.5):
        """
        """
        x = X.flatten('F')
        assert len(x) == wt.shape[2]
        Y = np.zeros(self.output_size)

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                diff = wt[i, j, :] - x
                dis  = diff.dot(diff)
                Y[i, j] = np.exp(-1.*dis / sig**2)

        return self.Normalize(Y)

    def view_weights(self):
        """
        """
        shape = self.weights.shape
        m = int(np.sqrt(shape[2]))
        img = np.zeros((shape[0]*m, shape[1]*m))
        for i in range(shape[0]):
            for j in range(shape[1]):
                img[i*m:(i+1)*m, j*m:(j+1)*m] = self.weights[i,j,:].reshape(m, m)

        plt.imshow(img)
        plt.show()

    def moveresp(self, display=True):
        """
        """
        plt.ion()
        for x in self.data:
            N = np.sqrt(len(x))
            X = x.reshape(int(N), int(N), order='F')
            y = self.response(X, self.weights)
            if display:
                plt.imshow(y)
                plt.pause(1)
        pass

    def load_weights(self, path):
        self.weights = np.load(path)

    def get_weights(self):
        """
        """
        np.save(config.SOM_weights_path, self.weights)
        return self.weights



if __name__ == '__main__':
    ## Data Generation....
    data = []
    for angle in range(0, 180, 2):
        _bar = Gstat.fixedGrating(N = config.N,
                                    theta = angle,
                                    display = False)
        data.append(_bar.flatten('F'))
    data = np.array(data)

    grts  = Gratings()
    # data.append(grts.fixedGrating(theta = 0, display = True).flatten('F'))
    # data.append(np.random.randn(10,10).flatten('F'))
    # data.append(Gstat.OrientationBar(N = config.N,theta = 0, display = True).flatten('F'))
    # data = np.array(data)

    SOM = SOM((10,10), data, 500, 0.01)
    SOM.fit()
    SOM.load_weights('./SOM_weights.npy')
    SOM.moveresp()
    SOM.view_weights()

    # Show trained weights
    # print "Trained weights from SOM,", SOM.get_weights()