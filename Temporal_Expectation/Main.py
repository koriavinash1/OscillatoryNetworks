import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from tqdm import tqdm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from Gratings import *
from Oscillator import CoupledNFM
from config import Config
from SOM import *
import pdb

Gstat = GaussianStatistics()
config= Config()
grts  = Gratings()
som   = SOM()
somwts= np.load('./SOM_weights.npy') 

rand = lambda N: np.random.randn(N, N)

class Main(object):
    def __init__(self, deltaT):
        self.deltaT = deltaT
        self.Flag   = False
        self.Flagcount  = 0
        self.oscillator = CoupledNFM()
        pass


    def display(self, Z, i, fig,  _type = '3d'):
        plt.clf()
        plt.ion()
        if _type == '3d':
            X = np.arange(0, config.N)
            Y = np.arange(0, config.N)
            X, Y = np.meshgrid(X, Y)
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


    def runNFM(self, _display = False, blink=True, random = False):
        """
        """
        # plt.ion()
        self.Zs     = np.zeros((config.N, config.N, int(config.T/config.dt)), dtype='complex64')
        for t in range(0, int(config.T/config.dt), 1):
            
            if t % config.deltaT == 0:
                temp_aff = rand(config.N) # som.response(rand(config.N), somwts)
                if random: config.deltaT = np.random.randint(800, 2000)

            if blink:
                if (np.random.uniform(0, 1) > 0.5) and t > config.TrainingTime:
                    self.Flag = True

                if self.Flag and self.Flagcount < config.deltaT:
                    if (np.random.uniform(0, 1) > 0.5): temp_aff = som.response(grts.fixedGrating(theta = 45), somwts) # [45, -45]
                    else: temp_aff = som.response(grts.fixedGrating(theta = -45), somwts) # [45, -45]
                    self.Flagcount += 1
                else:
                    self.Flag = False

            # lateralTraining ...
            if t < config.TrainingTime:
                if (np.random.uniform(0, 1) > 0.5): temp_aff = som.response(grts.fixedGrating(theta = 45), somwts) # [45, -45]
                else: temp_aff = som.response(grts.fixedGrating(theta = -45), somwts) # [45, -45]

            # if t % 200 == 0:
            #     plt.imshow(temp_aff)
            #     plt.title(t)
            #     plt.pause(0.005)

            self.oscillator.lateralDynamics(temp_aff)
            if (self.Flag and self.Flagcount < config.deltaT) or t < config.TrainingTime: 
                self.oscillator.updateLatWeights()
            
            self.Zs[:,:, t] = self.oscillator.Z

            if _display and t % 300 == 0: self.display(np.real(self.oscillator.Z), t, plt.figure('plot'))

        self.Zs = self.Zs[:, :, config.TrainingTime :]
        return self.Zs


    def filteringSignal(self):
        """
            Delta wave filtering...
            (0 - 4 Hz)
        """
        from scipy.signal import butter, lfilter

        filtered_ZsR = np.zeros((config.N, config.N, int(config.T/config.dt)))
        filtered_ZsI = np.zeros((config.N, config.N, int(config.T/config.dt)))

        fs = 100
        N = 10000
        

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            N, Wn = signal.buttord([1, 8], [2, 10], 1, 10, True)
            b, a = butter(N, Wn, btype='band', analog=False)
            return b, a


        def butter_bandpass_filter(data, fs = 100, lowcut = 0, highcut = 5, order=2):
            x = data
            freqs = np.fft.fftfreq(len(data), 1/fs)
            idx = np.argsort(freqs)
            ps = np.abs(np.fft.fft(x))**2
            plt.figure()
            plt.plot(freqs[idx], ps[idx])
            plt.title('Power spectrum (np.fft.fft)')

            f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
            plt.figure()
            plt.semilogy(f, np.sqrt(Pxx_spec))
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V RMS]')
            plt.title('Power spectrum (scipy.signal.welch)')
            plt.show()
            return x

        for xx in range(config.N):
            for yy in range(config.N):
                filtered_ZsR[xx, yy, :] = butter_bandpass_filter(np.real(self.Zs[xx, yy, :]))
                filtered_ZsI[xx, yy, :] = butter_bandpass_filter(np.imag(self.Zs[xx, yy, :]))

        return filtered_ZsR + 1j * filtered_ZsI


    def viewSignals(self, data):
        """
            data format complex number
        """
        # x = np.random.randint(0, config.N, 5)
        # y = np.random.randint(0, config.N, 5)

        # plt.figure('Real')
        # for i in range(5):
        #   plt.subplot(5, 1, i + 1)
        #   plt.plot(np.real(data[x[i], y[i], :]))
        #   plt.title("x:" + str(x[i]) + "  y:" + str(y[i]))

        # plt.figure('Imaginary')
        # for i in range(5):
        #   plt.subplot(5, 1, i + 1)
        #   plt.plot(np.imag(data[x[i], y[i], :]))
        #   plt.title("x:" + str(x[i]) + "  y:" + str(y[i]))

        signal = np.mean(data, axis=(0,1)) 
        plt.subplot(2,1,1)
        plt.plot(np.real(signal))
        plt.subplot(2,1,2)
        plt.plot(np.imag(signal))
        plt.show()
        pass


    def classifier(self, data):
        """
            delta Signal classifier 
        """
        # based on visualization determine threshold..
        signal = np.mean(data, axis=(0,1))
        print (np.mean(np.real(signal)), np.var(np.real(signal)))
        print (np.mean(np.imag(signal)), np.var(np.imag(signal)))

        if np.var(np.abs(np.real(signal))) >= config.Thresh and np.var(np.abs(np.imag(signal))) >= config.Thresh:
            return False

        else: return True


    def performExp(self, blink=True, random = False):
        """
        NFM => Filter => classifier
        """
        Z  = self.runNFM(blink=blink, random = random)
        # plt.plot(Z[5,5,:])
        # plt.show()
        # fZ = self.filteringSignal()
        # self.viewSignals(Z)
        print (self.classifier(Z))
        return np.var(np.abs(np.real(np.mean(Z, axis=(0,1)))))


if __name__ == '__main__':
    threshFalse = []
    for _ in tqdm(range(15)):
        exp = Main(config.deltaT)
        threshFalse.append(exp.performExp(blink = False))

    threshTrue = []
    for _ in tqdm(range(15)):
        exp = Main(config.deltaT)
        threshTrue.append(exp.performExp(blink = True, random = True))

    plt.plot(np.array(threshTrue), 'r')
    plt.plot(np.array(threshFalse), 'b')
    plt.show()
    print (np.mean(np.array(threshFalse)), np.mean(np.array(threshTrue)))