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
np.random.seed(2018)

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout

Gstat = GaussianStatistics()
config= Config()
grts  = Gratings()
som   = SOM()
somwts= np.load('./SOM_weights.npy') 

rand = lambda N: np.random.randn(N, N)

class Main(object):
    def __init__(self):
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


    def runNFM(self, _display = True, 
                    blink     = True, 
                    random    = config.random,
                    generate  = config.generate):
        """
        # plt.ion()
        """

        data = {'spatio_temporal_wave': [], 'label':[], 'base_wave': []} 
        self.Zs     = np.zeros((config.N, config.N, int(config.T/config.dt)), dtype='complex64')
        label = [1, 0]

        for t in tqdm(range(0, int(config.T/config.dt), 1)): 

            # aff. input ...
            if t % config.gdeltaT == 0:
                temp_aff = rand(config.N) # som.response(rand(config.N), somwts)
                label = [1, 0]
                if random: config.gdeltaT = np.random.randint(800, 2000)

                T = False
                if blink and t != 0: T = (np.random.uniform(0, 1) > 0.5)
                
                if blink and T :
                    temp_aff = som.response(grts.fixedGrating(theta = 45), somwts) # [45, -45]
                    # else: temp_aff = som.response(grts.fixedGrating(theta = -45), somwts) # [45, -45]
                    label = [0, 1]

            # lateralTraining ...
            if t < config.TrainingTime:
                temp_aff = som.response(grts.fixedGrating(theta = 45), somwts) # [45, -45]
                # else: temp_aff = som.response(grts.fixedGrating(theta = -45), somwts) # [45, -45]

            self.oscillator.lateralDynamics(temp_aff)

            if t < config.TrainingTime: 
                self.oscillator.updateLatWeights()
                if config.saveLat: np.save('./latweights.npy', self.oscillator.Wlat)

            self.Zs[:,:, t] = self.oscillator.Z

            data['spatio_temporal_wave'].append(self.oscillator.Z)
            data['base_wave'].append(np.mean(self.oscillator.Z))
            data['label'].append(label)

            if _display and t % 100 == 0: self.display(np.real(self.oscillator.Z), t, plt.figure('plot'))

        # save data to train classifier 
        if generate:
            np.save('Xtrain.npy', np.array(data['spatio_temporal_wave']))
            np.save('Ytrain.npy', np.array(data['label']))
            print (np.array(data['label']).shape)

        self.Zs = self.Zs[:, :, config.TrainingTime :]
        return data


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

    def classifier_net(self):
        model = Sequential([
                    Dense(28, input_shape=(100,), activation = 'relu'), # TODO:
                    Dense(2),
                    Activation('sigmoid')
                ])
        return model

    def fit_classifier(self):
        Xdata = np.load('./Xtrain.npy').reshape(-1, 100)
        Ydata = np.load('./Ytrain.npy')
        total = len(Xdata)
        Xtrain, Ytrain = Xdata[: int(0.9*total)], Ydata[: int(0.9*total)]
        Xtest, Ytest = Xdata[int(0.9*total):], Ydata[int(0.9*total):]

        print (Ytest.shape, Xtest.shape)
        model = self.classifier_net()
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy', 'mse'])

        model.fit(Xtrain, Ytrain, epochs=config.epoch, batch_size=128, validation_data=(Xtest, Ytest))

        score = model.evaluate(Xtest, Ytest, batch_size=128)
        print("Performance on held out test data: {}".format(score))

        model.save('./model.h5')
        return model

    def infer_classifier(self, x):
        model = load_model('model.h5')
        y = model.predict(x)[0]
        return y[0], y[1]

    def classifier(self, data):
        """
            delta Signal classifier 
        """
        # based on visualization determine threshold..
        true, false = [], []
        for i in tqdm(range(data.shape[2])):
            x = data[i, :,:]
            xp, yp = self.infer_classifier(x.reshape(1, 100))
            true.append(xp)
            false.append(yp)

        return np.array(true), np.array(false)


    def performExp(self, blink=True, random = False):
        """
        NFM => Filter => classifier
        """
        data   = self.runNFM(blink=blink, random = random)
        tp, fp = self.classifier(np.array(data['spatio_temporal_wave']))    
        base_wave = data['base_wave']
        
        plt.subplot(4, 1, 1)
        plt.plot(tp,'r')
        
        plt.subplot(4, 1, 2)
        plt.plot(fp,'b')
        
        plt.subplot(4, 1, 3)
        plt.plot(base_wave)

        plt.subplot(4, 1, 4)
        print np.array(data['label']).shape
        plt.plot(np.array(data['label'])[:, 1],'g')
        plt.show()
        pass


if __name__ == '__main__':
    exp = Main()
    # exp.runNFM()
    if config.Train: exp.fit_classifier()
    exp.performExp()

    # threshFalse = []
    # for _ in tqdm(range(15)):
    #     exp = Main()
    #     threshFalse.append(exp.performExp(blink = True))

    # threshTrue = []
    # for _ in tqdm(range(15)):
    #     exp = Main()
    #     threshTrue.append(exp.performExp(blink = True, random = True))

    # plt.plot(np.array(threshTrue), 'r')
    # plt.plot(np.array(threshFalse), 'b')
    # plt.show()
    # print (np.mean(np.array(threshFalse)), np.mean(np.array(threshTrue)))