import numpy as np
import sys
import socket
import struct
import os
import matplotlib.pyplot as plt
import io
import time
import scipy.signal as signal
import scipy.optimize as opt

samples_file = 'IQ_Samples/Samples(Splotches)'

SAMPLE_RATE = 5.88e6
CENTER_FREQUENCY = 0

PREDICTION_TIMESTEPS = 5

fft_window_samples = []


def getFFT_PSD(samples, num_samples, sample_rate):
    PSD = np.abs(np.fft.fft(samples))**2 / (num_samples*sample_rate)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    return PSD_shifted


def prepSamples(samples, window_size):
    X = []
    y = []
    for i in range(len(samples)-window_size):
        X.append(samples[i:i+window_size])
        y.append(samples[i+window_size])
    return np.array(X), np.array(y)


def readSamplesFromFile(filePath):
    with open(filePath, 'rb') as file:
        print('opening file')
        f = np.arange(SAMPLE_RATE/-2.0, SAMPLE_RATE/2.0, SAMPLE_RATE/1024)
        plt.ion()
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = ax.plot([], [], 'ro-')
        ln2, = ax.plot([], [], 'blue')

        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (Hz)')

        index = 0

        while file.peek(8192):

            # Read 1024 Complex numbers at a time from file
            data = file.read(8192)
            data = struct.unpack('<2048f', data)
            fft_window_samples.append(
                [complex(data[i], data[i+1])for i in range(0, len(data), 2)])

            print(len(fft_window_samples[index]))

            ft, Pxx_den = signal.periodogram(fft_window_samples[index], SAMPLE_RATE, 'flattop', scaling='spectrum')
            Pxx_den_dB = 10.0*np.log10(Pxx_den)
            alpha = normalize_fft_windows(Pxx_den_dB)
            ln.set_xdata(ft)
            ln.set_ydata(Pxx_den_dB)

            ln2.set_xdata(ft)

            x = np.linspace(0, 1, 1024)

            for i in range(x.size):
                x[i] = func(x[i], alpha[0], alpha[1], alpha[2], alpha[3], alpha[4])
            
            ln2.set_ydata(x)
            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            time.sleep(10)
            fig.canvas.flush_events()

            

            index += 1

    # Unpack 2 floats for each complex number
    ## complex(data[i], data[i+1])
    return 1
def func(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def normalize_fft_windows(window):
    q3, q1 = np.percentile(window, [75, 25])
    noise_iqr = q3-q1
    median = np.median(window)
    noise_ceiling = median+noise_iqr
    print(f'-----\nNoise IQR: {noise_iqr} \nmedian: {median}\nNoise Ceiling: {noise_ceiling}\n ')

    array = opt.curve_fit(func, xdata = np.linspace(0, 1, 1024), ydata = window)[0]
    print(array)
    return array

readSamplesFromFile(samples_file)
