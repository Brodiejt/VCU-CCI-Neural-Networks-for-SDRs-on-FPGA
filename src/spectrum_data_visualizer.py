import numpy as np
import sys
import socket
import struct
import os
import matplotlib.pyplot as plt
import io
import time

samples_file = 'IQ_Samples\Samples(Splotches)'

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

            array = np.array(getFFT_PSD(
                fft_window_samples[index], 1024, SAMPLE_RATE))

            xdata = f
            ydata = array

            ln.set_xdata(xdata)
            ln.set_ydata(ydata)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            time.sleep(0.05)
            fig.canvas.flush_events()
            index += 1

    # Unpack 2 floats for each complex number
    ## complex(data[i], data[i+1])
    return 1


readSamplesFromFile(samples_file)
