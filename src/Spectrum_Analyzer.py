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

PREDICTION_TIMESTEPS = 5

fft_window_samples = []

bitmap_windows = []

#prep samples for model
def prepSamples(samples, window_size):
    X = []
    y = []
    for i in range(len(samples)-window_size):
        X.append(samples[i:i+window_size])
        y.append(samples[i+window_size])
    return np.array(X), np.array(y)

def unpack_bytes_to_complex(bytes):
     data = struct.unpack('<2048f', bytes)
     return [complex(data[i], data[i+1])for i in range(0, len(data), 2)]

def get_spectrum_data(window):
    ft, Pxx_den = signal.periodogram(window, SAMPLE_RATE, 'flattop', scaling='spectrum') # run fft and calculate Power spectral density
    Pxx_den_dB = np.fft.fftshift(10.0*np.log10(Pxx_den))
    ft = np.fft.fftshift(ft)
    alpha , bitmap = normalize_fft_windows(Pxx_den_dB)
    return alpha, bitmap, ft, Pxx_den_dB
     
def func(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e


def apply_threshold(psd, threshold):
     if(psd-threshold > 6):
          return 1
     else:
          return 0

#approximates a curve for the noise, any frequency with 6dB above curve is considered in use 
def normalize_fft_windows(window):
    alpha = opt.curve_fit(func, xdata=np.linspace(0, 1, 1024), ydata=window)[0]
    x = np.linspace(0, 1, 1024)
    for i in range(x.size):
                x[i] = func(x[i], alpha[0], alpha[1],
                            alpha[2], alpha[3], alpha[4])+6
    
    bitmap = list(map(apply_threshold, window, x))
    
    return x, bitmap

#read and process samples from a local binary file in the IQ_Samples directory. Plots the FFT of 1024 complex samples and approximates
# a function for noise. When the signal strength of a given frequency is higher than the threshold mark that frequency as a 1 in the bitmap array
# Returns the coefficients of the aproximated function (alpha), and the bitmap for each frequncy in the fft window (bitmap)
def readSamplesFromFile(filePath):
    with open(filePath, 'rb') as file:
        print('opening file')

        #Set up matplot lib plot
        f = np.arange(SAMPLE_RATE/-2.0, SAMPLE_RATE/2.0, SAMPLE_RATE/1024) #create array for x axis
        plt.ion()
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = ax.plot([], [], 'ro-')
        ln2, = ax.plot([], [], 'blue')
        ln3, = ax.plot([], [], 'green')
        ln4, = ax.plot([], [], 'blue')
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (Hz)')

        index = 0


        while file.peek(8192):

            # Read 1024 Complex numbers at a time from file
            bytes = file.read(8192)
            fft_window_samples.append(unpack_bytes_to_complex(bytes))
            print(len(fft_window_samples[index]))
            
            fitted_curve, bitmap, ft, Pxx_den_dB  = get_spectrum_data(fft_window_samples[index])

            ln.set_xdata(ft)
            ln.set_ydata(Pxx_den_dB)

            ln2.set_xdata(ft)

            ln2.set_ydata(fitted_curve)

            ln3.set_xdata(ft)
            ln3.set_ydata(bitmap)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            time.sleep(2)
            fig.canvas.flush_events()

            bitmap_windows.append(bitmap)

            index += 1

    # Unpack 2 floats for each complex number
    ## complex(data[i], data[i+1])
    return 1

#Polynomial for noise approximation




readSamplesFromFile(samples_file)
