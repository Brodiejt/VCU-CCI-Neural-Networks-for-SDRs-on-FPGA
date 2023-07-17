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
import matlab.engine
import socket

samples_file = 'src/Binary_IQ_Samples/Samples(Splotches)'
SAMPLE_RATE = 4.88e6 
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
    ft, Pxx_den = signal.periodogram( window, SAMPLE_RATE, 'hamming', scaling='spectrum') # run fft and calculate Power spectral density
    Pxx_den_dB = np.fft.fftshift(10.0*np.log10(Pxx_den)) #convert fft to dB and shift
    ft = np.fft.fftshift(ft)
    noise_curve , bitmap = normalize_fft_windows(Pxx_den_dB)
    return noise_curve, bitmap, ft, Pxx_den_dB

#Polynomial for noise approximation
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

def get_bitmaps_from_file(filepath):
    index = 0
    bitmap_windows = []
    with open(filepath, 'rb') as file:
        while file.peek(8192):
            bytes = file.read(8192)
            fft_window_samples.append(unpack_bytes_to_complex(bytes))
            print(len(fft_window_samples[index]))

            fitted_curve, bitmap, ft, Pxx_den_dB  = get_spectrum_data(fft_window_samples[index])
            bitmap_windows.append(bitmap)
            index += 1
    return bitmap_windows

#read and process samples from a local binary file in the IQ_Samples directory. Plots the FFT of 1024 complex samples and approximates
# a function for noise. When the signal strength of a given frequency is higher than the threshold mark that frequency as a 1 in the bitmap array
def VisualizeSamplesFromFile(filePath):
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

            fig.canvas.draw()
            time.sleep(1)
            fig.canvas.flush_events()

            bitmap_windows.append(bitmap)

            index += 1

    # Unpack 2 floats for each complex number
    ## complex(data[i], data[i+1])
    return 1

def VisualizeSamplesFromServer(host, port):
     #Set up matplot lib plot
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.connect((host, port))
          s.setblocking(True)

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
          print(f'Connecting server IP:{host}, Port:{port}')
          
          bytes = []
          currTime = time.time()
          while 1:
               data = s.recv(8192)
               if(time.time() - currTime > 0.1):
                  bytes = data
                  currTime = time.time()
               else:
                    continue
               
               if(len(bytes) < 8192):
                    continue
               

            #    print(len(bytes))
               window = unpack_bytes_to_complex(bytes)
               fitted_curve, bitmap, ft, Pxx_den_dB  = get_spectrum_data(window)


               ln.set_xdata(ft)
               ln.set_ydata(Pxx_den_dB)
   
               ln2.set_xdata(ft)
   
               ln2.set_ydata(fitted_curve)
   
               ln3.set_xdata(ft)
               ln3.set_ydata(bitmap)
   
               ax.relim()
               #ax.autoscale_view()
   
               fig.canvas.draw()
               fig.canvas.flush_events()

VisualizeSamplesFromServer('127.0.0.1', 2000)

# eng = matlab.engine.start_matlab()
# training_data = get_bitmaps_from_file(samples_file)
# eng.workspace['bitmaps'] = training_data
# eng.save('spectrum_bitmap_data.mat','bitmaps',nargout=0)
# eng.exit()