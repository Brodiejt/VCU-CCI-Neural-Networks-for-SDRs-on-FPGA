import matlab.engine
import Spectrum_Analyzer as SA

mode = 0 #1 = Server, 0 = File
architecture = 'matlab' #matlab or tensorflow

samples_file = 'IQ_Samples/Samples(Splotches)'
SAMPLE_RATE = 5.88e6 

PREDICTION_TIMESTEPS = 5

fft_window_samples = []



def get_bitmaps_from_file(filepath):
    index = 0
    bitmap_windows = []
    with open(filepath, 'rb') as file:
        while file.peek(8192):
            bytes = file.read(8192)
            fft_window_samples.append(SA.unpack_bytes_to_complex(bytes))
            print(len(fft_window_samples[index]))

            fitted_curve, bitmap, ft, Pxx_den_dB  = SA.get_spectrum_data(fft_window_samples[index])
            bitmap_windows.append(bitmap)
            index += 1
    return bitmap_windows



if architecture == 'matlab':
    if mode:
        pass
    else:
        eng = matlab.engine.start_matlab()
        training_data = get_bitmaps_from_file(samples_file)
        eng.workspace['bitmaps'] = training_data
        eng.save('spectrum_bitmap_data.mat','bitmaps')
        eng.exit()
        




