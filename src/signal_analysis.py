import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import wave

import librosa
import librosa.display

from helpers.wavfilehelper import WavFileHelper
import helpers.tools as tl
import IPython.display as ipd # Display audio signal
from pyprojroot import here

import sklearn

######################################################################################################

# find files recursively
rawdatapath =  here() / "data/raw/"
csvdatapath  =  here() / "data/csv/"

#filefinder = RecursiveFileFinder()
files = tl.read_file(rawdatapath)

print("The dataset contains {} recordings".format(len(files)))
print("Number of recordings from patients {}".format(sum('/Patients/' in f for f in files)))
print("Number of recordings from healthy subjects {}".format(sum('/Healthy/' in f for f in files)))

################################### Recording features ################################################

wavfilehelper = WavFileHelper()

audiodata = []
for file_name in files:    
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

print(audiodf.num_channels.value_counts(normalize=True)) # Channels
print(audiodf.sample_rate.value_counts(normalize=True)) # Sample Rate
print(audiodf.bit_depth.value_counts(normalize=True)) # Bit depth

################################### Spectral features ################################################

x, sr = librosa.load(files[0])
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)

# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')

#--------------------------------------- Spectrogram -------------------------------------------------
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
'''
for fn in files:
    # load & augment
    x, sr = librosa.load(fn)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # spectrogram
    plt.figure(figsize=(10,10))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    
    # build filenames
    fn = fn.lower()
    fname = fn.split('/')[-3] + '_' + fn.split('/')[-2] +'_'+ fn.split('/').pop().replace('.wav','.png')
    #plotpath = '../data_LANNA/ProcessedOriginal/'+fname
    
    # save
    plt.axis('off');
    #plt.savefig(plotpath)                
    plt.clf()

'''
#----------------------------------- MFCC features ---------------------------------------
data = tl.load_audio_data(files, savecsv=1
                        , csvname = str(csvdatapath)+'/datasettest')



