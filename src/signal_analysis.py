import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import wave

import librosa
import librosa.display

from helpers.wavfilehelper import WavFileHelper
import tools as tl
import IPython.display as ipd # Display audio signal
from pyprojroot import here

import sklearn

######################################################################################################

# find files recursively
rawdatapath =  here() / "data/raw/"
csvdatapath  =  here() / "data/csv/"

# parameters
csvfilename = '/full_dataset'
keep_single = True

#filefinder = RecursiveFileFinder()
files = tl.read_file(rawdatapath, keep_single)

print("The dataset contains {} recordings".format(len(files)))
print("Number of recordings from patients {}".format(sum('/Patients/' in f for f in files)))
print("Number of recordings from healthy subjects {}".format(sum('/Healthy/' in f for f in files)))


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
lables = ['Patient', 'Healthy']
counts = [sum('/Patients/' in f for f in files), sum('/Healthy/' in f for f in files)]
ax.bar(lables,counts)
#plt.show()

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
data = tl.load_audio_data(files, savecsv=1
                        , csvname = str(csvdatapath)+csvfilename)



