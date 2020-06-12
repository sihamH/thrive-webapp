import numpy as np
import pandas as pd

import librosa

from helpers.wavfilehelper import WavFileHelper
import tools as tl
from pyprojroot import here

######################################################################################################

# find files recursively
rawdatapath =  here() / "data/raw/"
csvdatapath  =  here() / "data/csv/"

# parameters
csvfilename = '/full_dataset'
keep_single = True

# Load files recursively from folder
files = tl.read_file(rawdatapath, keep_single)

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



