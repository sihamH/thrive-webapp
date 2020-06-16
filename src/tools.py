import os
import librosa
import numpy as np
import csv

    
def read_file(path, keep_single):

    labels = []
    files = [] 
    for r, d, f in os.walk(path):
        for file in f:
            if keep_single:
                if (('.wav' in file) or ('.WAV' in file)):
                    files.append(os.path.join(r, file))
            else:    
                if (('.wav' in file) or ('.WAV' in file)) and ('SOUHL' not in r) and ('SAMOHL' not in r):
                    files.append(os.path.join(r, file))

    return files


def load_audio_data(files, savecsv, csvname):

    if savecsv:
        # csv header
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate onset pitches magnitudes'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        # create csv file
        file = open(csvname+'.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        
    for ff in files:
        # Load audio
        print(ff)
        y, sr = librosa.load(ff, mono=True, duration=30)
        rmse = librosa.feature.rmse(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # append labels
        if len(ff.split('/')) == 1:
            fname = ff.split('/')[0]
        else:
            fname = ff.split('/')[-3] + '_' + ff.split('/')[-2] +'_'+ ff.split('/').pop()
        
        to_append = f'{fname} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(onset)} {np.mean(pitches)} {np.mean(magnitudes)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        if 'Healthy' in ff:   
            c = 'H'
            to_append += f' {c}'
        else:
            c = 'P'
            to_append += f' {c}'

        # save to csv
        if savecsv:
            file = open(csvname+'.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

    return to_append


def prep_for_modeling(df):

    #Encoding the Labels
    labels = df.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    #Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype = float))

    return X, y    



    