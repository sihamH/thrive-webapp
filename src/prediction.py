from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tools as tl
from pyprojroot import here


def predict_NN(ft, modelpath):
    ft = ft.split()
    ft = ft[1:-1]

    X = np.asarray(ft, dtype=float)
    X = np.expand_dims(X, axis=1)
    X = np.transpose(X)

    # Prediction   
    model = tf.keras.models.load_model(modelpath)
    #model._make_predict_function()
    p = model.predict(X)
    tf.keras.backend.clear_session()

    return p


#filepath  = here() / 'data/raw/Healthy/H26/03_1SL/4PK3KRK.wav'
#features = tl.load_audio_data(files=[str(filepath)], savecsv=0
#                            , csvname = 'test')

#modelpath = here() / 'models/DLmodel.h5'
#p = predict_NN(features, modelpath)    
#print(p)
