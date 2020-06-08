'''
Name 0.0

Web Application for Insight Health Data Science Fellowship,
Boston, MA -- 2020A

Author: Siham Hachi

Last Revision: 2020.06.03

'''


import sys
sys.path.insert(1, './src')

import streamlit as st
import librosa
import tools as tl
import csv
import pandas as pd
import numpy as np
import os
import io
import prediction
from pyprojroot import here
import streamlit_theme as stt

import shutil

from typing import Dict


#st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)
#st.markdown("<h1 style='text-align: center;' class='test'>Welcome to SpeaKid!</h1>", unsafe_allow_html=True)

stt.set_theme({'primary': '#63d297'})
st.title("Welcome to Thrive!")
st.markdown(
"""
This is a demo of a Streamlit app that recognises children with speech pathology 
based on morpheme recordings. Upload an audio file below.
""")

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def main():
    """Run this function to run the app"""
    static_store = get_static_store()

    input_buffer = st.file_uploader("Upload", type="wav")
    if input_buffer:
        # Process you file here
        value = input_buffer.getvalue()

        # And add it to the static_store if not already in
        if not value in static_store.values():
            static_store[input_buffer] = value

        filename = 'testSave.wav'
        input_buffer.seek(0)
        with open(filename, 'wb') as f:
            shutil.copyfileobj(input_buffer, f)

        y, sr = librosa.load(filename, mono=True, duration=30)

        features = tl.load_audio_data([filename], savecsv=0, csvname='')
        modelpath = here() / 'models/DLmodel.h5'
        p = prediction.predict_NN(features, modelpath)

        if p == 0:
            st.title('There seems to be speech disorder')
        else: st.title('No speech disorder has been detected')

        os.remove(filename)


main()            



    



