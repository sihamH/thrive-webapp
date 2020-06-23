import sys
sys.path.insert(1, './src')
sys.path.insert(1, './models')
import os

from flask import Flask, render_template, request

import librosa
import tools as tl
import prediction

from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/how')
def how():
    return render_template('how.html')    

@app.route('/upload')
def upload():
    return render_template('upload.html')       

@app.route('/results', methods = ['GET', 'POST'])
def results():

    f = request.files['audio']
    if f.filename != '':
        f.save(secure_filename(f.filename))
        y, sr = librosa.load(f.filename, mono=True, duration=30)
        features = tl.load_audio_data([f.filename], savecsv=0, csvname='')
        modelpath = './models/DLmodel.h5'
        p = prediction.predict_NN(features, modelpath)

        if p == 1:
            return render_template('results.html', upload_text='There seems to be speech disorder', toto="red")
        else:
            return render_template('results.html', upload_text2='No speech disorder has been detected', toto2="green")
        os.remove()
    else:
        return render_template('upload.html', upload_text='Please upload a file') 
        
@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')  
    
if __name__ == '__main__':
   app.run(debug = True)