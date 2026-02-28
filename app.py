import static_ffmpeg
static_ffmpeg.add_paths()
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from flask_compress import Compress
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from werkzeug.utils import secure_filename
import librosa
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pydub import AudioSegment
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from tensorflow.keras.initializers import glorot_uniform
import time


app = Flask(__name__) 
Compress(app)

@app.route('/')
@app.route('/index')
def index():
    return render_template('genre_prediction.html') # This makes it the landing page

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')  

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        return render_template("preview.html",df_view = df)


#  multiple custom layers 
def GenreModel(input_shape = (288,432,4),classes=9):
    
    
    X_input = Input(input_shape)

    X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(256,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    
    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

    model = Model(inputs=X_input,outputs=X,name='GenreModel')

    return model

model = GenreModel(input_shape=(288,432,4),classes=9)
model.load_weights("MLPmusicgen.h5")


@app.route('/genre_prediction')
def genre_prediction():
    return render_template('genre_prediction.html')

class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']

def convert_mp3_to_wav(music_file):
    sound = AudioSegment.from_file(music_file)
    sound.export("static/music_file.wav", format="wav")

def extract_relevant(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    relevant_segment = wav[t1 * 1000: t2 * 1000]
    relevant_segment.export("extracted.wav", format='wav')

def predicts(image_data, model):
    image_array = img_to_array(image_data)
    print("Original image_array shape:", image_array.shape)
    image_array = np.reshape(image_array, (1, 288, 432, 4))
    prediction = model.predict(image_array / 255.0)
    class_label = np.argmax(prediction)
    return class_label, prediction

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # 1. Get the file from the website
    audio_file = request.files['audiofile']
    
    # 2. Save the original file (MP3 or WAV) to the tests folder
    m_path = os.path.join('static/tests/', secure_filename(audio_file.filename))
    audio_file.save(m_path)
    
    # 3. Handle MP3 or WAV conversion
    # We use from_file because it handles both formats easily
# Check if the file is an MP3
    if m_path.lower().endswith(".mp3"):
    # Convert MP3 to WAV using pydub
        sound = AudioSegment.from_mp3(m_path)
        sound.export("static/temp_converted.wav", format="wav")
    # Load the newly created WAV file
        wav = AudioSegment.from_wav("static/temp_converted.wav")
    else:
    # It is already a WAV, load it normally
        wav = AudioSegment.from_wav(m_path)

# Now take the 10-second clip (from 40s to 50s)
    relevant_segment = wav[40 * 1000: 50 * 1000]
    relevant_segment.export("static/extracted.wav", format="wav")

    # 5. Create the Mel Spectrogram (The Image for the AI)
    y, sr = librosa.load('static/extracted.wav', duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.Figure()
    plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('static/melspectrogram.png')    

    # 6. Feed the image to the AI model
    image_data = load_img('static/melspectrogram.png', color_mode='rgba', target_size=(288, 432))
    class_label, prediction = predicts(image_data, model)
    genre = class_labels[class_label]
    
    return render_template("genre_prediction.html", prediction=genre, audio_path=m_path)

@app.route('/chart')
def chart():
	return render_template('chart.html')

@app.route('/performance')
def performance():
	return render_template('performance.html')

if __name__ == "__main__":
    app.run()
