# 🎵 Music Genre Classification Web App

A professional Deep Learning project that identifies music genres from audio files (WAV/MP3) using a Convolutional Neural Network (CNN).

## 🚀 Live Demo
[Paste your Render link here once it's live!]

## 🛠️ Features
* **Dual Format Support:** Accepts both .wav and .mp3 files.
* **Visual Analysis:** Generates a Mel Spectrogram for every upload.
* **Modern UI:** Clean, responsive interface built with Flask and CSS.

## 📊 How it Works
1. **Upload:** User provides an audio track.
2. **Process:** System extracts a 10-second clip and converts it to a Mel Spectrogram image.
3. **Classify:** The CNN model analyzes the image to predict the genre.

## 📁 Project Structure
- `app.py`: Flask backend and prediction logic.
- `MLPmusicgen.h5`: Trained CNN model weights.
- `static/`: Contains CSS, Images, and generated Spectrograms.
- `templates/`: HTML files for the web interface.
