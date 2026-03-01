# 🎧 Music Genre Prediction (CNN)

A Flask-based deep learning application that translates audio signals into visual data to classify music genres. Built to handle real-world audio formats and deployed on high-RAM infrastructure for stable inference.

## 🔗 Live Application
[View the App on Hugging Face Spaces](https://huggingface.co/spaces/VishwaVidhatha/Music-Genre-Prediction)

## 🏗️ Engineering for Stability
This project was migrated from standard cloud hosting to Hugging Face Spaces to overcome specific hardware limitations:
* **Memory Management:** Utilizes 16GB RAM to support heavy TensorFlow model loading without 502/500 errors.
* **Audio Engine:** Leverages a Dockerized FFmpeg environment for reliable MP3/WAV processing.
* **Deployment Efficiency:** Uses `.renderignore` and optimized Docker layers to manage a 4.6GB training dataset without bloating the production container.

## 📊 How it Works
1. **Audio to Image:** The system takes a 10-second clip and generates a Mel Spectrogram.
2. **CNN Inference:** A Convolutional Neural Network (CNN) treats the spectrogram as an image to detect frequency patterns.
3. **Classification:** Results are mapped across 9 distinct genres including Blues, Metal, Pop, and Rock.



## 🛠️ Tech Stack
* **Backend:** Flask, Gunicorn
* **AI Framework:** TensorFlow (CNN Architecture)
* **Signal Processing:** Librosa, Pydub, FFmpeg
* **Data Handling:** NumPy, Pandas

## 📁 Key Files
- `app.py`: Main logic for audio processing and model inference.
- `MLPmusicgen.h5`: Saved CNN weights for the 9-genre classifier.
- `Dockerfile`: Configuration for the 16GB RAM container environment.

---
*Developed by Vishwa Vidhatha Gujjula | Data Engineer*