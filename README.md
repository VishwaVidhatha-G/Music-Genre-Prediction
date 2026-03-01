# 🎧 Music Genre Prediction (CNN)

A Flask application that classifies audio files into 9 distinct genres using a Convolutional Neural Network. The project focus was on migrating to a high-RAM infrastructure to ensure stable model inference and audio processing.

## 🔗 Live Application
[View the App on Hugging Face Spaces](https://huggingface.co/spaces/VishwaVidhatha/Music-Genre-Prediction)

## 🛠️ Infrastructure & Execution
This project was moved to Hugging Face Spaces to solve specific deployment failures encountered on smaller servers:
* **Memory Headroom:** Uses 16GB RAM to prevent "Internal Server Errors" during TensorFlow model loading.
* **Gunicorn Configuration:** Set with a 120-second timeout to allow the CPU enough time for deep learning inference.
* **Audio Handling:** Built with a Dockerized FFmpeg environment to process .mp3 and .wav formats without local software dependencies.

## 📊 Logic Flow
1. **Audio Extraction:** The system cuts a 10-second segment from the uploaded file.
2. **Internal Transformation:** The audio is converted into a Mel Spectrogram image in the background.
3. **CNN Classification:** The model analyzes the frequency patterns within that image to determine the genre.
4. **User Output:** The app displays the final predicted genre (e.g., Rock, Hiphop, Metal).



## 📋 Technical Setup
* **AI Framework:** TensorFlow 2.12 (CPU optimized).
* **Signal Processing:** Librosa and Pydub.
* **Server:** Flask on Gunicorn.

## 📁 Key Files
- `app.py`: Backend routing and signal processing logic.
- `MLPmusicgen.h5`: Saved CNN weights for classification.
- `Dockerfile`: Container environment for the 16GB RAM deployment.