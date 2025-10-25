AI-Powered Music Genre Classification (CNN)
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify 30-second audio segments into 10 distinct music genres (e.g., Rock, Classical, Disco, Jazz). The core of the solution involves advanced audio signal processing using the Librosa library to extract crucial features necessary for training the deep learning model.

This project was developed as part of the Elevate Labs Internship program (Project 11).

🚀 Project Overview
The objective is to accurately predict the genre of any given music clip. Instead of processing raw audio waveforms, the model is trained on Mel-Frequency Cepstral Coefficients (MFCCs), which are highly effective feature representations in audio processing, treating the audio as a 2D image for the CNN.

Key Features:

Feature Extraction Pipeline: Converts raw .wav or .au files into MFCC matrices.

CNN Architecture: Implements a deep learning model for 10-class classification.

Standalone Prediction: A script (predict.py) that loads the trained model to classify new, unseen audio files.

🛠️ Technology Stack
Component

Technology

Purpose

Language

Python 3.x

Core development language.

Audio Processing

Librosa

MFCC feature extraction and audio loading/resampling.

Deep Learning

TensorFlow / Keras

Building, compiling, and training the CNN model.

Numerical Ops

NumPy

Efficient handling of numerical array data.

Data Handling

JSON, Scikit-learn

Storing extracted features and splitting data.

⚙️ Setup and Installation
1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/AI-Powered-Music-Genre-Classifier.git](https://github.com/Madhusudan2005/AI-Powered-Music-Genre-Classifier.git)
cd AI-Powered-Music-Genre-Classifier

2. Environment Setup
It is highly recommended to use a Python virtual environment.

# Create and activate environment (for Linux/macOS)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

3. Data Acquisition (GTZAN Dataset)
The raw audio files for this project are external and must be downloaded separately (approximately 1 GB).

Download: Search for the GTZAN Dataset (e.g., on Kaggle).

Extraction: Create a folder named data/ in the project root. Extract the downloaded genres_original folder so that your structure looks like this:

.
├── genre_classifier.py
├── data/
│   └── genres_original/  <-- This folder contains 10 genre subfolders
│       ├── blues/
│       ├── classical/
│       └── ...
└── test_audio/

🚀 Usage Instructions
The project runs in two main phases: Training and Prediction.

Phase 1: Feature Extraction and Model Training
This step reads the audio, extracts features, and trains the CNN model.

Run the Classifier: Execute the main script. The script is configured to first check if data_mfcc.json exists. If not, it runs the extraction and then proceeds to train the model.

python genre_classifier.py

Output: This process will take several minutes (up to 15-20 min depending on your CPU). When complete, two files will be saved in the root directory:

data_mfcc.json (The extracted MFCC features).

genre_cnn_model.h5 (The trained Keras model).

Phase 2: Prediction on New Audio
After training, you can test the saved model on any new audio clip.

Place Test File: Add a 30-second audio file (e.g., .wav or .mp3) into the test_audio/ folder.

Update Script: Open predict.py and update the TEST_AUDIO_FILE variable path to point to your new audio file (e.g., "test_audio/my_new_song.wav").

Run Prediction:

python predict.py

The script will output the predicted genre and the model's confidence score.

Developed by G.Madhusudan
