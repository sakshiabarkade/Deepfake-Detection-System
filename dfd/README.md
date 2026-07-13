# TrueVision - DeepFake Detection

A DeepFake Detection Web Application using Deep Learning (ResNext and LSTM), Flask, and React that predicts whether a video is FAKE or REAL along with a confidence ratio.

## System Requirements

- Python 3.10.x (Required for compatibility with provided wheel files)
- pip (Python package installer)
- Windows OS (for provided wheel files)
- Visual C++ Build Tools (for Windows)

## Project Overview

This project implements a DeepFake Detection system using:
- Deep Learning techniques (ResNext and LSTM)
- Flask Backend
- React Frontend
- Face Recognition for processing

The system analyzes uploaded videos and determines if they are authentic or deepfake manipulated, providing a confidence score for the prediction.

## Quick Start Guide

1. **Environment Setup**
   - Install Python 3.10.x
   - Install Visual C++ Build Tools (Windows)
   - Clone this repository

2. **Install Dependencies**
   First, install the provided wheel files:
   ```bash
   pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
   pip install face_recognition-1.3.0-py2.py3-none-any.whl
   ```
   Then install other requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **Project Structure Setup**
   - Create "Uploaded_Files" folder in DeepFake_Detection directory
   - Create "model" folder in DeepFake_Detection directory

4. **Run the Application**
   ```bash
   cd DeepFake_Detection
   python server.py
   ```
   Access the application at http://localhost:5000

## Project Structure
```
DeepFake-Detection/
├── DeepFake_Detection/       # Main application directory
│   ├── model/               # Model directory (add df_model.pt here)
│   ├── static/              # Static files (CSS, JS, images)
│   ├── templates/           # HTML templates
│   ├── Uploaded_Files/      # Directory for uploaded videos
│   ├── requirements.txt     # Python dependencies
│   └── server.py           # Flask server
├── Project-Setup.txt        # Setup instructions
└── README.md               # This file
```

## Model Information

- TBA

## Performance Metrics

- Model Accuracy: 96.97%
- Detailed metrics and graphs available in the repository
- Model Deployment : Hugging Face, Currently Private

## Troubleshooting

1. **Wheel File Installation Issues**
   - Ensure Python 3.10.x is installed
   - Install Visual C++ Build Tools
   - Use the provided .whl files instead of pip installing packages directly (optional)

2. **Model Loading Issues**
   - Verify df_model.pt is in the correct location
   - Check Python and PyTorch versions compatibility

3. **Video Processing Issues**
   - Ensure proper video format (preferably MP4, )
   - Check if Uploaded_Files directory exists
   - Verify sufficient system memory

## Acknowledgments

- Celeb-DF dataset
- DeepFake++ detection implementation
