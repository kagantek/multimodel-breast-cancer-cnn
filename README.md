# Multi-Model Breast Cancer CNN

A Flask web application for breast cancer classification using deep learning models trained on three medical imaging modalities.

## Overview

This project deploys multiple convolutional neural network models for breast cancer detection. Users can upload medical images and receive predictions from different CNN architectures.

### Supported Modalities
- Mammography
- Ultrasound
- Histopathology

### Model Architectures
- ResNet50
- VGG16
- DenseNet121

Each architecture has two training variants: Initial Unfrozen and Fine-tuned.

## CNN Models

Pre-trained model weights are hosted on Hugging Face:

https://huggingface.co/ktek/thesis-cnn-models

Download the models and place them in the `model/` directory following this structure:

```
model/
├── mammography/
│   ├── initial_unfrozen/
│   └── fine_tune/
├── ultrasound/
│   ├── initial_unfrozen/
│   └── fine_tune/
└── histopathology/
    ├── initial_unfrozen/
    └── fine_tune/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The application runs on `http://localhost:3000`

## Features

- Single model predictions with confidence scores
- Side-by-side comparison of all three architectures
- Model performance visualizations (confusion matrix, ROC curve, learning curves)
- Classification metrics display

## Project Structure

```
├── app.py                 # Application entry point
├── config.py              # Model configuration and thresholds
├── src/
│   ├── routes/            # Flask blueprints
│   ├── services/          # Model loading and prediction
│   └── utils/             # Image processing
├── templates/             # HTML templates
├── static/                # CSS, JS, images
└── model/                 # CNN model weights (.h5)
```

## Tech Stack

- Python
- Flask
- TensorFlow/Keras
- Bootstrap 4

## Author

Kağan Tek - CSE492 Graduation Project
