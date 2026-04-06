---
title: Plant Disease Classification & Remedy
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
license: mit
---

# Plant Disease Classification & Remedy App

Upload an image of a plant leaf to identify potential diseases and get AI-powered treatment recommendations.

## Features

- **Disease Detection**: Uses deep learning to classify plant diseases from images
- **AI-Powered Remedies**: Get detailed treatment recommendations using Google Gemini AI
- **User-Friendly Interface**: Simple upload and instant results

## How to Use

1. Upload a clear image of a plant leaf showing symptoms
2. Wait for the AI to analyze the image
3. Get disease classification and remedy suggestions

## Supported Plants

- Pepper (Bell)
- Tomato
- Potato
- Corn
- Apple
- And many more...

## Technology Stack

- **Frontend**: Gradio
- **ML Framework**: PyTorch Lightning
- **AI Model**: Google Gemini 2.0
- **Computer Vision**: ResNet-based classifier

## Model Performance

- Classification accuracy: ~85%
- Supports 39 different plant disease classes
- Out-of-distribution detection for unknown diseases

## Privacy

Images are processed locally and not stored permanently. AI analysis uses secure API calls.