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

## Run Locally Without Docker

If Docker Desktop cannot start because virtualization is not available, you can still run the app with Python:

1. Open PowerShell in the project root.
2. Run `run-local.bat`.
3. Open `http://localhost:7860` in your browser.

## Deploy to Render

This project already includes `render.yaml` and `Dockerfile`. To deploy on Render:

1. Push your repository to GitHub.
2. Create a Docker web service on Render.
3. Set the environment variables `GEMINI_API_KEY` or `GOOGLE_API_KEY` in Render.
4. Use the existing `render.yaml` to build the service.

The app listens on port `7860`, and `app.py` is configured for production with `share=False`.

## Deploy to Railway

This project is also ready for Railway deployment using `railway.json`.

1. Push your repository to GitHub.
2. Create a new Railway project and connect your GitHub repo.
3. Railway will detect Python and use `railway.json` for build/start configuration.
4. Add the environment secrets to Railway:
   - `GEMINI_API_KEY`
   - `GOOGLE_API_KEY` (optional alternate)
5. Deploy the service.

Railway will use the existing `python app.py` start command and expose the app on the assigned port.
