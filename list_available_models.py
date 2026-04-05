#!/usr/bin/env python3
"""List available Gemini models from the API."""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")
    exit(1)

# Configure
genai.configure(api_key=api_key)

# List models
print("📋 Available Gemini Models:")
print("=" * 80)
for model in genai.list_models():
    print(f"\nModel: {model.name}")
    print(f"  Display Name: {model.display_name}")
    print(f"  Description: {model.description}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported Methods: {model.supported_generation_methods}")
