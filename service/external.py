import os
import time
import traceback
import warnings
from dotenv import load_dotenv
import google.generativeai as genai

# Suppress non-critical FutureWarnings for presentation
warnings.filterwarnings("ignore", category=FutureWarning)

# Load API key and environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")

# Configure GenAI
genai.configure(api_key=api_key)

SYS_INSTR = (
    "You are a plant disease expert. You will be given queries regarding plant diseases. "
    "Always respond in simple English, avoid technical jargon, and provide clear remedies. "
    "If a scientific name is needed, include it only in parentheses after the common name."
)
TXT_PROMPT = (
    "Suggest remedy for the disease in easy-to-understand bullet points. "
    "Use simple language and avoid hard-to-pronounce scientific terms. "
    "If the plant is healthy, say 'Plant is Healthy' and provide maintenance tips."
)
IMG_TXT_PROMPT = (
    "Based on the given image, identify the likely disease or health status of the plant "
    "and suggest the remedy in simple language."
)

MODEL_NAMES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
]

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds

# Fallback remedies for common plant diseases
FALLBACK_REMEDIES = {
    "pepper__bell___bacterial_spot": """
### Remedy for Bacterial Spot (Bell Pepper)

**Management Strategies:**
- Remove and destroy infected leaves to reduce bacterial spread
- Apply copper-based fungicides weekly during growing season
- Ensure proper plant spacing for good air circulation
- Avoid overhead watering; use drip irrigation instead
- Sanitize pruning tools between cuts
- Rotate crops with non-solanaceous plants
- Use disease-resistant pepper varieties when available
""",
    "pepper__bell___healthy": """
### Plant is Healthy! ✓

**Maintenance Tips:**
- Continue regular watering (1-2 inches per week)
- Monitor for any signs of disease
- Provide adequate sunlight (6-8 hours daily)
- Fertilize every 2-3 weeks during growing season
- Maintain good air circulation around plants
""",
}

def llm_strategy(llm_name, disease_name, image_file=None, return_both=False):
    if llm_name.lower() == "gemini":
        if return_both and image_file:
            return identify_disease_and_remedy_from_image(image_file)
        else:
            return get_response_from_gemini(disease_name, image_file)
    else:
        raise ValueError(f"❌ Unsupported LLM: {llm_name}")

def identify_disease_and_remedy_from_image(image_file) -> str:
    """
    Identify plant disease from image and return remedy in one call using Gemini.
    
    Args:
        image_file: Image bytes (JPEG)
    
    Returns:
        String with disease name and remedy in markdown format
    """
    import base64
    
    # Convert image bytes to base64
    image_b64 = base64.standard_b64encode(image_file).decode("utf-8")
    
    prompt = f"""{SYS_INSTR}

Please analyze this plant image and:
1. Identify the disease or health status
2. Provide 3-5 key management strategies

Format your response as:
### [Disease Name or Status]

[Brief description if needed]

**Management Strategies:**
- [Strategy 1]
- [Strategy 2]
- etc.

IMPORTANT: start with "### " followed immediately by the disease name or "Plant is Healthy" if the plant appears healthy.
Use simple language and avoid hard-to-pronounce scientific terms."""

    # Try each Gemini model with the image
    for model_name in MODEL_NAMES:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[INFO] Calling {model_name} with image (attempt {attempt+1}/{MAX_RETRIES})...")
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": image_b64},
                    prompt
                ])

                if response and hasattr(response, "text") and response.text:
                    print(f"[SUCCESS] {model_name} identified disease from image!")
                    return response.text.strip()
                else:
                    print(f"[WARN] {model_name} returned empty response")
                    break

            except Exception as e:
                print(f"[ERROR] {model_name} failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                
                if attempt == MAX_RETRIES - 1:
                    print(f"[INFO] {model_name} exhausted. Trying next model...")
                    break
                
                wait_time = BASE_DELAY * (2 ** attempt)
                time.sleep(wait_time)

    # Fallback
    print("[INFO] All Gemini models failed. Using generic fallback...")
    return """### Plant Disease Detected

**General Management:**
- Remove infected plant parts and dispose of them properly
- Apply appropriate fungicide or bactericide based on the disease type
- Improve air circulation by proper spacing and pruning
- Avoid overhead watering to reduce moisture on leaves
- Rotate crops annually
- Use disease-resistant varieties when available

**Note:** For precise diagnosis, please consult with a local agricultural extension office or a plant pathologist.
"""

def get_response_from_gemini(disease_name, image_file=None) -> str:
    """
    Get remedies from Gemini model for a plant disease.
    Falls back to hardcoded remedies if Gemini is unavailable.
    
    Args:
        disease_name: Name of the plant disease
        image_file: Optional image bytes (currently text-only for compatibility)
    
    Returns:
        Remedy text
    """
    prompt = f"""{SYS_INSTR}

Disease: {disease_name}

{TXT_PROMPT}

Use a clear disease name and easy remedies. Do not say you cannot pronounce terms."""

    # Try Gemini first
    for model_name in MODEL_NAMES:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[INFO] Calling {model_name} (attempt {attempt+1}/{MAX_RETRIES})...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)

                if response and hasattr(response, "text") and response.text:
                    print(f"[SUCCESS] {model_name} succeeded!")
                    return response.text.strip()
                else:
                    print(f"[WARN] {model_name} returned empty response")
                    break

            except Exception as e:
                err_str = str(e).lower()
                print(f"[ERROR] {model_name} failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                
                # If this is the last attempt for this model, try the next one
                if attempt == MAX_RETRIES - 1:
                    print(f"[INFO] {model_name} exhausted. Trying next model...")
                    break
                
                wait_time = BASE_DELAY * (2 ** attempt)
                time.sleep(wait_time)

    # Fallback: use hardcoded remedies
    print("[INFO] All Gemini models failed. Using fallback remedies...")
    disease_key = disease_name.lower().replace(" ", "_").replace("__bell__", "__bell___")
    
    if disease_key in FALLBACK_REMEDIES:
        return FALLBACK_REMEDIES[disease_key]
    
    # Generic fallback if disease not in list
    return f"""
### Remedy for {disease_name}

**General Management:**
- Remove infected plant parts and dispose of them properly
- Apply appropriate fungicide or bactericide based on the disease type
- Improve air circulation by proper spacing and pruning
- Avoid overhead watering to reduce moisture on leaves
- Rotate crops annually
- Use disease-resistant varieties when available
- Monitor plants regularly for early detection of disease symptoms

**Note:** For specific treatment, consult with a local agricultural extension office.
"""