import os
from PIL import Image
import torch

import torchvision.transforms.functional as F
import io
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, ServiceConfig, get_device
from service.external import llm_strategy


def transform_for_prediction(img: Image.Image):
    img = img.convert("RGB")
    z = F.resize(img, [224, 224])
    z = F.to_tensor(z)
    z = F.normalize(
        z,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return z.to(get_device()[1])


def classify_disease(image):
    if CLF_MODEL is None:
        raise RuntimeError("Classification model failed to load. Check server logs.")

    image_tensor = transform_for_prediction(image).unsqueeze(0)

    with torch.no_grad():
        # Lightning wrapper ke andar actual backbone call
        outputs = CLF_MODEL.model(image_tensor)

        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, 5, dim=1)

        print("Top 5 predictions:")
        for i in range(5):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            print(f"{i+1}. {idx} -> {ServiceConfig.ID2LABEL[idx]} ({prob:.4f})")

        prediction = top_indices[0][0].item()

    return ServiceConfig.ID2LABEL[prediction]




def normalize_label(label: str) -> str:

    if not isinstance(label, str):
        label = str(label)
    label = label.replace("__", " ").strip()
    label = " ".join(label.split())
    return label.title()


def parse_gemini_response(response_text: str) -> tuple[str, str]:


    if not isinstance(response_text, str):
        response_text = str(response_text)

    lines = [line.strip() for line in response_text.split('\n') if line.strip()]

    disease_name = "Plant Disease"
    for line in lines:
        if line.startswith('###'):
            disease_name = line.replace('###', '').strip()
            break

    
    disease_name = normalize_label(disease_name)


    remedy = response_text.strip()
    return disease_name, remedy


def workflow(image: Image.Image):
    try:
        # ✅ Step 1: classifier prediction
        classifier_label = classify_disease(image)

        classifier_label = normalize_label(classifier_label)

        if 'Healthy' in classifier_label:
            classifier_label = 'Plant is Healthy'

        # ✅ Convert image to bytes
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()

        try:
            # ✅ Gemini call with strong prompt
            disease_and_remedy = llm_strategy(
                ServiceConfig.LLM_MODEL_KEY,
                f"""
You are a plant disease expert.

The detected disease is: {classifier_label}

Give:
1. Exact treatment for this disease
2. Chemicals or fungicides if needed
3. Prevention tips

DO NOT rename the disease.
DO NOT say "Plant Disease Detected".
Give answer in clean bullet points.
""",
                image_file=image_bytes,
                return_both=True
            )

            if not isinstance(disease_and_remedy, str):
                raise ValueError("Invalid Gemini response")

            _, llm_remedy = parse_gemini_response(disease_and_remedy)

            disease_name = classifier_label

            remedy = llm_remedy

            if not remedy:
                raise ValueError("Empty remedy")

        except Exception as e:
            print("[ERROR] LLM failed:", e)
            traceback.print_exc()
            disease_name = classifier_label
            remedy = "Consult an agricultural expert for proper treatment."

        return disease_name, remedy

    except Exception as e:
        print("[ERROR] Workflow failed:", e)
        traceback.print_exc()
        return "Error", f"An error occurred: {str(e)}"