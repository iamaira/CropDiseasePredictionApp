import os
from PIL import Image
import torch
from acfg.modelconfig import ModelConfig
import torchvision.transforms.functional as F


from acfg.appconfig import CLF_MODEL, ServiceConfig, get_device
from service.external import llm_strategy


def transform_for_prediction(img: Image.Image):
    """Transforms a PIL image for model prediction.

    This function applies a series of transformations to prepare an image for model inference:
    1. Resizes the image to the model's expected input size
    2. Converts the image to a tensor
    3. Normalizes the tensor using preconfigured mean and std values

    Args:
        img (PIL.Image): Input image to transform

    Returns:
        torch.Tensor: Transformed image tensor ready for model inference
    """
    z = img
    z = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
    z = F.to_tensor(z)
    z = F.normalize(z, mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
    return z.to(get_device()[1])


def classify_disease(image):
    if CLF_MODEL is None:
        raise RuntimeError("Classification model failed to load. Check server logs.")
    
    image_tensor = transform_for_prediction(image).unsqueeze(0)

    with torch.no_grad():
        outputs = CLF_MODEL(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    return ServiceConfig.ID2LABEL[prediction]


import io
import traceback

def normalize_label(label: str) -> str:
    """Normalize model labels into readable text."""
    if not isinstance(label, str):
        label = str(label)
    label = label.replace("__", " ").strip()
    label = " ".join(label.split())
    return label.title()


def parse_gemini_response(response_text: str) -> tuple[str, str]:
    """Extract disease name and remedy from Gemini response."""
    if not isinstance(response_text, str):
        response_text = str(response_text)

    lines = [line.strip() for line in response_text.split('\n') if line.strip()]

    disease_name = "Plant Disease"
    for line in lines:
        if line.startswith('###'):
            disease_name = line.replace('###', '').strip()
            break
        if ':' in line:
            key, value = line.split(':', 1)
            if key.lower() in {'disease', 'diagnosis', 'identified disease', 'possible disease'}:
                disease_name = value.strip()
                break

    if disease_name == 'Plant Disease' and lines:
        first_line = lines[0]
        if isinstance(first_line, str) and len(first_line) < 60 and ' ' in first_line and not first_line.endswith(':'):
            disease_name = first_line

    disease_name = normalize_label(disease_name)
    if 'Healthy' in disease_name:
        disease_name = 'Plant is Healthy'

    remedy = response_text.strip()
    return disease_name, remedy


def workflow(image: Image.Image):
    try:
        classifier_label = classify_disease(image)
        if not isinstance(classifier_label, str):
            classifier_label = str(classifier_label)
        classifier_label = normalize_label(classifier_label)
        if 'Healthy' in classifier_label:
            classifier_label = 'Plant is Healthy'

        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()

        try:
            disease_and_remedy = llm_strategy(
                ServiceConfig.LLM_MODEL_KEY,
                '',
                image_file=image_bytes,
                return_both=True
            )
            if not isinstance(disease_and_remedy, str):
                raise ValueError(f"LLM returned non-string response: {type(disease_and_remedy).__name__}")

            llm_disease_name, llm_remedy = parse_gemini_response(disease_and_remedy)

            if llm_disease_name and llm_disease_name not in {'Plant Disease', 'Plant is Healthy'}:
                disease_name = llm_disease_name
            else:
                disease_name = classifier_label

            remedy = llm_remedy
            if not remedy:
                raise ValueError('Gemini returned empty remedy')
        except Exception as e:
            print(f"[ERROR] LLM strategy failed: {e}")
            traceback.print_exc()
            disease_name = classifier_label
            remedy = 'Please consult with a local agricultural extension office for specific diagnosis and treatment recommendations.'

        return disease_name, remedy
    except Exception as e:
        print(f"[ERROR] Workflow failed: {e}")
        traceback.print_exc()
        return 'Error', f'An error occurred while processing the image: {str(e)}'
