from PIL import Image

import torch

import torchvision.transforms.functional as F

import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, OOD_MODEL, ServiceConfig, get_device
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


def classify_disease(image_tensor):
    if CLF_MODEL is None:
        raise RuntimeError("Classification model failed to load. Check server logs.")

    with torch.no_grad():

        model_to_run = CLF_MODEL.model if hasattr(CLF_MODEL, "model") else CLF_MODEL
        outputs = model_to_run(image_tensor)

        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, 5, dim=1)

        print("=== TOP 5 PREDICTIONS ===", flush=True)
        for i in range(5):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            print(f"{i+1}. {ServiceConfig.ID2LABEL[idx]} -> {prob:.4f}", flush=True)

        pred_idx = top_indices[0][0].item()
        pred_prob = top_probs[0][0].item()

        label = ServiceConfig.ID2LABEL[pred_idx]

        return label, pred_prob



def normalize_label(label: str) -> str:

    if not isinstance(label, str):
        label = str(label)
    label = label.replace("__", " ").strip()
    label = " ".join(label.split())
    return label.title()


def clean_remedy_text(remedy: str, disease_name: str) -> str:
    if not isinstance(remedy, str):
        remedy = str(remedy)

    lines = [line.strip() for line in remedy.splitlines() if line.strip()]
    cleaned_lines = []

    for line in lines:
        lower_line = line.lower()
        # Remove heading lines and explicit disease labels
        if lower_line.startswith("###"):
            continue
        if lower_line.startswith("disease:") or lower_line.startswith("**disease name:"):
            continue
        if lower_line.startswith("### remedy for"):
            continue
        if lower_line.startswith("remedy for"):
            continue

        # Normalize leading markdown before checking the disease name
        stripped_line = line.lstrip('* ').strip()
        stripped_lower = stripped_line.lower()
        if stripped_lower.startswith(f"{disease_name.lower()}"):
            stripped_line = stripped_line[len(disease_name):].strip(" :-*")
            if not stripped_line:
                continue
            line = stripped_line

        line = line.replace("**", "")
        cleaned_lines.append(line)

    if not cleaned_lines:
        return remedy.strip()

    return "\n".join(cleaned_lines).strip()


def detect_out_of_distribution(image_tensor: torch.Tensor):
    if OOD_MODEL is None:
        return False, None

    OOD_MODEL.eval()
    with torch.no_grad():
        reconstructed = OOD_MODEL(image_tensor)
        score = torch.mean((reconstructed - image_tensor) ** 2).item()

    print(f"[OOD] score: {score:.6f}", flush=True)
    return score > ServiceConfig.OOD_THRESHOLD, score


def workflow(image: Image.Image):
    try:
        image_tensor = transform_for_prediction(image).unsqueeze(0)

        # 🔥 Step 1: classifier prediction
        classifier_label, confidence = classify_disease(image_tensor)
        print(f"[INFO] classifier confidence: {confidence:.4f}", flush=True)

        classifier_label = normalize_label(classifier_label)

        # 🚨 LOW CONFIDENCE CASE
        if confidence < 0.70:
            return (
                "Uncertain",
                f"Model confidence is low ({confidence:.2f}). Please upload a clearer leaf image with plain background."
            )

        # ✅ HEALTHY CASE
        if 'Healthy' in classifier_label:
            return (
                classifier_label,
                "The leaf appears healthy. No treatment is needed."
            )

        # 🤖 Gemini (optional)
        try:

            remedy = llm_strategy(
                ServiceConfig.LLM_MODEL_KEY,
                classifier_label,
                return_both=False
            )

            if not isinstance(remedy, str):
                raise ValueError("Invalid Gemini response")

            remedy = clean_remedy_text(remedy.strip(), classifier_label)

            if not remedy:
                raise ValueError("Empty remedy")

        except Exception as e:
            print("[ERROR] LLM failed:", e, flush=True)
            traceback.print_exc()
            
            remedy = "Consult an agricultural expert for proper treatment."

        return classifier_label, remedy

    except Exception as e:
        print("[ERROR] Workflow failed:", e, flush=True)
        traceback.print_exc()
        return "Error", f"An error occurred: {str(e)}"