from PIL import Image
import torch
import torchvision.transforms.functional as F
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, OOD_MODEL, ServiceConfig, get_device


def transform_for_prediction(img: Image.Image):
    img = img.convert("RGB")
    z = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
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

        print(f"FINAL PREDICTION: {label} ({pred_prob:.4f})", flush=True)
        return label, pred_prob


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        label = str(label)
    label = label.replace("__", " ").strip()
    label = " ".join(label.split())
    return label.title()


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

        classifier_label, confidence = classify_disease(image_tensor)
        classifier_label = normalize_label(classifier_label)

        print(f"[INFO] classifier confidence: {confidence:.4f}", flush=True)

        if confidence < 0.60:
            return (
                "Uncertain",
                f"Model confidence is low ({confidence:.2f}). Please upload a clearer single-leaf image with plain background."
            )

        if "Healthy" in classifier_label and confidence >= 0.60:
            return (
                classifier_label,
                "The leaf appears healthy. No treatment is needed."
            )

        return (
            classifier_label,
            f"Detected disease: {classifier_label}. Remedy generation is temporarily disabled."
        )

    except Exception as e:
        print("[ERROR] Workflow failed:", e, flush=True)
        traceback.print_exc()
        return "Error", f"An error occurred: {str(e)}"