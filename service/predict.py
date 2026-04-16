from PIL import Image
import torch
import torchvision.transforms.functional as F
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, ServiceConfig, get_device


REMEDY_DB = {
    "Apple Scab": """### Apple Scab
**Management Strategies:**
- Remove infected leaves and fallen debris.
- Apply fungicide at early leaf stage.
- Maintain good airflow around the plant.""",

    "Apple Black Rot": """### Apple Black Rot
**Management Strategies:**
- Remove infected fruits, leaves, and branches.
- Prune dead wood and sanitize tools.
- Use recommended fungicide if needed.""",

    "Apple Cedar Rust": """### Apple Cedar Rust
**Management Strategies:**
- Remove nearby cedar or juniper hosts if possible.
- Prune affected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant.
- Remove fallen infected leaves and destroy them.
- Monitor new leaves regularly for early symptoms.""",

    "Cherry Powdery Mildew": """### Cherry Powdery Mildew
**Management Strategies:**
- Remove infected leaves.
- Improve airflow and avoid overhead watering.
- Use sulfur or suitable fungicide if needed.""",

    "Corn Cercospora Leaf Spot Gray Leaf Spot": """### Corn Cercospora Leaf Spot Gray Leaf Spot
**Management Strategies:**
- Remove infected leaves if possible.
- Avoid prolonged leaf wetness.
- Use resistant variety and fungicide if severe.""",

    "Corn Common Rust": """### Corn Common Rust
**Management Strategies:**
- Monitor the rust pustules regularly.
- Use resistant varieties if available.
- Apply fungicide if infection spreads rapidly.""",

    "Corn Northern Leaf Blight": """### Corn Northern Leaf Blight
**Management Strategies:**
- Remove infected crop debris.
- Use resistant seeds.
- Apply fungicide if disease is severe.""",

    "Grape Black Rot": """### Grape Black Rot
**Management Strategies:**
- Remove infected leaves and fruits.
- Prune the canopy for airflow.
- Use preventive fungicide if needed.""",

    "Grape Esca Black Measles": """### Grape Esca Black Measles
**Management Strategies:**
- Prune infected parts carefully.
- Avoid plant stress.
- Consult an agricultural expert for vineyard-level management.""",

    "Grape Leaf Blight Isariopsis Leaf Spot": """### Grape Leaf Blight Isariopsis Leaf Spot
**Management Strategies:**
- Remove infected leaves.
- Avoid overhead watering.
- Use proper fungicide if needed.""",

    "Orange Haunglongbing Citrus Greening": """### Orange Haunglongbing Citrus Greening
**Management Strategies:**
- Remove severely infected trees if confirmed.
- Control psyllid insect vectors.
- Consult local agricultural authority immediately.""",

    "Peach Bacterial Spot": """### Peach Bacterial Spot
**Management Strategies:**
- Remove infected leaves and twigs.
- Avoid overhead watering.
- Use copper-based spray if recommended locally.""",

    "Pepper Bell Bacterial Spot": """### Pepper Bell Bacterial Spot
**Management Strategies:**
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray if recommended.
- Maintain field sanitation.
- Improve airflow around the crop.""",

    "Potato Early Blight": """### Potato Early Blight
**Management Strategies:**
- Remove affected leaves.
- Apply fungicide if spread increases.
- Rotate crops and avoid repeated planting in same soil.""",

    "Potato Late Blight": """### Potato Late Blight
**Management Strategies:**
- Remove infected leaves immediately.
- Avoid leaf wetness.
- Use late blight fungicide if needed.""",

    "Squash Powdery Mildew": """### Squash Powdery Mildew
**Management Strategies:**
- Remove infected leaves.
- Improve air circulation.
- Use sulfur or suitable fungicide if needed.""",

    "Strawberry Leaf Scorch": """### Strawberry Leaf Scorch
**Management Strategies:**
- Remove infected leaves.
- Avoid overhead watering.
- Use fungicide if disease spreads significantly.""",

    "Tomato Bacterial Spot": """### Tomato Bacterial Spot
**Management Strategies:**
- Remove infected leaves.
- Avoid splashing water on leaves.
- Use copper-based bactericide if needed.
- Keep proper spacing between plants.
- Remove plant debris from the soil surface.""",

    "Tomato Early Blight": """### Tomato Early Blight
**Management Strategies:**
- Remove lower infected leaves.
- Use fungicide if spread continues.
- Rotate crops and mulch the soil.""",

    "Tomato Late Blight": """### Tomato Late Blight
**Management Strategies:**
- Remove infected leaves quickly.
- Do not overhead water.
- Use late blight fungicide if needed.""",

    "Tomato Leaf Mold": """### Tomato Leaf Mold
**Management Strategies:**
- Improve ventilation around plants.
- Avoid high humidity.
- Use fungicide if disease spreads.""",

    "Tomato Septoria Leaf Spot": """### Tomato Septoria Leaf Spot
**Management Strategies:**
- Remove infected lower leaves.
- Avoid splashing soil onto leaves.
- Use fungicide if infection progresses.""",

    "Tomato Spider Mites Two Spotted Spider Mite": """### Tomato Spider Mites Two Spotted Spider Mite
**Management Strategies:**
- Spray water under leaves to reduce mites.
- Use neem oil or miticide if needed.
- Remove heavily infested leaves.""",

    "Tomato Target Spot": """### Tomato Target Spot
**Management Strategies:**
- Remove infected leaves.
- Improve airflow and reduce humidity.
- Use appropriate fungicide if needed.""",

    "Tomato Yellow Leaf Curl Virus": """### Tomato Yellow Leaf Curl Virus
**Management Strategies:**
- Remove severely infected plants.
- Control whiteflies.
- Use resistant varieties if possible.""",

    "Tomato Mosaic Virus": """### Tomato Mosaic Virus
**Management Strategies:**
- Remove infected plants.
- Sanitize hands and tools.
- Avoid tobacco contact near plants.""",

    "Plant is Healthy": """No treatment is needed.""",
}


def transform_for_prediction(img: Image.Image):
    img = img.convert("RGB")
    z = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
    z = F.to_tensor(z)
    z = F.normalize(z, mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD)
    return z.to(get_device()[1])


def classify_disease(image_tensor):
    if CLF_MODEL is None:
        raise RuntimeError("Classification model failed to load. Check server logs.")

    with torch.no_grad():
        outputs = CLF_MODEL(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

        prediction = top_idx.item()
        confidence = top_prob.item()

        print("=== TOP PREDICTION ===", flush=True)
        print(
            f"{prediction} -> {ServiceConfig.ID2LABEL[prediction]} ({confidence:.4f})",
            flush=True,
        )

    return ServiceConfig.ID2LABEL[prediction], confidence


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        label = str(label)
    label = label.replace("__", " ").strip()
    label = " ".join(label.split())
    return label.title()


def get_offline_remedy(label: str) -> str:
    return REMEDY_DB.get(
        label,
        "General plant care: remove infected leaves, avoid overwatering, improve airflow, and consult an agricultural expert if symptoms continue."
    )


def workflow(image: Image.Image):
    try:
        image_tensor = transform_for_prediction(image).unsqueeze(0)

        classifier_label, confidence = classify_disease(image_tensor)
        classifier_label = normalize_label(classifier_label)

        print(f"[INFO] classifier label: {classifier_label}", flush=True)
        print(f"[INFO] classifier confidence: {confidence:.4f}", flush=True)

        # Healthy only when model itself predicts healthy
        if "healthy" in classifier_label.lower():
            return (
                "Plant is Healthy",
                "No treatment is needed."
            )

        # Only very low confidence becomes uncertain
        if confidence < 0.35:
            return (
                "Uncertain",
                "Model is not confident. Please try another image."
            )

        # Disease / bacterial case
        remedy = get_offline_remedy(classifier_label)
        return classifier_label, remedy

    except Exception as e:
        print("[ERROR] Workflow failed:", e, flush=True)
        traceback.print_exc()
        return "Error", f"An error occurred: {str(e)}"