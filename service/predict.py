from PIL import Image
import torch
import torchvision.transforms.functional as F
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, get_device, ServiceConfig


REMEDY_DB = {
    "Apple Scab": """### Apple Scab
- Remove infected leaves and fallen debris.
- Apply fungicide at early leaf stage.
- Maintain good airflow around the plant.""",

    "Apple Black Rot": """### Apple Black Rot
- Remove infected fruits, leaves, and branches.
- Prune dead wood and sanitize tools.
- Use recommended fungicide if needed.""",

    "Apple Cedar Rust": """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts if possible.
- Prune infected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant.""",

    "Cherry Powdery Mildew": """### Cherry Powdery Mildew
- Remove infected leaves.
- Improve airflow and avoid overhead watering.
- Use sulfur or suitable fungicide if needed.""",

    "Corn Cercospora Leaf Spot Gray Leaf Spot": """### Corn Cercospora Leaf Spot Gray Leaf Spot
- Remove infected leaves if possible.
- Avoid prolonged leaf wetness.
- Use resistant variety and fungicide if severe.""",

    "Corn Common Rust": """### Corn Common Rust
- Monitor rust pustules regularly.
- Use resistant varieties if available.
- Apply fungicide if infection spreads rapidly.""",

    "Corn Northern Leaf Blight": """### Corn Northern Leaf Blight
- Remove infected crop debris.
- Use resistant seeds.
- Apply fungicide if disease is severe.""",

    "Grape Black Rot": """### Grape Black Rot
- Remove infected leaves and fruits.
- Prune the canopy for airflow.
- Use preventive fungicide if needed.""",

    "Grape Esca Black Measles": """### Grape Esca Black Measles
- Prune infected parts carefully.
- Avoid plant stress.
- Consult an agricultural expert for vineyard-level management.""",

    "Grape Leaf Blight Isariopsis Leaf Spot": """### Grape Leaf Blight Isariopsis Leaf Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use proper fungicide if needed.""",

    "Orange Haunglongbing Citrus Greening": """### Orange Haunglongbing Citrus Greening
- Remove severely infected trees if confirmed.
- Control psyllid insect vectors.
- Consult local agricultural authority immediately.""",

    "Peach Bacterial Spot": """### Peach Bacterial Spot
- Remove infected leaves and twigs.
- Avoid overhead watering.
- Use copper-based spray if recommended locally.""",

    "Pepper Bell Bacterial Spot": """### Pepper Bell Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray if recommended.
- Improve airflow around the crop.""",

    "Potato Early Blight": """### Potato Early Blight
- Remove affected leaves.
- Apply fungicide if spread increases.
- Rotate crops and avoid repeated planting in same soil.""",

    "Potato Late Blight": """### Potato Late Blight
- Remove infected leaves immediately.
- Avoid leaf wetness.
- Use late blight fungicide if needed.""",

    "Squash Powdery Mildew": """### Squash Powdery Mildew
- Remove infected leaves.
- Improve air circulation.
- Use sulfur or suitable fungicide if needed.""",

    "Strawberry Leaf Scorch": """### Strawberry Leaf Scorch
- Remove infected leaves.
- Avoid overhead watering.
- Use fungicide if disease spreads significantly.""",

    "Tomato Bacterial Spot": """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid splashing water on leaves.
- Use copper-based bactericide if needed.
- Keep proper spacing between plants.""",

    "Tomato Early Blight": """### Tomato Early Blight
- Remove lower infected leaves.
- Use fungicide if spread continues.
- Rotate crops and mulch the soil.""",

    "Tomato Late Blight": """### Tomato Late Blight
- Remove infected leaves quickly.
- Avoid overhead watering.
- Use late blight fungicide if needed.""",

    "Tomato Leaf Mold": """### Tomato Leaf Mold
- Improve ventilation around plants.
- Avoid high humidity.
- Use fungicide if disease spreads.""",

    "Tomato Septoria Leaf Spot": """### Tomato Septoria Leaf Spot
- Remove infected lower leaves.
- Avoid splashing soil onto leaves.
- Use fungicide if infection progresses.""",

    "Tomato Spider Mites Two Spotted Spider Mite": """### Tomato Spider Mites Two Spotted Spider Mite
- Spray water under leaves to reduce mites.
- Use neem oil or miticide if needed.
- Remove heavily infested leaves.""",

    "Tomato Target Spot": """### Tomato Target Spot
- Remove infected leaves.
- Improve airflow and reduce humidity.
- Use appropriate fungicide if needed.""",

    "Tomato Yellow Leaf Curl Virus": """### Tomato Yellow Leaf Curl Virus
- Remove severely infected plants.
- Control whiteflies.
- Use resistant varieties if possible.""",

    "Tomato Mosaic Virus": """### Tomato Mosaic Virus
- Remove infected plants.
- Sanitize hands and tools.
- Avoid tobacco contact near plants.""",

    "Plant is Healthy": "No treatment is needed.",
}


def transform_for_prediction(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    tensor = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
    tensor = F.to_tensor(tensor)
    tensor = F.normalize(
        tensor,
        mean=ModelConfig.IMG_MEAN,
        std=ModelConfig.IMG_STD,
    )
    return tensor.to(get_device()[1])


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        label = str(label)

    label = label.replace("__", " ").strip()
    label = " ".join(label.split())

    # common normalization fixes
    replacements = {
        "Apple Healthy": "Plant is Healthy",
        "Blueberry Healthy": "Plant is Healthy",
        "Cherry Healthy": "Plant is Healthy",
        "Corn Healthy": "Plant is Healthy",
        "Grape Healthy": "Plant is Healthy",
        "Peach Healthy": "Plant is Healthy",
        "Pepper Bell Healthy": "Plant is Healthy",
        "Potato Healthy": "Plant is Healthy",
        "Raspberry Healthy": "Plant is Healthy",
        "Soybean Healthy": "Plant is Healthy",
        "Strawberry Healthy": "Plant is Healthy",
        "Tomato Healthy": "Plant is Healthy",
    }

    titled = label.title()
    return replacements.get(titled, titled)


def get_offline_remedy(label: str) -> str:
    return REMEDY_DB.get(
        label,
        "General plant care: remove infected leaves, avoid overwatering, improve airflow, and consult an agricultural expert if symptoms continue."
    )


def classify_disease(image_tensor: torch.Tensor):
    if CLF_MODEL is None:
        raise RuntimeError("Classification model failed to load. Check server logs.")

    with torch.no_grad():
        model_to_run = CLF_MODEL.model if hasattr(CLF_MODEL, "model") else CLF_MODEL
        outputs = model_to_run(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

        prediction_idx = top_idx.item()
        confidence = top_prob.item()

        raw_label = ServiceConfig.ID2LABEL[prediction_idx]

        print("=== TOP PREDICTION ===", flush=True)
        print(f"{prediction_idx} -> {raw_label} ({confidence:.4f})", flush=True)

    return raw_label, confidence


def workflow(image: Image.Image):
    try:
        print("DEBUG: CLEAN WORKFLOW RUNNING", flush=True)

        image_tensor = transform_for_prediction(image).unsqueeze(0)

        raw_label, confidence = classify_disease(image_tensor)
        classifier_label = normalize_label(raw_label)

        print(f"[INFO] classifier label: {classifier_label}", flush=True)
        print(f"[INFO] classifier confidence: {confidence:.4f}", flush=True)

        # 1) explicit healthy prediction
        if "healthy" in raw_label.lower() or classifier_label == "Plant is Healthy":
            return (
                "Plant is Healthy",
                "No treatment is needed."
            )

        # 2) very low confidence => uncertain
        if confidence < 0.35:
            return (
                "Uncertain",
                "Model is not confident. Please try another clear leaf image."
            )

        # 3) disease prediction
        remedy = get_offline_remedy(classifier_label)
        return classifier_label, remedy

    except Exception as e:
        print("[ERROR] Workflow failed:", e, flush=True)
        traceback.print_exc()
        return (
            "Error",
            f"An error occurred: {str(e)}"
        )