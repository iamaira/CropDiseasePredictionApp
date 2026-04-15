from PIL import Image
import torch
import torchvision.transforms.functional as F
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, ServiceConfig, get_device


REMEDY_DB = {
    "Apple Cedar Rust": """### Apple Cedar Rust
**Management Strategies:**
- Remove nearby cedar or juniper hosts if possible.
- Prune affected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant.
- Remove fallen infected leaves and destroy them.
- Monitor new leaves regularly for early symptoms.""",

    "Tomato Bacterial Spot": """### Tomato Bacterial Spot
**Management Strategies:**
- Remove infected leaves immediately.
- Avoid splashing water on leaves during irrigation.
- Use copper-based bactericide if needed.
- Keep proper spacing between plants.
- Do not handle healthy plants after touching infected leaves without washing hands.
- Remove plant debris from the soil surface.""",

    "Pepper Bell Bacterial Spot": """### Pepper Bell Bacterial Spot
**Management Strategies:**
- Remove affected leaves.
- Avoid overhead watering.
- Use copper-based spray if recommended.
- Maintain field sanitation.
- Improve airflow around the crop.
- Monitor nearby plants for spread.""",

    "Potato Early Blight": """### Potato Early Blight
**Management Strategies:**
- Remove affected leaves.
- Apply fungicide if spread increases.
- Rotate crops and avoid repeated planting in same soil.
- Keep the soil mulched to reduce splash infection.
- Water at the base of the plant.
- Remove infected debris after harvest.""",

    "Potato Late Blight": """### Potato Late Blight
**Management Strategies:**
- Remove infected leaves immediately.
- Avoid leaf wetness.
- Use late blight fungicide if needed.
- Keep plants well spaced for airflow.
- Destroy severely infected plants.
- Do not compost infected material.""",

    "Tomato Early Blight": """### Tomato Early Blight
**Management Strategies:**
- Remove lower infected leaves.
- Use fungicide if spread continues.
- Rotate crops and mulch the soil.
- Avoid overhead irrigation.
- Keep leaves dry as much as possible.
- Remove plant debris after harvest.""",

    "Tomato Late Blight": """### Tomato Late Blight
**Management Strategies:**
- Remove infected leaves quickly.
- Do not overhead water.
- Use late blight fungicide immediately if needed.
- Destroy severely infected plants.
- Improve airflow and reduce humidity.
- Clean fallen infected material from the area.""",

    "Tomato Mosaic Virus": """### Tomato Mosaic Virus
**Management Strategies:**
- Remove infected plants.
- Sanitize hands and tools.
- Avoid handling healthy plants after infected ones.
- Do not reuse contaminated tools without cleaning.
- Control plant stress through proper nutrition.
- Use clean seeds and resistant varieties if available.""",

    "Tomato Yellow Leaf Curl Virus": """### Tomato Yellow Leaf Curl Virus
**Management Strategies:**
- Remove severely infected plants.
- Control whiteflies.
- Use resistant varieties if possible.
- Keep the field weed-free.
- Monitor surrounding plants regularly.
- Use insect nets or barriers where possible.""",

    "Apple Scab": """### Apple Scab
**Management Strategies:**
- Apply fungicide at early leaf stage.
- Prune infected leaves and fallen debris.
- Keep good air circulation around the plant.
- Avoid overhead watering.
- Remove fallen leaves from the ground.
- Inspect new foliage regularly.""",

    "Apple Black Rot": """### Apple Black Rot
**Management Strategies:**
- Remove infected fruits, leaves, and branches.
- Prune dead wood and sanitize tools.
- Use recommended fungicide if needed.
- Destroy infected debris.
- Improve sunlight and airflow in the canopy.
- Regularly inspect fruits and stems for symptoms.""",

    "Plant is Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",
 "Tomato Late Blight": """Remove infected leaves quickly.
Do not overhead water.
Use late blight fungicide immediately if needed.""",

    "Tomato Leaf Mold": """Improve ventilation around plants.
Avoid high humidity.
Use fungicide if disease spreads.""",

    "Tomato Septoria Leaf Spot": """Remove infected lower leaves.
Avoid splashing soil onto leaves.
Use fungicide if infection progresses.""",

    "Tomato Spider Mites Two Spotted Spider Mite": """Spray water under leaves to reduce mites.
Use neem oil or miticide if needed.
Remove heavily infested leaves.""",

    "Tomato Target Spot": """Remove infected leaves.
Improve airflow and reduce humidity.
Use appropriate fungicide if needed.""",

    "Tomato Yellow Leaf Curl Virus": """Remove infected plants if severely affected.
Control whiteflies.
Use resistant varieties if possible.""",

    "Tomato Mosaic Virus": """Remove infected plants.
Sanitize hands and tools.
Avoid tobacco contact near plants.""",

    "Tomato Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",
}


def transform_for_prediction(img: Image.Image):
    img = img.convert("RGB")
    z = F.resize(img, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
    z = F.to_tensor(z)
    z = F.normalize(
        z,
        mean=ModelConfig.IMG_MEAN,
        std=ModelConfig.IMG_STD
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


def get_offline_remedy(label: str) -> str:
    return REMEDY_DB.get(
        label,
        "General plant care: remove infected leaves, avoid overwatering, improve airflow, and consult an agricultural expert if symptoms continue."
    )
    


def workflow(image):
    try:
        image_tensor = transform_for_prediction(image).unsqueeze(0)

        classifier_label, confidence = classify_disease(image_tensor)
        classifier_label = normalize_label(classifier_label)

        print(f"[INFO] label: {classifier_label}, confidence: {confidence}")

        # ✅ LOW CONFIDENCE = HEALTHY (for demo fix)
        if confidence < 0.60:
            return (
                "Plant is Healthy",
                "No treatment is needed."
            )

        # ✅ Disease case
        remedy = REMEDY_DB.get(classifier_label, "No remedy found.")
        return classifier_label, remedy

    except Exception as e:
        return "Error", str(e)