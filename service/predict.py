from PIL import Image
import torch
import torchvision.transforms.functional as F
import traceback

from acfg.modelconfig import ModelConfig
from acfg.appconfig import CLF_MODEL, ServiceConfig, get_device


REMEDY_DB = {
    "Apple Scab": """Apply fungicide at early leaf stage.
Prune infected leaves and fallen debris.
Keep good air circulation around the plant.""",

    "Apple Black Rot": """Remove infected fruits, leaves, and branches.
Prune dead wood and sanitize tools.
Use recommended fungicide if infection is severe.""",

    "Apple Cedar Rust": """Remove nearby cedar/juniper hosts if possible.
Prune affected leaves and branches.
Use protective fungicide in early season.""",

    "Apple Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Blueberry Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Cherry Powdery Mildew": """Remove infected leaves.
Improve airflow and avoid overhead watering.
Use sulfur or recommended fungicide if needed.""",

    "Cherry Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Corn Cercospora Leaf Spot Gray Leaf Spot": """Remove infected leaves if possible.
Avoid leaf wetness for long periods.
Use disease-resistant variety and fungicide if severe.""",

    "Corn Common Rust": """Monitor leaf pustules regularly.
Use resistant varieties where possible.
Apply fungicide if infection spreads rapidly.""",

    "Corn Northern Leaf Blight": """Remove infected crop debris.
Use resistant seeds.
Apply recommended fungicide if disease is severe.""",

    "Corn Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Grape Black Rot": """Remove infected leaves and fruits.
Prune the canopy for airflow.
Use recommended fungicide preventively.""",

    "Grape Esca Black Measles": """Prune infected parts carefully.
Avoid plant stress from drought or poor nutrition.
Consult an agricultural expert for vineyard-level management.""",

    "Grape Leaf Blight Isariopsis Leaf Spot": """Remove infected leaves.
Avoid overhead watering.
Use proper fungicide if needed.""",

    "Grape Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Orange Haunglongbing Citrus Greening": """Remove severely infected trees if confirmed.
Control psyllid insect vectors.
Consult local agricultural authority immediately.""",

    "Peach Bacterial Spot": """Remove infected leaves and twigs.
Avoid overhead watering.
Use copper-based spray if recommended locally.""",

    "Peach Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Pepper Bell Bacterial Spot": """Remove infected leaves.
Avoid splashing water on leaves.
Use copper-based bactericide if needed.""",

    "Pepper Bell Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Potato Early Blight": """Remove affected leaves.
Apply fungicide if spread increases.
Rotate crops and avoid planting potatoes repeatedly in same soil.""",

    "Potato Late Blight": """Remove infected leaves immediately.
Avoid leaf wetness and improve airflow.
Use late blight fungicide urgently if disease spreads.""",

    "Potato Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Raspberry Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Soybean Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Squash Powdery Mildew": """Remove infected leaves.
Improve air circulation.
Use sulfur or recommended fungicide if needed.""",

    "Strawberry Leaf Scorch": """Remove infected leaves.
Avoid overhead watering.
Use fungicide if disease spreads significantly.""",

    "Strawberry Healthy": """The leaf appears healthy.
No treatment is needed.
Maintain proper watering and nutrition.""",

    "Tomato Bacterial Spot": """Remove infected leaves.
Avoid wetting leaves during irrigation.
Use copper-based spray if recommended.""",

    "Tomato Early Blight": """Remove lower infected leaves.
Use fungicide if spread continues.
Rotate crops and mulch the soil.""",

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


def get_offline_remedy(label):
    remedies = {
        "Apple Cedar Rust": "Remove nearby cedar/juniper hosts...",
        "Tomato Bacterial Spot": "Use copper-based bactericide..."
    }
    return remedies.get(label, "No remedy available.")


def workflow(image: Image.Image):
    try:
        image_tensor = transform_for_prediction(image).unsqueeze(0)

        classifier_label, confidence = classify_disease(image_tensor)
        classifier_label = normalize_label(classifier_label)

        print(f"[INFO] classifier confidence: {confidence:.4f}", flush=True)

        # Healthy fallback for low-confidence cases
        if confidence<0.75:
            return("Uncertain","Image unclear.")
        if "healthy" in classifier_label.lower() and confidence >= 0.80:
            return ("Plant is Healthy","No treatment required.")
        remedy = get_offline_remedy(classifier_label)
        return(classifier_label,remedy)
    except Exception as e:
        print("[ERROR] Workflow failed:", e, flush=True)
        traceback.print_exc()
        return "Error", f"An error occurred: {str(e)}"