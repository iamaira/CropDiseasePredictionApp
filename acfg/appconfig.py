from dataclasses import dataclass
import os
import torch
import traceback

from acfg.modelconfig import ModelConfig
from ml.app.anomaly import DiseaseOODModule
from ml.app.models.classification import DiseaseClassificationModel, ClassificationModule


# ✅ Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Model checkpoints inside service/static subfolders
CLASSIFY_MODEL_CHECKPOINT = os.path.join(
    BASE_DIR, "../service/static/PlantDiseaseClassificationModel/best.ckpt"
)

OOD_MODEL_CHECKPOINT = os.path.join(
    BASE_DIR, "../service/static/PlantDiseaseOODModel/best.ckpt"
)


def get_device():
    """Gets the appropriate device for PyTorch operations."""
    if torch.cuda.is_available():
        return "cuda", "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps", "mps"
    else:
        return "cpu", "cpu"


@dataclass
class ServiceConfig:
    LLM_MODEL_KEY = "gemini"
    OOD_THRESHOLD = 0.034
    ID2LABEL = (
        "Apple scab",
        "Apple Black rot",
        "Apple Cedar rust",
        "Apple healthy",
        "Blueberry healthy",
        "Cherry Powdery mildew",
        "Cherry healthy",
        "Corn Cercospora leaf spot Gray leaf spot",
        "Corn Common rust",
        "Corn Northern Leaf Blight",
        "Corn healthy",
        "Grape Black rot",
        "Grape Esca Black Measles",
        "Grape Leaf blight Isariopsis Leaf Spot",
        "Grape healthy",
        "Orange Haunglongbing Citrus greening",
        "Peach Bacterial spot",
        "Peach healthy",
        "Pepper bell Bacterial spot",
        "Pepper bell healthy",
        "Potato Early blight",
        "Potato Late blight",
        "Potato healthy",
        "Raspberry healthy",
        "Soybean healthy",
        "Squash Powdery mildew",
        "Strawberry Leaf scorch",
        "Strawberry healthy",
        "Tomato Bacterial spot",
        "Tomato Early blight",
        "Tomato Late blight",
        "Tomato Leaf Mold",
        "Tomato Septoria leaf spot",
        "Tomato Spider mites Two spotted spider mite",
        "Tomato Target Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Tomato mosaic virus",
        "Tomato healthy",
    )


# ✅ Load classification model with error handling
CLF_MODEL = None
OOD_MODEL = None

try:
    print("[INFO] Loading classification model...")
    CLF_MODEL = ClassificationModule.load_from_checkpoint(
        CLASSIFY_MODEL_CHECKPOINT,
        model=DiseaseClassificationModel(
            ModelConfig.PRETRAINED_MODEL_NAME,
            num_classes=len(ServiceConfig.ID2LABEL)
        ),
        num_classes=len(ServiceConfig.ID2LABEL)
    ).to(get_device()[1])

    CLF_MODEL.eval()
    print("[INFO] Classification model loaded successfully")
    WORKFLOW_READY = True

except Exception as e:
    print("[WARNING] Classification model failed:", e)
    CLF_MODEL = None
    WORKFLOW_READY = False

except Exception as e:
    print("[WARNING] Classification model failed:", e)
    CLF_MODEL = None
    WORKFLOW_READY = False
try:
    print("[INFO] Loading OOD model...")
    OOD_MODEL = DiseaseOODModule.load_from_checkpoint(
        OOD_MODEL_CHECKPOINT
    ).to(get_device()[1])
    OOD_MODEL.eval()
    print("[INFO] OOD model loaded successfully")
    WORKFLOW_READY = True

except Exception as e:
    print("[WARNING] OOD model failed, continuing without it:", e)
    OOD_MODEL = None
    WORKFLOW_READY = False