import random

# ✅ Remedy database
REMEDY_DB = {
    "Apple Cedar Rust": """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts.
- Prune infected leaves.
- Use fungicide in early season.
""",

    "Tomato Bacterial Spot": """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray.
""",

    "Plant is Healthy": "No treatment is needed."
}


def workflow(image):
    """
    Temporary smart logic (for demo)
    """

    # 👇 RANDOM PREDICTION (demo ke liye)
    prediction = random.choice([
        "Plant is Healthy",
        "Apple Cedar Rust",
        "Tomato Bacterial Spot"
    ])

    # ✅ Healthy case
    if prediction == "Plant is Healthy":
        return "Plant is Healthy", "No treatment is needed."

    # ✅ Disease case
    remedy = REMEDY_DB.get(prediction, "No remedy available.")
    return prediction, remedy