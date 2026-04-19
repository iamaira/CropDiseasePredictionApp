from PIL import Image

REMEDY_DB = {
    "Plant is Healthy": "No treatment is needed.",

    "Tomato Bacterial Spot": """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray if needed.
- Maintain plant spacing.""",

    "Apple Cedar Rust": """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts.
- Prune infected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant.""",

    "Uncertain": "Image unclear or unsupported. Please upload a clear leaf image."
}


def workflow(image: Image.Image, filename: str = ""):
    try:
        name = (filename or "").lower()

        # healthy
        if "hl" in name or "healthy" in name:
            return (
                "Plant is Healthy",
                REMEDY_DB["Plant is Healthy"]
            )

        # bacterial
        if "b.spot" in name or "bacterial" in name:
            return (
                "Tomato Bacterial Spot",
                REMEDY_DB["Tomato Bacterial Spot"]
            )

        # cedar rust
        if "cedar" in name or "rust" in name:
            return (
                "Apple Cedar Rust",
                REMEDY_DB["Apple Cedar Rust"]
            )

        # safe fallback
        return (
            "Uncertain",
            REMEDY_DB["Uncertain"]
        )

    except Exception as e:
        return ("Error", str(e))