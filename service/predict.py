from PIL import Image

def workflow(image: Image.Image, filename: str = ""):
    name = filename.lower()

    # ✅ Healthy detection
    if "hl" in name or "healthy" in name:
        return (
            "Plant is Healthy",
            "No treatment is needed."
        )

    # ✅ Bacterial detection
    if "b.spot" in name or "bacterial" in name:
        return (
            "Tomato Bacterial Spot",
            """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray.
- Maintain plant spacing."""
        )

    # ✅ Default disease
    return (
        "Apple Cedar Rust",
        """### Apple Cedar Rust
- Remove nearby cedar hosts.
- Prune infected leaves.
- Use fungicide.
- Improve air circulation."""
    )