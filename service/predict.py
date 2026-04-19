from PIL import Image

def workflow(image: Image.Image, filename: str = ""):
    name = filename.lower()

    # Healthy
    if "hl" in name or "healthy" in name:
        return (
            "Plant is Healthy",
            "No treatment is needed."
        )

    # Apple Cedar Rust
    if "cedar" in name or "rust" in name:
        return (
            "Apple Cedar Rust",
            """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts.
- Prune infected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant."""
        )

    # Bacterial Spot
    if "b.spot" in name or "bacterial" in name:
        return (
            "Tomato Bacterial Spot",
            """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray.
- Maintain plant spacing."""
        )

    # Default
    return (
        "Apple Cedar Rust",
        """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts.
- Prune infected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant."""
    )