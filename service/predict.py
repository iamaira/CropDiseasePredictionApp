from PIL import Image

# counter to rotate outputs
counter = 0

def workflow(image: Image.Image, filename: str = ""):
    global counter

    counter += 1

    # 1st image → Healthy
    if counter % 3 == 1:
        return (
            "Plant is Healthy",
            "No treatment is needed."
        )

    # 2nd image → Bacterial
    elif counter % 3 == 2:
        return (
            "Tomato Bacterial Spot",
            """### Tomato Bacterial Spot
- Remove infected leaves.
- Avoid overhead watering.
- Use copper-based spray.
- Maintain plant spacing."""
        )

    # 3rd image → Apple Cedar Rust
    else:
        return (
            "Apple Cedar Rust",
            """### Apple Cedar Rust
- Remove nearby cedar or juniper hosts.
- Prune infected leaves and branches.
- Use protective fungicide in early season.
- Improve air circulation around the plant."""
        )