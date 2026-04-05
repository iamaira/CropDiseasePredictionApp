import os
import zipfile
import shutil
import random

# --- Config ---
base_dir = "ml/input/PlantDiseaseClassificationDataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
split_ratio = 0.8  # 80% train, 20% val

zip_path = r"C:\Users\ahsan\CropDiseasePredictionApp-main\CropDiseasePredictionApp-main\plantdisease.zip"  # your Kaggle downloaded zip

# --- Step 1: Extract ---
if not os.path.exists(base_dir) or not os.listdir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)
else:
    print("✅ Dataset already extracted")

# --- Step 2: Prepare train/val split ---
print("Organizing into train/val split...")

# If train/val already exist → skip
if os.path.exists(train_dir) and os.path.exists(val_dir):
    print("✅ Train/Val split already exists. Skipping split step.")
else:
    # Detect dataset root
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    if not subfolders:
        raise RuntimeError(
            f"No dataset folders found in {base_dir}. "
            "Delete base_dir and re-run script to re-extract."
        )

    if len(subfolders) == 1:
        extracted_root = os.path.join(base_dir, subfolders[0])  # e.g. "PlantVillage"
    else:
        extracted_root = base_dir  # classes are directly inside

    # Create train and val directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Loop through class folders
    for cls in os.listdir(extracted_root):
        cls_path = os.path.join(extracted_root, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        # Create class subfolders
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Move images (only if they still exist)
        for img in train_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_dir, cls, img)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

        for img in val_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_dir, cls, img)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

    # Cleanup original extracted folder (only if it’s a nested folder)
    if extracted_root != base_dir:
        shutil.rmtree(extracted_root, ignore_errors=True)

print("✅ Dataset ready at:", base_dir)