import os
import shutil
import zipfile

base_dir = os.path.abspath(os.path.join("ml", "input", "PlantDiseaseClassificationDataset", "train"))
zip_path = os.path.abspath(os.path.join("ml", "input", "plantdisease.zip"))

print(f"Base train path: {base_dir}")
print(f"ZIP path: {zip_path}")

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"ZIP file not found: {zip_path}")

if os.path.exists(os.path.join(base_dir, "PlantVillage")):
    print("Removing existing nested PlantVillage folder...")
    shutil.rmtree(os.path.join(base_dir, "PlantVillage"))

os.makedirs(base_dir, exist_ok=True)

import pathlib

with zipfile.ZipFile(zip_path, "r") as archive:
    for member in archive.infolist():
        # Normalize POSIX ZIP paths to the current OS path format
        member_path = pathlib.PurePosixPath(member.filename)
        target_path = pathlib.Path(base_dir, *member_path.parts)

        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, open(target_path, "wb") as target:
            target.write(source.read())

print("Extraction complete.")
print("Train directory content sample:")
for item in sorted(os.listdir(base_dir))[:50]:
    print(item)
