import os
import shutil
import zipfile

base = os.path.join("ml", "input", "PlantDiseaseClassificationDataset", "train")
bad = os.path.join(base, "PlantVillage")
if os.path.exists(bad):
    shutil.rmtree(bad)

zip_path = os.path.join("ml", "input", "plantdisease.zip")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(base)

print("done. train contents:")
print("\n".join(sorted(os.listdir(base))[:50]))