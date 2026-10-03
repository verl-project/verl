import json
import os

import trimesh
from datasets import Dataset, Features, Image, Value
from huggingface_hub import login

# Log in to HF Hub (optional if you've already done `huggingface-cli login`)
login(token=os.getenv("HF_TOKEN"))  # Or replace with string token

# Paths
image_dir = "dataset/images"
stl_dir = "dataset/stls"
labels_path = "dataset/labels.json"

# Load labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# Build data entries
data = []
for image_filename in os.listdir(image_dir):
    if not image_filename.endswith(".png"):
        continue
    image_path = os.path.join(image_dir, image_filename)

    # Extract base ID
    base_id = image_filename.split("_")[0]

    stl_path = os.path.join(stl_dir, f"{base_id}.stl")
    label = labels.get(base_id, "unknown")

    # Load STL features (e.g., centroid + bounding box + volume as 9 floats)
    stl_features = [0.0] * 9
    if os.path.exists(stl_path):
        try:
            mesh = trimesh.load(stl_path, force="mesh")
            bbox = mesh.bounding_box.extents
            centroid = mesh.centroid
            volume = mesh.volume
            stl_features = list(centroid) + list(bbox) + [volume]
        except Exception as e:
            print(f"⚠️ Failed to process {stl_path}: {e}")

    data.append(
        {
            "image": image_path,
            "label": label,
            "stl_features": stl_features,
            "id": base_id,
        }
    )

# Define dataset schema
features = Features(
    {
        "id": Value("string"),
        "image": Image(),  # Load images from file paths
        "label": Value("string"),
        "stl_features": Value("string"),  # Store as JSON string for simplicity
    }
)

# Convert stl_features to JSON strings for compatibility
for item in data:
    item["stl_features"] = json.dumps(item["stl_features"])

# Create Dataset
dataset = Dataset.from_list(data).cast(features)

# Push to Hub
dataset.push_to_hub("venkatacrc/stl-image-dataset", private=True)
