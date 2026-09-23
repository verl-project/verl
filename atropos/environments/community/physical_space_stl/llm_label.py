import json
import os

import torch
import trimesh
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

STL_DIR = "dataset/stls"
IMG_DIR = "dataset/images"
LABEL_FILE = "dataset/labels.json"

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Load Mistral or other small LLM
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name, torch_dtype=torch.float16, device_map="auto"
)


def extract_trimesh_features(mesh):
    return {
        "volume": mesh.volume,
        "surface_area": mesh.area,
        "bounding_box": mesh.bounding_box.extents.tolist(),
        "num_faces": len(mesh.faces),
        "num_vertices": len(mesh.vertices),
        "is_watertight": mesh.is_watertight,
        "euler_number": mesh.euler_number,
    }


def caption_image(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_label_with_llm(features, caption):
    prompt = f"""You are a 3D object classifier.

Mesh features:
{json.dumps(features, indent=2)}

Rendered Image Caption:
"{caption}"

Based on this information, return a short label (1-3 words) that describes the object.
Label:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    output_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("Label:")[-1].strip()


def main():
    labels = {}

    for filename in os.listdir(STL_DIR):
        if not filename.endswith(".stl"):
            continue

        stem = os.path.splitext(filename)[0]
        stl_path = os.path.join(STL_DIR, filename)
        img_path = os.path.join(IMG_DIR, f"{stem}_0001.png")

        if not os.path.exists(img_path):
            print(f"Missing image for {stem}, skipping...")
            continue

        try:
            mesh = trimesh.load_mesh(stl_path)
            features = extract_trimesh_features(mesh)
            caption = caption_image(img_path)
            label = generate_label_with_llm(features, caption)
            labels[stem] = label
            print(f"Labeled {stem}: {label}")
        except Exception as e:
            print(f"Error processing {stem}: {e}")

    with open(LABEL_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSaved labels to {LABEL_FILE}")


if __name__ == "__main__":
    main()
