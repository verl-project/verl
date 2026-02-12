import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"CLIPScorer initialized on {self.device} with {model_name}")
        except Exception as e:
            print(
                f"Error initializing CLIPModel: {e}. Ensure model name is correct and you have internet."
            )
            self.model = None
            self.processor = None
            raise

    @torch.no_grad()  # Ensure no gradients are computed during inference
    def score_images(self, images_np_list: list, target_text_description: str):
        if not self.model or not self.processor:
            print("CLIPScorer not properly initialized.")
            return [0.0] * len(images_np_list)  # Low score on error

        try:
            pil_images = [
                Image.fromarray(img_arr.astype(np.uint8)) for img_arr in images_np_list
            ]

            inputs = self.processor(
                text=[target_text_description],  # Single text prompt
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            image_text_similarity_scores = (
                outputs.logits_per_image.squeeze().tolist()
            )  # Squeeze to remove the text dim

            if not isinstance(image_text_similarity_scores, list):  # If only one image
                image_text_similarity_scores = [image_text_similarity_scores]

            return image_text_similarity_scores

        except Exception as e:
            print(f"Error in CLIP scoring: {e}")
            return [0.0] * len(images_np_list)  # Low score on error
