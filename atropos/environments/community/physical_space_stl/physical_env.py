import glob
import os
import random
import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import trimesh
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Fix the relative imports for running directly
try:
    from .judgement_model import CLIPScorer
    from .pyrender_utils import PyRenderOffline
except ImportError:
    from judgement_model import CLIPScorer
    from pyrender_utils import PyRenderOffline

system_prompt = (
    "You are an expert in 3D modeling and computer-aided design. Your task is to analyze the "
    "blueprints or wireframe views of objects and generate the corresponding STL file content. "
    "STL (stereolithography) files represent 3D models as a collection of triangular facets.\n\n"
    "You may use <think> </think> tags to work through your reasoning about the shape, "
    "dimensions, and geometric features of the model. Be methodical in your approach.\n\n"
    "STL files can be in ASCII or binary format. For this task, generate ASCII STL content that "
    "accurately represents the 3D model shown in the provided views.\n\n"
    "Your final output must be enclosed in <stl> </stl> tags, containing only the valid STL content "
    "and nothing else. The STL content should begin with 'solid' and end with 'endsolid'.\n\n"
    "Example of STL format:\n"
    "<stl>\n"
    "solid model\n"
    "  facet normal nx ny nz\n"
    "    outer loop\n"
    "      vertex x1 y1 z1\n"
    "      vertex x2 y2 z2\n"
    "      vertex x3 y3 z3\n"
    "    endloop\n"
    "  endfacet\n"
    "  ... more facets ...\n"
    "endsolid model\n"
    "</stl>"
)


class PhysicalRow(TypedDict):
    prompt: str
    image: np.ndarray
    stl: str


class PhysicalEnv(BaseEnv):
    name = "physical"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        # Initialize renderer and CLIP scorer
        self.renderer = PyRenderOffline(width=224, height=224)
        self.clip_scorer = CLIPScorer()

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="google/gemma-3-27b-it",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            max_token_length=2048,
            wandb_name="physical",
        )
        server_configs = [
            APIServerConfig(
                model_name="google/gemma-3-27b-it",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    def load_stl_file(self, stl_path):
        """Load an STL file into a trimesh object"""
        try:
            mesh = trimesh.load(stl_path)
            return mesh
        except Exception as e:
            print(f"Error loading STL file {stl_path}: {e}")
            return None

    def generate_query_from_images(self, images):
        """Generate a query based on the rendered images of the STL file"""
        # In a real implementation, this would use a vision model to generate a description
        # For this simplified version, we'll use different templates to add variety
        templates = [
            "Create a 3D model (STL file) for the object shown in these technical drawings. "
            "Be precise with the geometry.",
            "Based on these wireframe views, generate the STL code for this 3D object. "
            "Pay attention to all visible features.",
            "Using these blueprint images as reference, provide the STL file format data "
            "to recreate this 3D model.",
            "These are technical views of a 3D object. Generate the STL representation "
            "that would produce this exact shape.",
            "Reconstruct this 3D model from the provided wireframe views and output "
            "the STL file content.",
        ]
        return random.choice(templates)

    async def setup(self):
        # Load all STL files from sample_data
        self.stl_files = glob.glob(os.path.join("sample_data", "*.stl"))
        if not self.stl_files:
            raise ValueError("No STL files found in the sample_data directory")

        print(f"Found {len(self.stl_files)} STL files")

        # Split files into train and test sets (80/20 split)
        random.seed(42)
        random.shuffle(self.stl_files)
        split_idx = int(len(self.stl_files) * 0.8)
        self.train_files = self.stl_files[:split_idx]
        self.test_files = self.stl_files[split_idx:]

        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, stl_path: str) -> number:
        # Load the STL file
        mesh = self.load_stl_file(stl_path)
        if mesh is None:
            return 0

        # Render the images
        images = self.renderer.render_mesh_to_images(mesh)

        # Generate a query from the images
        query = self.generate_query_from_images(images)

        # Get a completion from the model
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        # Extract the STL content from the completion
        response_content = completion.choices[0].message.content
        stl_content = self.extract_stl_content(response_content)

        # Load the original mesh directly
        original_mesh = mesh

        # Save the generated STL content to a temporary file
        temp_file = f"temp_generated_{random.randint(1000, 9999)}.stl"
        try:
            with open(temp_file, "w") as f:
                f.write(stl_content)

            # Load the generated mesh
            generated_mesh = trimesh.load(temp_file)

            # Score the generated mesh against the original
            score = self.score_meshes_similarity(original_mesh, generated_mesh)

            # Cleanup
            os.remove(temp_file)

            return score
        except Exception as e:
            print(f"Error processing generated STL: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return 0

    def extract_stl_content(self, response_content):
        """Extract STL content from the model's response"""
        # Find content between <stl> and </stl> tags
        match = re.search(r"<stl>(.*?)</stl>", response_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def score_meshes_similarity(self, original_mesh, generated_mesh):
        """Score the similarity between two meshes"""
        # This is a simple implementation - in practice you'd want more sophisticated metrics
        # Compare basic properties like number of vertices, faces, and volume
        orig_stats = {
            "vertices": len(original_mesh.vertices),
            "faces": len(original_mesh.faces),
            "volume": original_mesh.volume or 1.0,
            "surface_area": original_mesh.area or 1.0,
        }

        gen_stats = {
            "vertices": len(generated_mesh.vertices),
            "faces": len(generated_mesh.faces),
            "volume": generated_mesh.volume or 1.0,
            "surface_area": generated_mesh.area or 1.0,
        }

        # Calculate ratios (capped at 1.0 for when generated > original)
        vertex_ratio = min(gen_stats["vertices"] / max(orig_stats["vertices"], 1), 1.0)
        face_ratio = min(gen_stats["faces"] / max(orig_stats["faces"], 1), 1.0)
        volume_ratio = min(gen_stats["volume"] / max(orig_stats["volume"], 1), 1.0)
        area_ratio = min(
            gen_stats["surface_area"] / max(orig_stats["surface_area"], 1), 1.0
        )

        # Average the ratios for a final score
        score = (vertex_ratio + face_ratio + volume_ratio + area_ratio) / 4.0
        return score

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for stl_file in self.test_files[
            :10
        ]:  # Limit to 10 files for evaluation to keep it manageable
            eval_tasks.append(self.rollout_and_score_eval(stl_file))
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/similarity_score", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: PhysicalRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        stl_path = item["stl_path"]
        mesh = self.load_stl_file(stl_path)
        images = self.renderer.render_mesh_to_images(mesh)
        query = self.generate_query_from_images(images)

        # For original STL content, we'll just store the file path instead of the content
        # as the files may be binary and can't be simply read as text
        original_stl_path = stl_path

        user_message = {"role": "user", "content": query}

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "original_stl_path": original_stl_path,
                    "images": images,
                    "finish_reason": chat_completion.finish_reason,
                }
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            response_content = item["messages"][-1]["content"]
            stl_content = self.extract_stl_content(response_content)

            # Save the generated STL content to a temporary file
            temp_file = f"temp_generated_{random.randint(1000, 9999)}.stl"
            try:
                with open(temp_file, "w") as f:
                    f.write(stl_content)

                # Load the original STL directly from its path
                original_stl_path = item["original_stl_path"]
                original_mesh = trimesh.load(original_stl_path)

                # Load the generated mesh
                generated_mesh = trimesh.load(temp_file)

                # Score the generated mesh against the original
                mesh_similarity = self.score_meshes_similarity(
                    original_mesh, generated_mesh
                )

                # Generate rendered images of the produced STL
                generated_images = self.renderer.render_mesh_to_images(generated_mesh)

                # Use CLIP to score the visual similarity
                images_reward = 0.0
                if len(generated_images) > 0 and len(item["images"]) > 0:
                    # Extract query from the user message
                    query = item["messages"][1]["content"]

                    # Score the visual similarity using CLIP
                    clip_scores = self.clip_scorer.score_images(generated_images, query)
                    images_reward = (
                        sum(clip_scores) / len(clip_scores) / 100.0
                    )  # Normalize to roughly 0-1

                # Combine mesh similarity and image similarity for final reward
                reward = 0.5 * mesh_similarity + 0.5 * images_reward

                out_dict = tokenize_for_trainer(
                    self.tokenizer, item["messages"], item["finish_reason"]
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

                # Remove obviously bad examples
                if len([1 for i in masks if i != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(reward)

                self.percent_correct_buffer.append(reward)

                # Clean up temporary files
                os.remove(temp_file)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            except Exception as e:
                print(f"Error in scoring: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Apply length penalty if all scores are similar
        if all(abs(score - scores["scores"][0]) < 0.1 for score in scores["scores"]):
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) > 0:
                max_allowed_length = self.config.max_token_length
                length_threshold = max_allowed_length * 0.5

                scores["scores"] = []
                for length in token_lengths:
                    if length <= length_threshold:
                        scores["scores"].append(1.0)
                    else:
                        percentage_of_range = (length - length_threshold) / (
                            max_allowed_length - length_threshold
                        )
                        percentage_of_range = min(percentage_of_range, 1.0)
                        scores["scores"].append(1.0 - percentage_of_range)

        if all([scores["scores"][0] == score for score in scores["scores"]]):
            return None  # If all the same, we return None

        return scores

    async def get_next_item(self) -> PhysicalRow:
        stl_path = self.train_files[self.iter % len(self.train_files)]
        self.iter += 1

        # Load the STL file and render it
        mesh = self.load_stl_file(stl_path)
        if mesh is None:
            # Skip this file and try the next one if there's an issue
            return await self.get_next_item()

        # Render the mesh to get images
        images = self.renderer.render_mesh_to_images(mesh)

        # Generate a query from the images
        query = self.generate_query_from_images(images)

        # Return a row with the prompt, image, and path to the STL file
        return {
            "prompt": query,
            "image": images[0] if images else np.zeros((224, 224, 3), dtype=np.uint8),
            "stl_path": stl_path,
        }

    @classmethod
    async def test_sample_stl(cls):
        """Test loading and rendering a sample STL file"""
        # Create temporary environment instance
        env_config = BaseEnvConfig(
            tokenizer_name="google/gemma-3-27b-it",
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            max_token_length=2048,
            wandb_name="physical_test",
        )

        server_configs = [
            APIServerConfig(
                model_name="google/gemma-3-27b-it",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        env = cls(env_config, server_configs, slurm=False, testing=True)

        # Find sample STL files
        stl_files = glob.glob(os.path.join("sample_data", "*.stl"))
        if not stl_files:
            print("No STL files found in sample_data/")
            return

        # Test loading and rendering the first file
        print(f"Testing with STL file: {stl_files[0]}")
        mesh = env.load_stl_file(stl_files[0])
        if mesh is None:
            print("Failed to load STL file")
            return

        print(
            f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        )

        # Render the mesh
        try:
            images = env.renderer.render_mesh_to_images(mesh)
            print(f"Successfully rendered {len(images)} images")

            # Save the first image for inspection
            from PIL import Image

            img = Image.fromarray(images[0])
            img.save("test_render.png")
            print("Saved test render to test_render.png")
        except Exception as e:
            print(f"Error rendering: {e}")

        print("Test completed")


if __name__ == "__main__":
    PhysicalEnv.cli()
