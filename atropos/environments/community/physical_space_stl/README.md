# Physical Environment

Our project is a physical environment to train LLMs to generate STL files, the same files used in physical CAD designs.

## Setup

```sh
$ pip install pyrender trimesh pyglet matplotlib torch transformers pydantic vllm numpy requests tenacity wandb
```

Shared libraries for Ubuntu GL rendering.
```sh
$ sudo apt-get install libglfw3-dev libgles2-mesa-dev libnvidia-gl-570-server
```

# Training Data Generation
- Use `dataset_scr.py` to create the following directory structure:
  ```
  dataset/
  ├── stls/
  │   ├── model_0001.stl
  │   ├── model_0002.stl
  │   └── ...
  ├── images/
  │   ├── model_0001.png
  │   ├── model_0002.png
  │   └── ...
  └── labels.json
  ```
- Use `render_stl.py` to generate images from STL files.
- Use `llm_label.py` to label the STL and image files.
- Use `prepare_push_hf_dataset.py` to push the dataset to Hugging Face.

Generated run: https://wandb.ai/csxl/atropos-environments_hack0/runs/dlexyg5r
Training run (ran out of memory): https://wandb.ai/csxl/grpo-physical-trainer/runs/t61am7gu
