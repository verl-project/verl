from setuptools import setup, find_packages

setup(
    name="atropos-verl-integration",
    version="0.1.0",
    description="Atropos integration for verl - GRPO training with external rollouts",
    author="Nous Research",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28.0",
        "tenacity>=8.0.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "vllm>=0.3.0",
        "ray>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "protobuf>=3.20.0",
        "pydantic>=2.0.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
