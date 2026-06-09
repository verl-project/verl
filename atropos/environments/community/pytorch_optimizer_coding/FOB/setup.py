from setuptools import setup

with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="pytorch-fob",
    version="0.1.0",
    description="Fast Optimizer Benchmark",
    url="https://github.com/automl/fob",
    author="Simon Blauth, Tobias Bürger, Zacharias Häringer",
    license="MIT",
    packages=["pytorch_fob"],
    install_requires=requirements,
)
