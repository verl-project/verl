# Using FOB with NePS for HPO
Run all commands from the root of the FOB repository.

## Setup
```bash
conda create -n fob-neps python=3.10 -y
conda activate fob-neps
pip install -r requirements.txt
pip install -r examples/neps/requirements.txt # this will downgrade some packages
pip install -e .
```

## Example
```bash
python examples/neps/hpo.py examples/neps/experiment.yaml
```
