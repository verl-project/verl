"""
Deployment script for optimizer code execution using Modal.
Organizes Modal app, images, volumes, and sandbox for running optimizer code in isolation.
"""

import datetime
import os
import time

from modal.app import App
from modal.image import Image
from modal.sandbox import Sandbox
from modal.secret import Secret
from modal.volume import Volume

# --- Modal App and Images ---
APP_NAME = "optimizer-test"
SANDBOX_APP_NAME = "new_sandbox_test"

app = App(APP_NAME)

base_image = Image.debian_slim().pip_install("dotenv", "datetime")
sandbox_image = Image.debian_slim().pip_install(
    "groq", "dotenv", "datetime", "torch", "numpy"
)

# --- Volumes and Secrets ---
benchmark_volume = Volume.from_name("benchmark-responses")
optimizers_volume = Volume.from_name("optimizers")
sys_prompt_volume = Volume.from_name("optimizerSystemPrompt")


# --- Utility Functions ---
def _write_optimizer_code_to_volume(code: str, volume: Volume) -> str:
    """Write the optimizer code to the optimizers volume and return the filename."""
    optimizer_code = code.replace("```python", "").replace("```", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimizer_{timestamp}.py"
    file_path = os.path.join("/optimizers", filename)
    with open(file_path, "w") as f:
        f.write(optimizer_code)
    volume.commit()
    print(f"Committed {filename} to volume.")
    return filename, optimizer_code


# --- Modal Function ---
@app.function(
    image=base_image,
    scaledown_window=60 * 5,
    min_containers=2,
    timeout=60 * 30,
    volumes={"/optimizers": optimizers_volume, "/sysPrompt": sys_prompt_volume},
    secrets=[Secret.from_name("optimizerSecret")],
)
def send_code(code: str):
    """Send and execute optimizer code in the sandbox environment."""
    filename, optimizer_code = _write_optimizer_code_to_volume(code, optimizers_volume)

    # --- Sandbox Setup ---
    sandbox_app = App.lookup(SANDBOX_APP_NAME, create_if_missing=True)
    sandbox = Sandbox.create(app=sandbox_app, image=sandbox_image, timeout=60 * 60)

    # Write code to sandbox
    with sandbox.open(filename, "w") as f:
        f.write(optimizer_code)
    with sandbox.open(filename, "rb") as f:
        print(f.read())

    time.sleep(1)
    process = sandbox.exec("python", filename)

    stdout = process.stdout.read()
    stderr = process.stderr.read()

    print(stdout)
    print("-" * 32)
    print(stderr)

    return_obj = {
        "stdout": stdout,
        "stderr": stderr,
        "code": code,
        "filename": filename,
    }

    sandbox.terminate()

    return return_obj


def main():
    """Entrypoint for local execution. Accepts code as input."""
    # Example usage: pass code as a string argument
    return_obj = send_code.remote(
        """
import torch
import math

class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)

                # Compute bias-corrected moments
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

# Test the optimizer
x = torch.tensor([0.0], requires_grad=True)
optimizer = CustomOptimizer([x], lr=0.1)

for step in range(20):
    optimizer.zero_grad()
    loss = (x - 3) ** 2
    loss.backward()
    optimizer.step()
    print(f"Step {step + 1}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"\\nOptimal x: {x.item():.4f}")
"""
    )

    print("\n===== Optimizer Execution Result =====")
    print(f"Filename: {return_obj['filename']}")
    print("--- STDOUT ---")
    print(
        return_obj["stdout"].decode()
        if isinstance(return_obj["stdout"], bytes)
        else return_obj["stdout"]
    )
    print("--- STDERR ---")
    print(
        return_obj["stderr"].decode()
        if isinstance(return_obj["stderr"], bytes)
        else return_obj["stderr"]
    )
    print("--- CODE ---")
    print(return_obj["code"])
    print("======================================\n")
