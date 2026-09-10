import json
import time

import numpy as np
from scipy.spatial import cKDTree

from .starmap_compression import StarMapCompressionEnv

# Instantiate the environment
env = StarMapCompressionEnv(data_path="galaxy_subset.npy", views_path="user_views.npy")

# Save initial data for comparison
initial_data = env.data.copy()

# Run RL steps and log metrics for env.py process
num_steps = 5
env.reset()
metrics = []
start_time = time.time()

for step in range(num_steps):
    print(f"Step {step + 1}")
    env.run_rl_step(timeout_seconds=60)
    # Save compressed data
    np.save(f"compressed_data_step_{step + 1}.npy", env.data)
    # Log metrics for env.py process
    state = env._get_state()
    reward = -env.evaluate()["avg_data_size"] / 1000 + 5 * len(env.data) / len(
        env.original_data
    )
    total_points = sum(
        np.sum(np.sqrt(np.sum((env.data - v) ** 2, axis=1)) < env.view_radius)
        for v in env.views
    )
    metric = {
        "step": step + 1,
        "num_points": len(env.data),
        "avg_data_size": env.evaluate()["avg_data_size"],
        "total_points_in_view": total_points,
        "reward": reward,
        "timestamp": time.time(),
    }
    metrics.append(metric)

# Save metrics as .jsonl
with open("starmap_metrics.jsonl", "w") as f:
    for metric in metrics:
        f.write(json.dumps(metric) + "\n")

# Compare initial and final data
final_data = env.data
print(f"Initial data shape: {initial_data.shape}, Final data shape: {final_data.shape}")
print(f"Data changed: {not np.array_equal(initial_data, final_data)}")
print(f"Total time: {time.time() - start_time:.2f} seconds")

# Verify view distances
data = np.load("galaxy_subset.npy")
views = np.load("user_views.npy")
tree = cKDTree(data)
distances, _ = tree.query(views)
print(f"View distances: min={distances.min()}, max={distances.max()}")
