import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# File paths
BASE_PATH = "."
ORIGINAL_DATA = os.path.join(BASE_PATH, "galaxy_subset.npy")
VIEWS = os.path.join(BASE_PATH, "user_views.npy")
COMPRESSED_FILES = [
    os.path.join(BASE_PATH, f"compressed_data_step_{i}.npy") for i in range(1, 6)
]  # Steps 1–5

# Load data
original_data = np.load(ORIGINAL_DATA)
views = np.load(VIEWS)
compressed_data = []
for f in COMPRESSED_FILES:
    if os.path.exists(f):
        compressed_data.append(np.load(f))
    else:
        print(f"Warning: {f} not found")

if not compressed_data:
    raise FileNotFoundError("No compressed data files found")

# 1. Static 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot original data (subsampled for clarity)
subsample = np.random.choice(len(original_data), 200, replace=False)
ax.scatter(
    original_data[subsample, 0],
    original_data[subsample, 1],
    original_data[subsample, 2],
    c="gray",
    alpha=0.3,
    label="Gaia Subset (1000 points)",
    s=10,
)

# Plot initial and final compressed data
ax.scatter(
    compressed_data[0][:, 0],
    compressed_data[0][:, 1],
    compressed_data[0][:, 2],
    c="blue",
    label=f"Initial RL (Step 1, {len(compressed_data[0])} points)",
    s=50,
)
ax.scatter(
    compressed_data[-1][:, 0],
    compressed_data[-1][:, 1],
    compressed_data[-1][:, 2],
    c="red",
    label=f"Final RL (Step 5, {len(compressed_data[-1])} points)",
    s=50,
)

# Plot views
ax.scatter(
    views[:, 0],
    views[:, 1],
    views[:, 2],
    c="green",
    marker="*",
    s=200,
    label="Three.js Views (10 points)",
)

# Labels and legend
ax.set_xlabel("X (arbitrary units)")
ax.set_ylabel("Y (arbitrary units)")
ax.set_zlabel("Z (arbitrary units)")
ax.set_title("StarMapCompression: RL-Driven Compression of Gaia Data")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "starmap_compression_static.png"), dpi=300)
plt.show()

# 2. Animation of Compression Progression
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")


def update(frame):
    ax.clear()
    # Plot original data (subsampled)
    ax.scatter(
        original_data[subsample, 0],
        original_data[subsample, 1],
        original_data[subsample, 2],
        c="gray",
        alpha=0.3,
        label="Gaia Subset (1000 points)",
        s=10,
    )
    # Plot compressed data for current step
    data = compressed_data[frame]
    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c="blue",
        label=f"Step {frame+1} ({len(data)} points)",
        s=50,
    )
    # Plot views
    ax.scatter(
        views[:, 0],
        views[:, 1],
        views[:, 2],
        c="green",
        marker="*",
        s=200,
        label="Three.js Views (10 points)",
    )
    # Labels and legend
    ax.set_xlabel("X (arbitrary units)")
    ax.set_ylabel("Y (arbitrary units)")
    ax.set_zlabel("Z (arbitrary units)")
    ax.set_title("StarMapCompression: RL Compression Progression")
    ax.legend()


# Create animation
ani = FuncAnimation(
    fig, update, frames=len(compressed_data), interval=1000, repeat=True
)
ani.save(
    os.path.join(BASE_PATH, "starmap_compression_animation.gif"), writer="pillow", fps=1
)
plt.show()
