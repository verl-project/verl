import logging
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from atroposlib.envs.base import BaseEnv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


class StarMapCompressionEnv(BaseEnv):
    def __init__(self, data_path, views_path):
        logging.info("Initializing StarMapCompressionEnv")
        self.original_data = np.load(data_path)
        self.views = np.load(views_path)
        logging.info(
            f"Original data shape: {self.original_data.shape}, Views shape: {self.views.shape}"
        )
        logging.info(
            f"Original data range: min={self.original_data.min(axis=0)}, max={self.original_data.max(axis=0)}"
        )

        # Sample points based on density
        self.sampled_data = self._density_sample(self.original_data)
        logging.info(f"Sampled data shape: {self.sampled_data.shape}")

        # Apply PCA to reduce to 2D
        self.pca_data, self.pca_views = self._apply_pca(self.sampled_data, self.views)
        logging.info(
            f"PCA data shape: {self.pca_data.shape}, PCA views shape: {self.pca_views.shape}"
        )

        # Compute adaptive density threshold
        data_range = self.original_data.max(axis=0) - self.original_data.min(axis=0)
        cell_volume = (
            np.prod(data_range[data_range > 0]) if np.any(data_range > 0) else 1.0
        )
        self.density_threshold = max(len(self.pca_data) / cell_volume * 0.001, 1e-6)
        logging.info(f"Adaptive density threshold: {self.density_threshold}")

        # Build initial octree
        self.octree_data = self._build_octree(self.pca_data)
        logging.info(f"Octree data shape: {self.octree_data.shape}")

        # Quantize the octree data
        self.quantized_data = self._quantize_data(self.octree_data)
        logging.info(f"Quantized data shape: {self.quantized_data.shape}")

        # Map quantized data back to original points
        self.data = self._map_to_original(self.quantized_data)
        logging.info(f"Final mapped data shape: {self.data.shape}")

        # Update views to use mapped data
        self.views = self._map_to_original(self.pca_views)
        # Optional: Center views around data mean (uncomment if needed)
        # data_mean = self.original_data.mean(axis=0)
        # self.views = self.views - self.views.mean(axis=0) + data_mean
        logging.info(f"Final views shape: {self.views.shape}")

        # Scale view radius and grid sizes
        valid_range = data_range[data_range > 0]
        self.view_radius = (
            min(valid_range) * 2.0 if len(valid_range) > 0 else 50.0
        )  # ~53.4 to cover min view distance
        self.partition_methods = [
            self.view_radius * 0.5,
            self.view_radius,
            self.view_radius * 1.5,
        ]
        logging.info(
            f"View radius: {self.view_radius}, Partition methods: {self.partition_methods}"
        )

        self.max_steps = 50
        self.client = OpenAI(base_url="http://localhost:9001/v1")
        self.current_method = 0

    def _density_sample(self, data, sample_fraction=0.1, radius=50.0):
        if len(data) == 0:
            return data
        tree = cKDTree(data)
        density = np.array(
            [len(tree.query_ball_point(point, radius)) for point in data]
        )
        density = density / (density.sum() + 1e-10)
        num_samples = max(1, int(len(data) * sample_fraction))
        indices = np.random.choice(
            len(data), size=num_samples, p=density, replace=False
        )
        return data[indices]

    def _apply_pca(self, data, views):
        if len(data) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        pca_views = pca.transform(views)
        pca_data_3d = np.pad(pca_data, ((0, 0), (0, 1)), mode="constant")
        pca_views_3d = np.pad(pca_views, ((0, 0), (0, 1)), mode="constant")
        return pca_data_3d, pca_views_3d

    def _build_octree(self, data, min_points=2, max_depth=5, density_threshold=None):
        if density_threshold is None:
            density_threshold = self.density_threshold

        if len(data) < min_points or max_depth <= 0:
            if len(data) > 0:
                return np.mean(data, axis=0, keepdims=True)
            return np.array([]).reshape(0, 3)

        min_coords = data.min(axis=0)
        max_coords = data.max(axis=0)
        center = (min_coords + max_coords) / 2
        extent = (max_coords - min_coords) / 2
        cell_volume = np.prod(extent[extent > 0]) if np.any(extent > 0) else 1.0
        density = len(data) / cell_volume if cell_volume > 0 else 0
        logging.info(
            f"Octree level {5-max_depth}: density={density}, threshold={density_threshold}"
        )

        # Skip density pruning for root node
        if max_depth < 5 and density < density_threshold:
            return np.array([]).reshape(0, 3)

        octants = [[] for _ in range(8)]
        for point in data:
            idx = 0
            if point[0] > center[0]:
                idx |= 1
            if point[1] > center[1]:
                idx |= 2
            if point[2] > center[2]:
                idx |= 4
            octants[idx].append(point)

        reduced_data = []
        for octant in octants:
            if len(octant) > 0:
                octant_data = np.array(octant)
                subtree = self._build_octree(
                    octant_data, min_points, max_depth - 1, density_threshold
                )
                if len(subtree) > 0:
                    reduced_data.append(subtree)

        if not reduced_data:
            return np.array([]).reshape(0, 3)
        return np.vstack(reduced_data)

    def _quantize_data(self, data, bits=8):
        if len(data) == 0:
            return data
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        scale = (2**bits - 1) / (data_max - data_min + 1e-10)
        quantized = np.round((data - data_min) * scale).astype(np.uint8)
        dequantized = (quantized / scale) + data_min
        return dequantized

    def _map_to_original(self, simplified_data):
        if len(simplified_data) == 0:
            return simplified_data
        tree = cKDTree(self.original_data)
        distances, indices = tree.query(simplified_data, k=1)
        return self.original_data[indices]

    def reset(self):
        self.step_count = 0
        self.current_method = 0
        return self._get_state()

    def step(self, action):
        self.step_count += 1
        self.current_method = action
        avg_data_size = self._evaluate_partition()
        # Update data based on grid size
        self.data = self._recompress_data(self.partition_methods[action])
        # Reward balances size, retention, points in view, and quality
        total_points = sum(
            np.sum(np.sqrt(np.sum((self.data - v) ** 2, axis=1)) < self.view_radius)
            for v in self.views
        )
        quality = (
            np.mean([np.min(np.sum((self.data - v) ** 2, axis=1)) for v in self.views])
            if len(self.data) > 0
            else 0
        )
        reward = (
            -avg_data_size / 1000
            + 5 * len(self.data) / len(self.original_data)
            + total_points / len(self.original_data)
            - quality / 1e6
        )
        done = self.step_count >= self.max_steps
        logging.info(
            f"Step reward: data_size={avg_data_size}, points={len(self.data)}, "
            f"total_points={total_points}, quality={quality}, reward={reward}"
        )
        return self._get_state(), reward, done, {}

    def _evaluate_partition(self):
        cell_size = self.partition_methods[self.current_method]
        total_size = 0
        total_points = 0
        for view in self.views:
            distances = np.sqrt(np.sum((self.data - view) ** 2, axis=1))
            points_in_view = np.sum(distances < self.view_radius)
            total_points += points_in_view
            data_size = points_in_view * 32
            cells_per_axis = int(np.ceil(2 * self.view_radius / max(cell_size, 1e-6)))
            num_cells = cells_per_axis**3
            cell_overhead = num_cells * 10
            total_size += data_size + cell_overhead
            logging.info(
                f"View: points_in_view={points_in_view}, num_cells={num_cells}, "
                f"cell_size={cell_size}, data_size={data_size}, cell_overhead={cell_overhead}"
            )

        avg_data_size = total_size / len(self.views) if len(self.views) > 0 else 0
        logging.info(
            f"Avg data size: {avg_data_size} bytes for grid size {cell_size}, Total points in views: {total_points}"
        )
        return avg_data_size

    def _recompress_data(self, grid_size):
        # Adjust octree max_depth, min_points, quantization bits, and sampling
        scale = self.view_radius / max(grid_size, 1e-6)
        density_threshold = self.density_threshold / scale
        max_depth = int(3 + 2 * scale)  # 3 to 5
        min_points = max(1, int(2 / scale))  # 2 for large grids, higher for small
        bits = max(4, int(6 + 2 * scale))  # 6 to 8 bits
        sample_fraction = min(1.0, 0.3 + scale * 0.3)  # 0.3 to 0.6
        logging.info(
            f"Recompress: grid_size={grid_size}, density_threshold={density_threshold}, "
            f"max_depth={max_depth}, min_points={min_points}, bits={bits}, sample_fraction={sample_fraction}"
        )
        octree_data = self._build_octree(
            self.pca_data,
            min_points=min_points,
            max_depth=max_depth,
            density_threshold=density_threshold,
        )
        quantized_data = self._quantize_data(octree_data, bits=bits)
        compressed_data = self._map_to_original(quantized_data)
        # Sample points to introduce variation
        if len(compressed_data) > 0:
            num_samples = max(1, int(len(compressed_data) * sample_fraction))
            indices = np.random.choice(
                len(compressed_data), size=num_samples, replace=False
            )
            compressed_data = compressed_data[indices]
        logging.info(
            f"Recompressed data shape for grid size {grid_size}: {compressed_data.shape}"
        )
        return compressed_data

    def _get_state(self):
        return {"method": self.current_method, "data_size": len(self.data)}

    def evaluate(self):
        avg_data_size = self._evaluate_partition()
        return {"avg_data_size": avg_data_size / 1000}

    def get_next_item(self):
        return self.data

    def run_rl_step(self, timeout_seconds=60):
        self.current_method = 0
        initial_result = self.evaluate()
        initial_avg_data_size = initial_result["avg_data_size"]
        logging.info(f"Before RL Step: avg_data_size = {initial_avg_data_size} KB")

        best_reward = float("-inf")
        best_action = np.random.choice(3)
        result_queue = queue.Queue()
        rewards = []

        def step_with_action(action):
            try:
                state, reward, done, info = self.step(action)
                result_queue.put((reward, action))
                logging.info(
                    f"Action {action} (grid size {self.partition_methods[action]}): reward={reward}"
                )
            except Exception as e:
                logging.error(f"Error in action {action}: {e}")
                result_queue.put((float("-inf"), action))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(step_with_action, action) for action in range(3)]
            for action, future in enumerate(futures):
                try:
                    future.result(timeout=timeout_seconds)
                    reward, chosen_action = result_queue.get()
                    rewards.append((reward, chosen_action))
                except TimeoutError:
                    logging.warning(
                        f"Action {action} timed out after {timeout_seconds} seconds"
                    )
                    rewards.append((float("-inf"), action))

        for reward, action in rewards:
            logging.info(f"Action {action} reward: {reward}")

        for reward, action in rewards:
            if reward > best_reward or (
                reward == best_reward and np.random.random() < 0.5
            ):  # Random tiebreaker
                best_reward = reward
                best_action = action

        self.step(best_action)
        result = self.evaluate()
        logging.info(
            f"After RL Step: Chose grid size {self.partition_methods[best_action]}, "
            f"avg_data_size = {result['avg_data_size']} KB, Reward: {best_reward}"
        )
