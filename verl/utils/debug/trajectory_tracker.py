# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trajectory tracker for saving intermediate results to HDFS.

The tracker can be inserted into training code to persist tensors for
offline comparison. Each process communicates with a Ray actor that
serializes data and uploads it to HDFS.
"""

import io
import os
import tempfile
from collections import deque

import ray
import torch

from verl.utils.hdfs_io import copy, makedirs

remote_copy = ray.remote(copy)


@ray.remote
def save_to_hdfs(data: io.BytesIO, name, hdfs_dir, verbose):
    """Save a serialized tensor buffer to HDFS.

    This is a Ray remote function that writes the buffer to a local temp file
    and then copies it to the specified HDFS directory.

    Args:
        data: A BytesIO buffer containing the serialized tensor data.
        name: Base name for the output file (without extension).
        hdfs_dir: Target HDFS directory path.
        verbose: Whether to print progress information.

    """
    filename = name + ".pth"
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_filepath = os.path.join(tmpdirname, filename)
        with open(local_filepath, "wb") as f:
            f.write(data.getbuffer())
        # upload to hdfs

        if verbose:
            print(f"Saving {local_filepath} to {hdfs_dir}")
        try:
            copy(local_filepath, hdfs_dir)
        except Exception as e:
            print(e)


@ray.remote
class TrajectoryTracker:
    """Ray actor that manages asynchronous tensor uploads to HDFS."""

    def __init__(self, hdfs_dir, verbose) -> None:
        """Initialize the tracker.

        Args:
            hdfs_dir: HDFS directory where tensors will be stored.
            verbose: Whether to print upload progress.

        """
        self.hdfs_dir = hdfs_dir
        makedirs(hdfs_dir)
        self.verbose = verbose

        self.handle = deque()

    def dump(self, data: io.BytesIO, name):
        """Submit a buffer for asynchronous upload to HDFS.

        Args:
            data: A BytesIO buffer containing serialized tensor data.
            name: Base name for the output file.

        """
        # get a temp file and write to it
        self.handle.append(save_to_hdfs.remote(data, name, self.hdfs_dir, self.verbose))

    def wait_for_hdfs(self):
        """Block until all pending HDFS uploads have completed."""
        while len(self.handle) != 0:
            future = self.handle.popleft()
            ray.get(future)


def dump_data(data, name):
    """Serialize and upload data to HDFS via the global tracker.

    This is a no-op unless the ``VERL_ENABLE_TRACKER`` environment variable
    is set to ``"1"``.

    Args:
        data: Arbitrary data serializable by ``torch.save``.
        name: Identifier used as the HDFS filename.

    """
    enable = os.getenv("VERL_ENABLE_TRACKER", "0") == "1"
    if not enable:
        return
    buffer = io.BytesIO()
    torch.save(data, buffer)
    tracker = get_trajectory_tracker()
    ray.get(tracker.dump.remote(buffer, name))


def get_trajectory_tracker():
    """Get or create the global TrajectoryTracker Ray actor.

    The actor is configured via environment variables:

    - ``VERL_TRACKER_HDFS_DIR``: Target HDFS directory (required).
    - ``VERL_TRACKER_VERBOSE``: Set to ``"1"`` for verbose output.

    Returns:
        TrajectoryTracker: A handle to the named Ray actor.

    """
    hdfs_dir = os.getenv("VERL_TRACKER_HDFS_DIR", default=None)
    verbose = os.getenv("VERL_TRACKER_VERBOSE", default="0") == "1"
    assert hdfs_dir is not None
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(
        hdfs_dir, verbose
    )
    return tracker


if __name__ == "__main__":
    # testing
    os.environ["VERL_ENABLE_TRACKER"] = "1"
    os.environ["VERL_TRACKER_HDFS_DIR"] = "~/debug/test"

    @ray.remote
    def process(iter):
        """Simulate a training process that dumps random tensor data."""
        data = {"obs": torch.randn(10, 20)}
        dump_data(data, f"process_{iter}_obs")

    ray.init()

    output_lst = []

    for i in range(10):
        output_lst.append(process.remote(i))

    out = ray.get(output_lst)

    tracker = get_trajectory_tracker()
    ray.get(tracker.wait_for_hdfs.remote())
