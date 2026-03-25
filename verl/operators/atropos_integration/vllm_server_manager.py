"""
vLLM Server Manager for Atropos integration.

Manages the lifecycle of vLLM inference servers:
- Launching servers
- Weight updates
- Health monitoring
- Cleanup
"""

import logging
import os
import signal
import socket
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port: int, timeout: float = 5.0) -> bool:
    """Kill any process using the specified port."""
    if not is_port_in_use(port):
        return True

    logger.info(f"Port {port} is in use, attempting to kill existing process...")

    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            logger.info(f"Killing {len(pids)} processes on port {port}...")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass

            start = time.time()
            while time.time() - start < timeout:
                if not is_port_in_use(port):
                    return True
                time.sleep(0.5)

            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(1)
            return not is_port_in_use(port)
    except FileNotFoundError:
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], timeout=5)
            time.sleep(1)
            return not is_port_in_use(port)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    except subprocess.TimeoutExpired:
        pass

    logger.warning(f"Could not kill process on port {port}")
    return False


class VLLMServerManager:
    """
    Manages vLLM inference server lifecycle for Atropos integration.

    This manager:
    1. Launches vLLM server on a dedicated port
    2. Provides the endpoint URL to Atropos for inference calls
    3. Handles weight updates by reloading model weights
    4. Monitors server health
    5. Cleans up on exit
    """

    def __init__(
        self,
        model_path: str,
        port: int = 8100,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        gpu_num: int = 1,
        trust_remote_code: bool = True,
    ):
        """
        Initialize vLLM server manager.

        Args:
            model_path: Path to model checkpoint (HuggingFace model ID or local path)
            port: Port for vLLM server
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enforce_eager: Force eager execution mode (required for some features)
            max_model_len: Maximum sequence length
            gpu_num: Number of GPUs to use
            trust_remote_code: Trust remote code in model loading
        """
        self.model_path = model_path
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.max_model_len = max_model_len
        self.gpu_num = gpu_num
        self.trust_remote_code = trust_remote_code
        self.process: Optional[subprocess.Popen] = None
        self._健康检查_count = 0

    @property
    def endpoint(self) -> str:
        """Get the vLLM server endpoint URL."""
        return f"http://localhost:{self.port}/v1"

    @property
    def generate_endpoint(self) -> str:
        """Get the /generate endpoint for Atropos to call."""
        return f"http://localhost:{self.port}/generate"

    def is_running(self) -> bool:
        """Check if the vLLM server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def _build_command(self) -> list:
        """Build the vLLM server launch command."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.gpu_num > 1:
            cmd.extend(["--gpu-num", str(self.gpu_num)])
        return cmd

    def start(self, timeout: float = 120.0) -> bool:
        """
        Launch the vLLM server.

        Args:
            timeout: Maximum time to wait for server to be ready

        Returns:
            True if server started successfully
        """
        if is_port_in_use(self.port):
            logger.info(f"Port {self.port} already in use, attempting cleanup...")
            if not kill_process_on_port(self.port):
                logger.error(f"Could not free port {self.port}")
                return False

        logger.info(f"Starting vLLM server: {' '.join(self._build_command())}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            self._build_command(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )

        # Wait for server to be ready
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"vLLM server died. STDOUT: {stdout.decode()[:500]}")
                logger.error(f"STDERR: {stderr.decode()[:500]}")
                return False

            if is_port_in_use(self.port):
                # Give it a moment to fully initialize
                time.sleep(2)
                if is_port_in_use(self.port):
                    logger.info(f"vLLM server started on port {self.port}")
                    return True

            time.sleep(1)

        logger.error(f"vLLM server failed to start within {timeout}s")
        self.stop()
        return False

    def stop(self):
        """Stop the vLLM server."""
        if self.process is None:
            return

        logger.info("Stopping vLLM server...")
        try:
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate gracefully, killing...")
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            else:
                self.process.kill()
            self.process.wait()
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")
        finally:
            self.process = None

    def restart(self, timeout: float = 120.0) -> bool:
        """
        Restart the vLLM server.

        Useful for权重 updates that require server restart.

        Returns:
            True if restart successful
        """
        logger.info("Restarting vLLM server...")
        self.stop()
        time.sleep(2)
        return self.start(timeout=timeout)

    def check_health(self) -> bool:
        """Check if vLLM server is healthy by making a simple request."""
        try:
            import requests
            response = requests.get(
                f"http://localhost:{self.port}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def update_weights(self, model_path: str = None) -> bool:
        """
        Update the model weights in the vLLM server.

        This typically requires a server restart for vLLM.

        Args:
            model_path: New model path (uses original if None)

        Returns:
            True if update successful
        """
        if model_path:
            self.model_path = model_path
        return self.restart()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
