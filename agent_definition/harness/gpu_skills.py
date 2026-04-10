"""
gpu_skills.py

GPU connection skills for reproducibility agents.
Each skill connects to a remote GPU over SSH, runs a command, and returns output.
"""

import subprocess
from abc import ABC, abstractmethod


class GPUSkill(ABC):
    """Base class for GPU execution skills."""

    @abstractmethod
    def run_command(self, command: str) -> str:
        """Run a shell command on the remote GPU and return stdout + stderr."""

    def _ssh_run(self, ssh_args: list[str], command: str) -> str:
        result = subprocess.run(
            ssh_args + [command],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return (result.stdout + result.stderr).strip()


class ServerlessGPUSkill(GPUSkill):
    """
    Connects to the FPT Cloud serverless GPU.

    SSH command:
        ssh root@tcp-endpoint.serverless.fptcloud.com -p 34919 \
            -i ~/.ssh/id_rsa -o ProxyJump=none -o StrictHostKeyChecking=no
    """

    def __init__(
        self,
        host: str = "tcp-endpoint.serverless.fptcloud.com",
        port: int = 34919,
        key_path: str = "~/.ssh/id_rsa",
    ):
        self.host = host
        self.port = port
        self.key_path = key_path

    def run_command(self, command: str) -> str:
        return self._ssh_run([
            "ssh", f"root@{self.host}",
            "-p", str(self.port),
            "-i", self.key_path,
            "-o", "ProxyJump=none",
            "-o", "StrictHostKeyChecking=no",
        ], command)


class GPUSandboxSkill(GPUSkill):
    """
    Connects to the McGill-NLP shared GPU sandbox (8x RTX A6000, 384GB VRAM).

    SSH command:
        ssh -p 2222 USERNAME@ec2-35-182-158-243.ca-central-1.compute.amazonaws.com

    Setup:
        1. Upload your SSH public key at https://gpu-sandbox-keys-upload.mcgill-nlp.org
           (site password: O8zF1-BP27u7E9Ut)
        2. Wait for approval, then connect with your approved username.

    Notes:
        - Persistent storage at /data — use this for checkpoints and results.
        - Home dirs may not persist across restarts.
        - Check nvidia-smi before launching large jobs.
    """

    def __init__(
        self,
        username: str,
        host: str = "ec2-35-182-158-243.ca-central-1.compute.amazonaws.com",
        port: int = 2222,
        key_path: str = "~/.ssh/id_ed25519",
    ):
        self.username = username
        self.host = host
        self.port = port
        self.key_path = key_path

    def run_command(self, command: str) -> str:
        return self._ssh_run([
            "ssh",
            "-p", str(self.port),
            "-i", self.key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.username}@{self.host}",
        ], command)
