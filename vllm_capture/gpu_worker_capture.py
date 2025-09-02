"""
GPU Worker with Activation Capture Support (hook-free)

Uses GPUModelRunnerCorrect, which integrates activation capture at the
runner level via intermediate_tensors. No PyTorch hooks are required.
"""

import os
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from .v1.worker.gpu_model_runner_correct import GPUModelRunnerCorrect
from .activations.capture import ActivationCaptureConfig


class WorkerCapture(BaseWorker):
    """
    Extended GPU worker that supports activation capture during inference.
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        capture_config: Optional[ActivationCaptureConfig] = None,
    ):
        # Store capture config before parent init
        # Read from environment if not provided
        self.capture_config = capture_config or ActivationCaptureConfig.from_env()
        
        if self.capture_config.enabled:
            print(f"[WorkerCapture] Activation capture ENABLED (hook-free)")
        
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker
        )
        
        # Store is_driver for later use
        self.is_driver = is_driver_worker
        
    def init_device(self):
        """Initialize device and replace model runner with capture-enabled version."""
        # Call parent's init_device which sets up device and creates model runner
        super().init_device()
        
        # NOW replace the model runner with our corrected hook-free runner
        if self.capture_config.enabled:
            print(f"[WorkerCapture] Replacing model runner with GPUModelRunnerCorrect")
            print(f"[WorkerCapture] Device: {self.device}")
            self.model_runner = GPUModelRunnerCorrect(
                self.vllm_config,
                self.device,
            )
            print(f"[WorkerCapture] Model runner replaced successfully")
            self.activation_reader = None
    
    def load_model(self) -> None:
        """Load model and ensure hooks are registered."""
        # Verify we have our capture-enabled model runner
        if not isinstance(self.model_runner, GPUModelRunnerCorrect):
            print("ERROR: Model runner is not GPUModelRunnerCorrect!")
            print(f"Model runner type: {type(self.model_runner)}")
        
        # Call parent's load_model - this will also register hooks via our override
        super().load_model()
        
        # Verify hooks were registered
        if self.capture_config.enabled and isinstance(self.model_runner, GPUModelRunnerCorrect):
            print(f"[WorkerCapture] Model loaded with hook-free capture (intermediate_tensors)")
            
    def get_last_activation_manifest(self):
        """Retrieve activation manifest from the runner for the last completed request."""
        if hasattr(self, 'model_runner') and isinstance(self.model_runner, GPUModelRunnerCorrect):
            try:
                return self.model_runner.get_activation_manifest()
            except Exception:
                return None
        return None
        
    def cleanup(self):
        """Clean up resources including activation buffers."""
        if hasattr(self, 'model_runner'):
            try:
                self.model_runner.cleanup()
            except Exception:
                pass
        if getattr(self, 'activation_reader', None):
            try:
                self.activation_reader.cleanup()
            except Exception:
                pass
        super().cleanup()
