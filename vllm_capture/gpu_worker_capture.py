"""
GPU Worker with Activation Capture Support
This worker uses the GPUModelRunnerCapture for hook-based activation capture.
"""

import os
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from vllm.v1.worker.gpu_model_runner_capture import (
    GPUModelRunnerCapture, 
    ActivationCaptureConfig,
    ActivationReader
)


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
            print(f"[WorkerCapture] Activation capture ENABLED with layers: {self.capture_config.layers_to_capture}")
        
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
        
        # NOW replace the model runner with our capture-enabled one
        if self.capture_config.enabled:
            print(f"[WorkerCapture] Replacing model runner with GPUModelRunnerCapture")
            print(f"[WorkerCapture] Device: {self.device}")
            self.model_runner = GPUModelRunnerCapture(
                self.vllm_config, 
                self.device,
                capture_config=self.capture_config
            )
            print(f"[WorkerCapture] Model runner replaced successfully")
            
            # If driver, set up activation reader
            if self.is_driver and self.capture_config.enabled:
                # Access parallel_config from vllm_config
                parallel_config = self.vllm_config.parallel_config
                world_size = parallel_config.world_size if parallel_config else 1
                self.activation_reader = ActivationReader(
                    world_size=world_size,
                    buffer_size_gb=self.capture_config.buffer_size_gb
                )
            else:
                self.activation_reader = None
    
    def load_model(self) -> None:
        """Load model and ensure hooks are registered."""
        # Verify we have our capture-enabled model runner
        if not isinstance(self.model_runner, GPUModelRunnerCapture):
            print("ERROR: Model runner is not GPUModelRunnerCapture!")
            print(f"Model runner type: {type(self.model_runner)}")
        
        # Call parent's load_model - this will also register hooks via our override
        super().load_model()
        
        # Verify hooks were registered
        if self.capture_config.enabled and isinstance(self.model_runner, GPUModelRunnerCapture):
            print(f"[WorkerCapture] Model loaded and hooks should be registered")
            
    def get_captured_activations(self):
        """Retrieve captured activations from all workers."""
        if self.activation_reader:
            return self.activation_reader.read_activations()
        return {}
        
    def cleanup(self):
        """Clean up resources including activation buffers."""
        if hasattr(self, 'model_runner'):
            self.model_runner.cleanup()
        if self.activation_reader:
            self.activation_reader.cleanup()
        super().cleanup()