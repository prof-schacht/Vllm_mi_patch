"""
vLLM Activation Capture System

A fork extension of vLLM that enables real-time neural activation capture
during inference without double computation.

Key Features:
- Single-pass activation capture using PyTorch hooks
- Selective layer capture with configurable compression
- Zero-copy shared memory transfer for efficiency
- Post-hoc marking of behaviorally interesting events

Author: Research Team
Date: August 2025
"""

from .gpu_model_runner_capture import (
    GPUModelRunnerCapture,
    SharedActivationBuffer,
    ActivationCaptureConfig,
    ActivationReader
)

from .gpu_worker_capture import WorkerCapture

__version__ = "1.0.0"
__all__ = [
    "GPUModelRunnerCapture",
    "SharedActivationBuffer", 
    "ActivationCaptureConfig",
    "ActivationReader",
    "WorkerCapture"
]