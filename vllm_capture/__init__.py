"""
vLLM Activation Capture System

Hook-free activation capture designed to work with vLLM’s runner layer
using an intermediate_tensors integration. Captures ALL tokens (prefill
and decode) and supports fast, practical compression modes.

Key Properties:
- No PyTorch forward hooks (CUDA graphs can remain enabled)
- All tokens captured via runner-level integration
- Compression modes: random projection + uint8, full uint8, or top-k sparse

Author: Research Team
Date: September 2025
"""

from .activations.capture import (
    ActivationCaptureConfig,
    ActMode,
    ReturnMode,
    ActivationCollector,
)

__version__ = "1.1.0"

# Note: vLLM-dependent runner/worker classes live under package paths
#   vllm_capture.v1.worker.gpu_model_runner_correct
#   vllm_capture.gpu_worker_capture
# Import them directly from those modules when vLLM is available.

__all__ = [
    "ActivationCaptureConfig",
    "ActMode",
    "ReturnMode",
    "ActivationCollector",
]
