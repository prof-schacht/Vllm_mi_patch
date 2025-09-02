"""
Compatibility shim for different vLLM versions.

Handles import differences between vLLM versions, particularly the 
VLLMConfig -> VllmConfig naming change.
"""

# Handle VllmConfig vs VLLMConfig naming across versions
try:
    from vllm.config import VllmConfig
except ImportError:
    try:
        from vllm.config import VLLMConfig as VllmConfig
    except ImportError:
        # Fallback for even older versions
        VllmConfig = None

# Export the working import
__all__ = ['VllmConfig']