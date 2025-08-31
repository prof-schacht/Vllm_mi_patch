"""
Extended GPU Model Runner with Activation Capture via Hooks
This is the core of our vLLM fork - capturing activations during the actual inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
from dataclasses import dataclass
import time
import threading
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class ActivationCaptureConfig:
    """Configuration for activation capture."""
    enabled: bool = False
    layers_to_capture: List[int] = None  # None = all layers
    capture_attention: bool = True
    capture_mlp: bool = True
    capture_residual: bool = False
    compression_k: Optional[int] = 256  # SVD compression dimension
    buffer_size_gb: float = 2.0  # Per-GPU buffer size
    suspicion_threshold: float = 0.7
    random_sample_rate: float = 0.01
    event_triggered: bool = True  # Only capture on behavioral events
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        import os
        config = cls()
        
        # Check if capture is enabled
        if os.environ.get("VLLM_CAPTURE_ENABLED", "0") == "1":
            config.enabled = True
            
        # Parse layers to capture
        if "VLLM_CAPTURE_LAYERS" in os.environ:
            layers_str = os.environ["VLLM_CAPTURE_LAYERS"]
            if layers_str.lower() == "all":
                config.layers_to_capture = None
            else:
                config.layers_to_capture = [int(x) for x in layers_str.split(",")]
        
        # Parse compression
        if "VLLM_CAPTURE_COMPRESSION_K" in os.environ:
            try:
                config.compression_k = int(os.environ["VLLM_CAPTURE_COMPRESSION_K"])
            except ValueError:
                config.compression_k = None  # No compression if invalid
            
        # Parse buffer size
        if "VLLM_CAPTURE_BUFFER_SIZE_GB" in os.environ:
            config.buffer_size_gb = float(os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"])
            
        # Parse sample rate
        if "VLLM_CAPTURE_SAMPLE_RATE" in os.environ:
            config.random_sample_rate = float(os.environ["VLLM_CAPTURE_SAMPLE_RATE"])
            config.event_triggered = False  # Disable event triggering if sampling
            
        return config
    

class SharedActivationBuffer:
    """
    Shared memory buffer for zero-copy activation transfer from worker to main process.
    This avoids serialization overhead when transferring large tensors.
    """
    
    def __init__(self, name: str, size_gb: float = 2.0, mode: str = 'write'):
        self.name = name
        self.size_bytes = int(size_gb * 1024**3)
        self.mode = mode
        
        if mode == 'write':
            # Try to clean up existing buffer first
            try:
                old_shm = shared_memory.SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()
                print(f"[SharedActivationBuffer] Cleaned up existing buffer: {name}")
            except FileNotFoundError:
                pass  # No existing buffer to clean up
            
            # Create shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=self.size_bytes, name=name)
            self.buffer = np.ndarray((self.size_bytes,), dtype=np.uint8, buffer=self.shm.buf)
            self.write_pos = 0
            self.metadata_queue = mp.Queue()
            self.metadata = []  # Track written metadata for easy access
        else:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name)
            self.buffer = np.ndarray((self.size_bytes,), dtype=np.uint8, buffer=self.shm.buf)
            self.metadata = []  # Also initialize for read mode
            # Note: In read mode, metadata must be populated by reading from queue or other source
            
    def write_tensor(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bool:
        """Write tensor to shared memory with metadata."""
        if self.mode != 'write':
            raise ValueError("Buffer is in read mode")
            
        # Convert to numpy for zero-copy transfer
        tensor_np = tensor.detach().cpu().numpy()
        tensor_bytes = tensor_np.tobytes()
        tensor_size = len(tensor_bytes)
        
        # Check if there's space
        if self.write_pos + tensor_size > self.size_bytes:
            return False  # Buffer full
        
        # Write tensor data
        self.buffer[self.write_pos:self.write_pos + tensor_size] = np.frombuffer(tensor_bytes, dtype=np.uint8)
        
        # Add metadata with location info
        metadata.update({
            'offset': self.write_pos,
            'size': tensor_size,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'timestamp': time.time()
        })
        self.metadata_queue.put(metadata)
        self.metadata.append(metadata)  # Also track in list
        
        self.write_pos += tensor_size
        return True
    
    def read_tensor(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Read tensor from shared memory using metadata."""
        if self.mode != 'read':
            raise ValueError("Buffer is in write mode")
            
        offset = metadata['offset']
        size = metadata['size']
        shape = metadata['shape']
        dtype = getattr(torch, metadata['dtype'].split('.')[-1])
        
        # Read bytes and reconstruct tensor
        tensor_bytes = self.buffer[offset:offset + size]
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        return torch.from_numpy(tensor_np).to(dtype)
    
    def cleanup(self):
        """Clean up shared memory."""
        self.shm.close()
        if self.mode == 'write':
            self.shm.unlink()


class GPUModelRunnerCapture(GPUModelRunner):
    """
    Extended GPU Model Runner that captures activations during forward pass.
    
    Key innovation: Hooks fire during the ACTUAL inference that generates outputs.
    No double inference, no hoping for identical results - we catch the model red-handed.
    """
    
    def __init__(self, vllm_config, device, capture_config: Optional[ActivationCaptureConfig] = None):
        # Call parent with proper parameters
        super().__init__(vllm_config, device)
        
        self.capture_config = capture_config or ActivationCaptureConfig()
        self.hooks = []
        self.captured_activations = {}
        
        # Get rank from parallel config (default to 0 for single GPU)
        rank = getattr(self.parallel_config, 'rank', 0) if self.parallel_config else 0
        
        # Shared memory buffer for zero-copy transfer
        if self.capture_config.enabled:
            buffer_name = f"vllm_capture_rank_{rank}"
            self.activation_buffer = SharedActivationBuffer(
                name=buffer_name,
                size_gb=self.capture_config.buffer_size_gb,
                mode='write'
            )
            
            # Don't register hooks here - model doesn't exist yet!
            # Will register in load_model method
            
            # Track which requests should be captured
            self.capture_requests = set()
    
    def load_model(self, eep_scale_up: bool = False) -> None:
        """Load model and register activation hooks."""
        # Call parent's load_model
        super().load_model(eep_scale_up=eep_scale_up)
        
        # Now register hooks after model is loaded
        if self.capture_config.enabled:
            print(f"[GPUModelRunnerCapture] Model loaded, registering hooks...")
            self._register_activation_hooks()
            print(f"[GPUModelRunnerCapture] Hooks registered successfully")
            
    def _register_activation_hooks(self):
        """Register forward hooks on model layers for activation capture."""
        if not self.capture_config.enabled:
            return
            
        # Clear any existing hooks
        self._remove_hooks()
        
        # Find model layers
        model = self.model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama, Qwen, Mistral style
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT style
            layers = model.transformer.h
        else:
            print(f"Warning: Could not find layers in model architecture")
            return
            
        # Determine which layers to capture
        if self.capture_config.layers_to_capture:
            layers_to_hook = self.capture_config.layers_to_capture
        else:
            # Capture all layers
            layers_to_hook = list(range(len(layers)))
            
        print(f"Registering hooks on {len(layers_to_hook)} layers")
        
        for layer_idx in layers_to_hook:
            if layer_idx >= len(layers):
                continue
                
            layer = layers[layer_idx]
            
            # Create hook function
            def make_hook(idx):
                def hook(module, input, output):
                    # Only capture if we're in a capture-enabled request
                    if not self._should_capture_current():
                        return
                    
                    # Handle different output types
                    if isinstance(output, tuple):
                        activation = output[0]  # Usually hidden states
                    else:
                        activation = output
                        
                    # Compress on GPU before transfer
                    compressed = self._compress_activation_gpu(activation)
                    
                    # Store in captured activations
                    self.captured_activations[f"layer_{idx}"] = compressed
                    
                return hook
            
            # Register the hook
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)
            
    def _compress_activation_gpu(self, activation: torch.Tensor) -> torch.Tensor:
        """Compress activation on GPU to reduce memory transfer."""
        if not self.capture_config.compression_k or self.capture_config.compression_k == 0:
            # No compression - return full tensor
            # Still detach from graph to avoid memory issues
            return activation.detach().clone()
            
        # Only keep last token for efficiency (most relevant for generation)
        if activation.dim() == 3 and activation.shape[1] > 1:
            activation = activation[:, -1:, :]  # Keep last token only
            
        # Flatten batch and sequence dimensions
        original_shape = activation.shape
        if activation.dim() == 3:
            batch, seq, hidden = activation.shape
            activation = activation.view(batch * seq, hidden)
            
        # Apply SVD compression on GPU
        try:
            U, S, V = torch.svd_lowrank(activation, q=self.capture_config.compression_k)
            compressed = {
                'U': U,
                'S': S, 
                'V': V,
                'shape': original_shape
            }
        except:
            # Fallback if SVD fails
            compressed = activation
            
        return compressed
    
    def _should_capture_current(self) -> bool:
        """Determine if current forward pass should capture activations."""
        # Check if we're in a request marked for capture
        # This will be set based on behavioral events or random sampling
        return len(self.capture_requests) > 0
    
    def _should_capture_request(self, req_id: str, metadata: Dict[str, Any]) -> bool:
        """Determine if a specific request should have activations captured."""
        if not self.capture_config.enabled:
            return False
            
        # Always capture if explicitly marked
        if metadata.get('capture_activations', False):
            return True
            
        # Event-triggered capture
        if self.capture_config.event_triggered:
            suspicion_score = metadata.get('suspicion_score', 0)
            if suspicion_score > self.capture_config.suspicion_threshold:
                return True
                
        # Random sampling
        if np.random.random() < self.capture_config.random_sample_rate:
            return True
            
        return False
    
    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """
        Execute model with activation capture.
        
        This is the key method - hooks fire during THIS forward pass,
        capturing activations from the exact inference that generates outputs.
        """
        # Determine which requests should be captured
        self.capture_requests.clear()
        
        # v1 uses scheduled_new_reqs
        if hasattr(scheduler_output, 'scheduled_new_reqs'):
            for req_data in scheduler_output.scheduled_new_reqs:
                req_id = req_data.req_id
                # For now, capture all requests (no metadata available)
                if self._should_capture_request(req_id, {}):
                    self.capture_requests.add(req_id)
        elif hasattr(scheduler_output, 'scheduled_req_ids'):
            # Legacy path if API changes
            for req_id in scheduler_output.scheduled_req_ids:
                req_metadata = scheduler_output.req_metadata.get(req_id, {})
                if self._should_capture_request(req_id, req_metadata):
                    self.capture_requests.add(req_id)
                
        # Clear previous captures
        self.captured_activations.clear()
        
        # Run normal model execution - hooks will fire automatically
        output = super().execute_model(scheduler_output, intermediate_tensors)
        
        # If we captured activations, transfer them to shared memory
        if self.captured_activations and self.activation_buffer:
            self._transfer_activations_to_buffer(scheduler_output)
            
        return output
    
    def _transfer_activations_to_buffer(self, scheduler_output):
        """Transfer captured activations to shared memory for main process."""
        for req_id in self.capture_requests:
            for layer_name, activation in self.captured_activations.items():
                # Prepare metadata
                metadata = {
                    'request_id': req_id,
                    'layer': layer_name,
                    'timestep': getattr(scheduler_output, 'timestep', 0),  # May not exist in v1
                    'rank': getattr(self.parallel_config, 'rank', 0) if self.parallel_config else 0,
                }
                
                # Handle compressed activations
                if isinstance(activation, dict) and 'U' in activation:
                    # Store compressed components
                    for component in ['U', 'S', 'V']:
                        component_metadata = metadata.copy()
                        component_metadata['component'] = component
                        self.activation_buffer.write_tensor(
                            activation[component],
                            component_metadata
                        )
                else:
                    # Store regular tensor
                    self.activation_buffer.write_tensor(activation, metadata)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def cleanup(self):
        """Clean up resources."""
        self._remove_hooks()
        if hasattr(self, 'activation_buffer'):
            self.activation_buffer.cleanup()


# Extension point for the main engine to read activations
class ActivationReader:
    """
    Main process reader that retrieves activations from worker shared memory.
    """
    
    def __init__(self, world_size: int, buffer_size_gb: float = 2.0):
        self.world_size = world_size
        self.buffer_size_gb = buffer_size_gb
        self.buffers = {}
        # Don't connect to buffers yet - they don't exist until model runner is created
        # Will connect lazily on first read attempt
                
        # Storage for retrieved activations
        self.activation_store = {}
        
    def read_activations(self) -> Dict[str, Any]:
        """Read available activations from all workers."""
        # Lazy connect to buffers if not already connected
        if not self.buffers:
            for rank in range(self.world_size):
                buffer_name = f"vllm_capture_rank_{rank}"
                try:
                    self.buffers[rank] = SharedActivationBuffer(
                        name=buffer_name,
                        size_gb=self.buffer_size_gb,
                        mode='read'
                    )
                    print(f"Connected to buffer for rank {rank}")
                except:
                    print(f"Warning: Could not connect to buffer for rank {rank}")
        
        activations = {}
        
        for rank, buffer in self.buffers.items():
            # Read metadata queue
            while not buffer.metadata_queue.empty():
                metadata = buffer.metadata_queue.get()
                tensor = buffer.read_tensor(metadata)
                
                # Organize by request ID
                req_id = metadata['request_id']
                if req_id not in activations:
                    activations[req_id] = {}
                    
                layer_name = metadata['layer']
                activations[req_id][layer_name] = {
                    'tensor': tensor,
                    'metadata': metadata
                }
                
        return activations
    
    def cleanup(self):
        """Clean up resources."""
        for buffer in self.buffers.values():
            buffer.cleanup()