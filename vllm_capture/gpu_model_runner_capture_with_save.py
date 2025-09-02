"""
Extended GPU Model Runner that saves activations to disk in real-time.
This is a production example showing how to actually save captured activations.
"""

import os
import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Import the base capture class
from .gpu_model_runner_capture import GPUModelRunnerCapture, ActivationCaptureConfig


class GPUModelRunnerCaptureWithSave(GPUModelRunnerCapture):
    """
    Extended capture class that saves activations to disk during generation.
    This shows the production way to extract and save real activations.
    """
    
    def __init__(self, vllm_config, device, capture_config: Optional[ActivationCaptureConfig] = None):
        super().__init__(vllm_config, device, capture_config)
        
        # Set up save directory
        self.save_dir = Path(os.environ.get("VLLM_CAPTURE_SAVE_DIR", "/tmp/vllm_activations"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory with timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.inference_count = 0
        
        print(f"[GPUModelRunnerCaptureWithSave] Saving activations to: {self.session_dir}")
        
    def _transfer_activations_to_buffer(self, scheduler_output):
        """
        Override to save activations to disk instead of just buffer.
        This is called after execute_model when activations are captured.
        """
        # First call parent to maintain buffer functionality
        super()._transfer_activations_to_buffer(scheduler_output)
        
        # Now save the captured activations to disk
        if self.captured_activations:
            self._save_activations_to_disk()
            
    def _save_activations_to_disk(self):
        """
        Save the currently captured activations to disk.
        This is the REAL extraction - not simulated data!
        """
        if not self.captured_activations:
            return
            
        # Create file for this inference
        inference_file = self.session_dir / f"inference_{self.inference_count:06d}.pt"
        
        # Prepare data to save
        save_data = {
            'inference_id': self.inference_count,
            'timestamp': datetime.now().isoformat(),
            'layers_captured': list(self.layers_to_capture) if self.layers_to_capture else "all",
            'compression': self.capture_config.compression_k if self.capture_config.compression_k else None,
            'activations': {},
            'metadata': {}
        }
        
        # Save each layer's activation
        for layer_name, activation_tensor in self.captured_activations.items():
            # Move to CPU for saving
            cpu_tensor = activation_tensor.cpu()
            
            # Store the real tensor
            save_data['activations'][layer_name] = cpu_tensor
            
            # Add metadata about this activation
            save_data['metadata'][layer_name] = {
                'shape': list(cpu_tensor.shape),
                'dtype': str(cpu_tensor.dtype),
                'device': str(activation_tensor.device),
                'mean': float(cpu_tensor.mean().item()),
                'std': float(cpu_tensor.std().item()),
                'min': float(cpu_tensor.min().item()),
                'max': float(cpu_tensor.max().item()),
                'size_bytes': cpu_tensor.numel() * cpu_tensor.element_size()
            }
        
        # Save to disk
        torch.save(save_data, inference_file)
        
        # Also save a summary JSON for easy inspection
        import json
        summary_file = self.session_dir / f"inference_{self.inference_count:06d}_summary.json"
        summary_data = {
            'inference_id': self.inference_count,
            'timestamp': save_data['timestamp'],
            'layers_captured': save_data['layers_captured'],
            'compression': save_data['compression'],
            'layers': save_data['metadata']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"[GPUModelRunnerCaptureWithSave] Saved inference {self.inference_count}:")
        print(f"  File: {inference_file}")
        print(f"  Layers: {len(self.captured_activations)}")
        total_size = sum(m['size_bytes'] for m in save_data['metadata'].values())
        print(f"  Total size: {total_size / (1024**2):.2f} MB")
        
        self.inference_count += 1
        
    def cleanup(self):
        """Clean up and save session summary."""
        # Save session summary
        summary_file = self.session_dir / "session_summary.json"
        import json
        
        summary = {
            'session_id': self.session_id,
            'session_dir': str(self.session_dir),
            'total_inferences': self.inference_count,
            'capture_config': {
                'enabled': self.capture_config.enabled,
                'layers': list(self.layers_to_capture) if self.layers_to_capture else "all",
                'compression_k': self.capture_config.compression_k,
                'buffer_size_gb': self.capture_config.buffer_size_gb,
                'sample_rate': self.capture_config.sample_rate
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"[GPUModelRunnerCaptureWithSave] Session complete:")
        print(f"  Total inferences: {self.inference_count}")
        print(f"  Saved to: {self.session_dir}")
        
        # Call parent cleanup
        super().cleanup()


def load_saved_activations(session_dir: Path, inference_id: int) -> Dict[str, Any]:
    """
    Load previously saved activations from disk.
    
    Args:
        session_dir: Directory containing saved activations
        inference_id: ID of the inference to load
        
    Returns:
        Dictionary containing activations and metadata
    """
    inference_file = session_dir / f"inference_{inference_id:06d}.pt"
    
    if not inference_file.exists():
        raise FileNotFoundError(f"Inference file not found: {inference_file}")
        
    data = torch.load(inference_file)
    
    print(f"Loaded inference {inference_id}:")
    print(f"  Timestamp: {data['timestamp']}")
    print(f"  Layers: {data['layers_captured']}")
    print(f"  Activations: {list(data['activations'].keys())}")
    
    return data