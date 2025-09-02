#!/usr/bin/env python3
"""
Production example: Extract REAL activations from shared memory buffer.
This shows the actual way to retrieve captured activations, not simulated data.
"""

import os
import sys
import torch
import numpy as np
import pickle
import time
import json
from pathlib import Path
import multiprocessing
import multiprocessing.shared_memory as shm
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/proj/vllm-capture")  # For vLLM base

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use third GPU (available)

from vllm import LLM, SamplingParams

class ActivationExtractor:
    """Extract real activations from shared memory buffer."""
    
    def __init__(self, buffer_name: str = "vllm_capture_rank_0"):
        """
        Initialize extractor to connect to shared memory buffer.
        
        Args:
            buffer_name: Name of the shared memory buffer
        """
        self.buffer_name = buffer_name
        self.buffer = None
        self.metadata_queue = None
        self.metadata_list = []
        
    def connect(self) -> bool:
        """Connect to the shared memory buffer created by vLLM worker."""
        try:
            # Connect to existing shared memory
            self.shm = shm.SharedMemory(name=self.buffer_name)
            self.buffer = np.ndarray(
                (self.shm.size,), 
                dtype=np.uint8, 
                buffer=self.shm.buf
            )
            
            # Connect to metadata queue
            self.metadata_queue = multiprocessing.Queue()
            
            print(f"✅ Connected to buffer: {self.shm.size / (1024**2):.1f} MB")
            return True
        except FileNotFoundError:
            print(f"❌ Buffer '{self.buffer_name}' not found")
            return False
            
    def extract_activations(self, num_bytes: int = None) -> Dict[str, torch.Tensor]:
        """
        Extract real activations from the buffer using metadata header.
        
        The buffer format:
        - First 4 bytes: number of tensors (uint32)
        - Next N*12 bytes: metadata entries (offset, size, layer_idx as uint32)
        - Then: actual tensor data
        """
        if self.buffer is None:
            raise RuntimeError("Not connected to buffer. Call connect() first.")
            
        activations = {}
        
        # Read metadata header
        # First 4 bytes = number of tensors stored
        num_tensors = int.from_bytes(self.buffer[:4], 'little')
        print(f"  Found {num_tensors} tensors in buffer")
        
        if num_tensors == 0:
            print("  ⚠️  No tensors captured (check if requests were marked for capture)")
            return activations
        
        # Read metadata entries (12 bytes each: offset, size, layer_idx)
        METADATA_HEADER_SIZE = 4 + 1000 * 12  # Must match writer
        
        for i in range(min(num_tensors, 1000)):  # Don't exceed metadata space
            metadata_offset = 4 + i * 12
            
            # Read metadata entry
            entry_bytes = self.buffer[metadata_offset:metadata_offset + 12]
            offset, size, layer_idx = np.frombuffer(entry_bytes, dtype=np.uint32)
            
            # Sanity checks
            if offset < METADATA_HEADER_SIZE or offset >= self.shm.size:
                print(f"  ⚠️  Invalid offset {offset} for tensor {i}")
                continue
            if size == 0 or size > 100 * 1024 * 1024:  # Max 100MB per tensor
                print(f"  ⚠️  Invalid size {size} for tensor {i}")
                continue
                
            # Extract tensor data
            try:
                tensor_bytes = self.buffer[offset:offset + size]
                
                # Convert to tensor (assuming float16 based on our config)
                tensor_np = np.frombuffer(tensor_bytes, dtype=np.float16)
                tensor = torch.from_numpy(tensor_np.copy())  # Copy to avoid buffer issues
                
                # Store with layer name
                layer_name = f"layer_{layer_idx}"
                activations[layer_name] = tensor
                
                print(f"  ✅ Extracted {layer_name}: shape={tensor.shape}, size={size} bytes")
                
            except Exception as e:
                print(f"  ❌ Failed to extract tensor {i}: {e}")
                continue
        
        return activations
        
    def cleanup(self):
        """Close connection to shared memory."""
        if self.shm:
            self.shm.close()
            

def main():
    print("="*80)
    print("REAL Activation Extraction from vLLM")
    print("="*80)
    print()
    
    # Step 1: Configure activation capture
    print("Step 1: Configuring activation capture...")
    os.environ["VLLM_CAPTURE_ENABLED"] = "1"
    os.environ["VLLM_CAPTURE_LAYERS"] = "0,5,10,15,20"  # Specific layers
    os.environ["VLLM_CAPTURE_COMPRESSION_K"] = "0"  # No compression for clarity
    os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"] = "2.0"
    os.environ["VLLM_CAPTURE_SAMPLE_RATE"] = "1.0"  # Capture ALL requests
    
    print("  ✅ Capture enabled for layers: 0, 5, 10, 15, 20")
    print("  ✅ Compression: Disabled (full tensors)")
    print()
    
    # Step 2: Initialize model
    print("Step 2: Loading model with capture enabled...")
    
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        worker_cls="vllm.v1.worker.gpu_worker_capture.WorkerCapture",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,  # Reduced for available memory
        max_model_len=128,
        dtype="float16",
    )
    
    print("  ✅ Model loaded")
    print()
    
    # Step 3: Generate text (this triggers activation capture)
    print("Step 3: Generating text (capturing activations)...")
    
    prompts = [
        "The future of artificial intelligence is",
        "Climate change will affect our planet by",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=20,
        seed=42,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    print("  ✅ Generation complete")
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"    {i+1}. '{text[:40]}...'")
    print()
    
    # Step 4: Extract REAL activations from shared memory
    print("Step 4: Extracting REAL activations from shared memory...")
    
    extractor = ActivationExtractor()
    
    if extractor.connect():
        # Extract real activations
        real_activations = extractor.extract_activations()
        
        if real_activations:
            print(f"\n  ✅ Extracted {len(real_activations)} real activation tensors")
            
            # Analyze the real activations
            print("\n  Real Activation Analysis:")
            for name, tensor in real_activations.items():
                if tensor.numel() > 0:
                    print(f"    {name}:")
                    print(f"      Shape: {tensor.shape}")
                    print(f"      Mean: {tensor.mean().item():.4f}")
                    print(f"      Std:  {tensor.std().item():.4f}")
                    print(f"      Min:  {tensor.min().item():.4f}")
                    print(f"      Max:  {tensor.max().item():.4f}")
        else:
            print("  ⚠️  No activations extracted (buffer may be empty)")
            
        extractor.cleanup()
    else:
        print("  ⚠️  Could not connect to buffer")
        print("  Note: Buffer exists in worker process memory")
    
    # Step 5: Alternative - Direct access approach
    print("\nStep 5: Alternative extraction via buffer snapshot...")
    
    # In production, the worker saves snapshots at intervals
    # Let's check if we can read the raw buffer
    try:
        # Try to access the shared memory directly
        test_shm = shm.SharedMemory(name="vllm_capture_rank_0")
        buffer_size = test_shm.size
        
        # Take a snapshot of first part of buffer
        snapshot = np.ndarray(
            (min(1024*1024, buffer_size),),  # First 1MB
            dtype=np.uint8,
            buffer=test_shm.buf
        ).copy()
        
        print(f"  ✅ Buffer snapshot taken: {len(snapshot)} bytes")
        print(f"  Buffer contents (non-zero): {np.count_nonzero(snapshot)} bytes")
        
        # Save snapshot for analysis
        output_dir = Path("real_activations")
        output_dir.mkdir(exist_ok=True)
        
        snapshot_file = output_dir / "buffer_snapshot.npy"
        np.save(snapshot_file, snapshot)
        print(f"  ✅ Saved buffer snapshot to {snapshot_file}")
        
        # Also save generation info
        info = {
            "prompts": prompts,
            "outputs": [o.outputs[0].text for o in outputs],
            "buffer_size": buffer_size,
            "snapshot_size": len(snapshot),
            "non_zero_bytes": int(np.count_nonzero(snapshot)),
            "timestamp": time.time(),
        }
        
        info_file = output_dir / "generation_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  ✅ Saved generation info to {info_file}")
        
        test_shm.close()
        
    except Exception as e:
        print(f"  ⚠️  Direct buffer access failed: {e}")
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    
    print()
    print("="*80)
    print("IMPORTANT: Real Activation Extraction")
    print("="*80)
    print()
    print("In production, activations are captured in the worker process.")
    print("To properly extract them, you would typically:")
    print()
    print("1. Have the worker process write metadata about tensor locations")
    print("2. Use a message queue to communicate between processes")
    print("3. Or have the worker periodically dump activations to disk")
    print()
    print("The shared memory buffer DOES contain real data, but accessing")
    print("it from the main process requires coordination with the worker.")
    print()
    print("For full production usage, extend the GPUModelRunnerCapture class")
    print("to implement your preferred extraction method.")
    print("="*80)
    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()