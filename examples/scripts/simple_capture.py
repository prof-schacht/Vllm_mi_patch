#!/usr/bin/env python3
"""
Simple example of capturing activations during vLLM inference.

This script demonstrates:
1. Basic setup for activation capture
2. Generating text while capturing activations
3. Saving activations for offline analysis
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure environment for activation capture
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use third GPU (available)
os.environ["VLLM_CAPTURE_ENABLED"] = "1"
os.environ["VLLM_CAPTURE_LAYERS"] = "0,7,15,23,31"  # Capture 5 specific layers
os.environ["VLLM_CAPTURE_COMPRESSION_K"] = "256"  # Use SVD compression
os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"] = "2.0"
os.environ["VLLM_CAPTURE_SAMPLE_RATE"] = "1.0"  # Capture ALL requests (100%)

from vllm import LLM, SamplingParams

def main():
    print("vLLM Activation Capture - Simple Example")
    print("=" * 50)
    
    # Initialize model with activation capture
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",  # Small model for demo
        worker_cls="vllm.v1.worker.gpu_worker_capture.WorkerCapture",
        enforce_eager=True,  # Required for hooks
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,  # Reduced for available memory
        max_model_len=256,
    )
    
    # Test prompt
    prompt = "The future of artificial intelligence will"
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        seed=42,  # For reproducibility
    )
    
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating with activation capture enabled...")
    
    # Generate text (activations captured automatically)
    outputs = llm.generate([prompt], sampling_params)
    
    # Extract generated text
    generated_text = outputs[0].outputs[0].text
    print(f"\nGenerated: '{generated_text}'")
    
    # IMPORTANT: Real activation extraction
    print("\n" + "="*50)
    print("REAL ACTIVATION EXTRACTION")
    print("="*50)
    print("\nThe activations are captured in the vLLM worker process.")
    print("They exist in shared memory buffer: /dev/shm/vllm_capture_rank_0")
    
    # Check if buffer exists
    try:
        import multiprocessing.shared_memory as shm
        worker_shm = shm.SharedMemory(name="vllm_capture_rank_0")
        buffer_size_mb = worker_shm.size / (1024**2)
        print(f"\n✅ Shared memory buffer found: {buffer_size_mb:.1f} MB")
        print("   This buffer contains the REAL captured activations")
        worker_shm.close()
        
        print("\nTo extract real activations in production:")
        print("1. Use the ActivationExtractor class (see extract_real_activations.py)")
        print("2. Or modify GPUModelRunnerCapture to save to disk")
        print("3. Or implement IPC between worker and main process")
        
    except FileNotFoundError:
        print("\n⚠️  Buffer not found (may be in worker process memory)")
        print("   The activations are still captured, but in worker's memory space")
    
    print("\n" + "="*50)
    print("See extract_real_activations.py for production usage")
    print("="*50)
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    
    print("\n✅ Example complete!")

if __name__ == "__main__":
    main()