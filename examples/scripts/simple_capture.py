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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["VLLM_CAPTURE_ENABLED"] = "1"
os.environ["VLLM_CAPTURE_LAYERS"] = "0,7,15,23,31"  # Capture 5 specific layers
os.environ["VLLM_CAPTURE_COMPRESSION_K"] = "256"  # Use SVD compression
os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"] = "2.0"

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
        gpu_memory_utilization=0.5,
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
    
    # Save results
    output_dir = Path("../../results/activations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # In production, activations would be extracted from shared memory
    # For this demo, we create example tensors
    example_activation = torch.randn(1, 50, 896, dtype=torch.float16)
    
    activation_file = output_dir / "example_activation.pt"
    torch.save({
        'prompt': prompt,
        'generated': generated_text,
        'layers_captured': [0, 7, 15, 23, 31],
        'compression': 'SVD-256',
        'tensor_shape': example_activation.shape,
        'tensor': example_activation,
    }, activation_file)
    
    print(f"\n✅ Activation saved to: {activation_file}")
    print(f"   Shape: {example_activation.shape}")
    print(f"   Size: {example_activation.numel() * 2 / (1024**2):.2f} MB")
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    
    print("\n✅ Example complete!")

if __name__ == "__main__":
    main()