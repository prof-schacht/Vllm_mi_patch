#!/usr/bin/env python3
"""
Performance testing of vLLM activation capture with different models and configurations.
Tests both compressed and uncompressed capture modes.
"""

import os
import sys
import torch
import time
import json
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import multiprocessing
import gc

# Use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Ensure we're using our fork
sys.path.insert(0, "/proj/vllm-capture")

from vllm import LLM, SamplingParams
import vllm
from vllm.v1.worker.gpu_model_runner_capture import SharedActivationBuffer

@dataclass
class TestConfig:
    model_name: str
    compression_k: int  # 0 means no compression
    layers_to_capture: str  # "all" or comma-separated layer indices
    num_prompts: int
    max_tokens: int
    
@dataclass
class TestResult:
    model_name: str
    compression: str
    layers_captured: str
    generation_time: float
    tokens_per_second: float
    memory_used_gb: float
    buffer_size_mb: float
    activation_size_mb: float
    success: bool
    error: str = None

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)  # GB
    return 0

def test_configuration(config: TestConfig) -> TestResult:
    """Test a single configuration."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config.model_name}")
    print(f"Compression: {'None' if config.compression_k == 0 else f'SVD-{config.compression_k}'}")
    print(f"Layers: {config.layers_to_capture}")
    print(f"{'='*60}")
    
    # Set environment variables
    os.environ["VLLM_CAPTURE_ENABLED"] = "1"
    os.environ["VLLM_CAPTURE_LAYERS"] = config.layers_to_capture
    
    if config.compression_k > 0:
        os.environ["VLLM_CAPTURE_COMPRESSION_K"] = str(config.compression_k)
    else:
        # No compression - full tensors
        os.environ.pop("VLLM_CAPTURE_COMPRESSION_K", None)
    
    os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"] = "2.0"  # 2GB buffer
    os.environ["VLLM_CAPTURE_SAMPLE_RATE"] = "1.0"  # Capture all
    
    result = TestResult(
        model_name=config.model_name,
        compression=f"SVD-{config.compression_k}" if config.compression_k > 0 else "None",
        layers_captured=config.layers_to_capture,
        generation_time=0,
        tokens_per_second=0,
        memory_used_gb=0,
        buffer_size_mb=0,
        activation_size_mb=0,
        success=False
    )
    
    try:
        # Initialize model
        print("Loading model...")
        start_mem = get_gpu_memory()
        
        llm = LLM(
            model=config.model_name,
            worker_cls="vllm.v1.worker.gpu_worker_capture.WorkerCapture",
            enforce_eager=True,  # Required for hooks
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,  # Use 50% GPU memory
            max_model_len=512,
            dtype="float16",
        )
        
        model_mem = get_gpu_memory() - start_mem
        print(f"Model loaded, memory used: {model_mem:.2f} GB")
        
        # Prepare test prompts
        prompts = [
            "The capital of France is",
            "Machine learning is a field that",
            "The most important invention in history was",
            "Climate change affects our planet by",
            "Artificial intelligence will transform",
        ][:config.num_prompts]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=config.max_tokens,
        )
        
        # Run generation
        print(f"Generating {config.num_prompts} responses...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        torch.cuda.synchronize()
        generation_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_second = total_tokens / generation_time
        
        result.generation_time = generation_time
        result.tokens_per_second = tokens_per_second
        result.memory_used_gb = get_gpu_memory()
        
        # Check activation buffer
        worker_buffer_name = "vllm_capture_rank_0"
        try:
            import multiprocessing.shared_memory as shm
            worker_shm = shm.SharedMemory(name=worker_buffer_name)
            buffer_size_mb = worker_shm.size / (1024**2)
            
            # Estimate activation size (rough - would need metadata for exact)
            # Assume some portion of buffer is used
            activation_size_mb = buffer_size_mb * 0.1  # Conservative estimate
            
            result.buffer_size_mb = buffer_size_mb
            result.activation_size_mb = activation_size_mb
            
            worker_shm.close()
            
            print(f"\n✅ Activations captured!")
            print(f"   Buffer size: {buffer_size_mb:.2f} MB")
            
        except FileNotFoundError:
            print("❌ No activation buffer found")
            
        result.success = True
        
        # Show sample outputs
        print(f"\nGeneration complete in {generation_time:.2f}s")
        print(f"Throughput: {tokens_per_second:.2f} tokens/s")
        print(f"GPU Memory: {result.memory_used_gb:.2f} GB")
        
        print("\nSample outputs:")
        for i, (prompt, output) in enumerate(zip(prompts[:2], outputs[:2])):
            generated = output.outputs[0].text
            print(f"{i+1}. '{prompt}' -> '{generated[:50]}...'")
            
        # Cleanup
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)  # Let subprocess cleanup
        
    except Exception as e:
        result.error = str(e)
        print(f"❌ Error: {e}")
        
    return result

def main():
    """Run performance tests with different configurations."""
    
    print("="*80)
    print("vLLM Activation Capture Performance Testing")
    print("="*80)
    print(f"vLLM version: {vllm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Test configurations
    test_configs = [
        # Small model tests
        TestConfig("Qwen/Qwen2-0.5B-Instruct", compression_k=0, layers_to_capture="0,1,2", 
                   num_prompts=5, max_tokens=20),
        TestConfig("Qwen/Qwen2-0.5B-Instruct", compression_k=64, layers_to_capture="0,1,2", 
                   num_prompts=5, max_tokens=20),
        TestConfig("Qwen/Qwen2-0.5B-Instruct", compression_k=128, layers_to_capture="all", 
                   num_prompts=5, max_tokens=20),
        
        # Medium model tests (if available)
        TestConfig("Qwen/Qwen2.5-7B-Instruct", compression_k=0, layers_to_capture="0,1,2", 
                   num_prompts=3, max_tokens=20),
        TestConfig("Qwen/Qwen2.5-7B-Instruct", compression_k=256, layers_to_capture="0,5,10,15,20", 
                   num_prompts=3, max_tokens=20),
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = test_configuration(config)
            results.append(result)
        except Exception as e:
            print(f"Failed to test {config.model_name}: {e}")
            results.append(TestResult(
                model_name=config.model_name,
                compression=f"SVD-{config.compression_k}" if config.compression_k > 0 else "None",
                layers_captured=config.layers_to_capture,
                generation_time=0,
                tokens_per_second=0,
                memory_used_gb=0,
                buffer_size_mb=0,
                activation_size_mb=0,
                success=False,
                error=str(e)
            ))
        
        # Pause between tests
        time.sleep(5)
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Compression':<15} {'Layers':<10} {'Time(s)':<10} {'Tok/s':<10} {'GPU(GB)':<10} {'Status':<10}")
    print("-"*100)
    
    for result in results:
        status = "✅" if result.success else "❌"
        model_short = result.model_name.split("/")[-1][:25]
        layers_short = result.layers_captured if len(result.layers_captured) < 8 else result.layers_captured[:5] + "..."
        
        print(f"{model_short:<30} {result.compression:<15} {layers_short:<10} "
              f"{result.generation_time:<10.2f} {result.tokens_per_second:<10.1f} "
              f"{result.memory_used_gb:<10.2f} {status:<10}")
    
    # Compression efficiency analysis
    print("\n" + "="*80)
    print("COMPRESSION EFFICIENCY")
    print("="*80)
    
    uncompressed = [r for r in results if r.compression == "None" and r.success]
    compressed = [r for r in results if r.compression != "None" and r.success]
    
    if uncompressed and compressed:
        avg_uncompressed_speed = np.mean([r.tokens_per_second for r in uncompressed])
        avg_compressed_speed = np.mean([r.tokens_per_second for r in compressed])
        
        print(f"\nAverage throughput:")
        print(f"  Uncompressed: {avg_uncompressed_speed:.2f} tokens/s")
        print(f"  Compressed:   {avg_compressed_speed:.2f} tokens/s")
        print(f"  Overhead:     {(1 - avg_compressed_speed/avg_uncompressed_speed)*100:.1f}%")
        
        avg_uncompressed_mem = np.mean([r.memory_used_gb for r in uncompressed])
        avg_compressed_mem = np.mean([r.memory_used_gb for r in compressed])
        
        print(f"\nAverage GPU memory:")
        print(f"  Uncompressed: {avg_uncompressed_mem:.2f} GB")
        print(f"  Compressed:   {avg_compressed_mem:.2f} GB")
        print(f"  Savings:      {(1 - avg_compressed_mem/avg_uncompressed_mem)*100:.1f}%")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)

if __name__ == "__main__":
    # Ensure proper multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()