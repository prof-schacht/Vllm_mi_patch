#!/usr/bin/env python3
"""
Comprehensive testing of vLLM activation capture with Qwen2-7B model.
Tests scaling from 3 layers to all layers with real measurements.
Monitors GPU utilization in background.
"""

import os
import sys
import torch
import time
import json
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import multiprocessing
import numpy as np
import gc

# Use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Ensure we're using our fork
sys.path.insert(0, "/proj/vllm-capture")

from vllm import LLM, SamplingParams
import vllm

@dataclass
class LayerTestResult:
    model_name: str
    num_layers_captured: int
    layer_indices: str
    compression: Optional[int]
    
    # Performance metrics
    generation_time: float
    tokens_per_second: float
    total_tokens: int
    
    # Memory metrics
    gpu_memory_before_gb: float
    gpu_memory_after_gb: float
    gpu_memory_peak_gb: float
    buffer_size_mb: float
    estimated_activation_size_mb: float
    
    # GPU utilization
    avg_gpu_util: float
    peak_gpu_util: float
    
    # Status
    success: bool
    error: Optional[str] = None

class GPUMonitor:
    """Background thread to monitor GPU utilization."""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.running = False
        self.thread = None
        self.samples = []
        self.memory_samples = []
        
    def start(self):
        """Start monitoring GPU."""
        self.running = True
        self.samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.samples:
            return {
                'avg_util': np.mean(self.samples),
                'peak_util': np.max(self.samples),
                'avg_memory': np.mean(self.memory_samples) if self.memory_samples else 0,
                'peak_memory': np.max(self.memory_samples) if self.memory_samples else 0,
            }
        return {'avg_util': 0, 'peak_util': 0, 'avg_memory': 0, 'peak_memory': 0}
    
    def _monitor_loop(self):
        """Monitor loop that runs in background."""
        while self.running:
            try:
                # Get GPU utilization using nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                     '--format=csv,noheader,nounits', f'-i={self.gpu_id}'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    if len(parts) >= 2:
                        util = float(parts[0])
                        mem_mb = float(parts[1])
                        self.samples.append(util)
                        self.memory_samples.append(mem_mb / 1024)  # Convert to GB
            except:
                pass
            time.sleep(0.1)  # Sample every 100ms

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def get_model_layer_count(model_name: str) -> int:
    """Get the number of layers in a model."""
    # For Qwen models, we can estimate based on size
    if "7B" in model_name or "7b" in model_name:
        return 32  # Qwen2-7B has 32 layers
    elif "0.5B" in model_name or "0.5b" in model_name:
        return 24  # Qwen2-0.5B has 24 layers
    else:
        return 32  # Default assumption

def test_layer_configuration(
    model_name: str,
    layer_indices: List[int],
    compression_k: Optional[int],
    num_prompts: int = 5,
    max_tokens: int = 50
) -> LayerTestResult:
    """Test a specific layer configuration."""
    
    layer_str = ','.join(map(str, layer_indices)) if layer_indices else "all"
    comp_str = f"SVD-{compression_k}" if compression_k else "None"
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Layers: {layer_str} ({len(layer_indices) if layer_indices else 'all'} layers)")
    print(f"Compression: {comp_str}")
    print(f"{'='*70}")
    
    # Set environment variables
    os.environ["VLLM_CAPTURE_ENABLED"] = "1"
    os.environ["VLLM_CAPTURE_LAYERS"] = layer_str if layer_indices else "all"
    
    if compression_k:
        os.environ["VLLM_CAPTURE_COMPRESSION_K"] = str(compression_k)
    else:
        os.environ.pop("VLLM_CAPTURE_COMPRESSION_K", None)
    
    # Increase buffer size for more layers
    if layer_indices is None:
        buffer_gb = 4.0  # All layers needs more buffer
    else:
        buffer_gb = 4.0 if len(layer_indices) > 10 else 2.0
    os.environ["VLLM_CAPTURE_BUFFER_SIZE_GB"] = str(buffer_gb)
    os.environ["VLLM_CAPTURE_SAMPLE_RATE"] = "1.0"
    
    result = LayerTestResult(
        model_name=model_name,
        num_layers_captured=len(layer_indices) if layer_indices else get_model_layer_count(model_name),
        layer_indices=layer_str,
        compression=compression_k,
        generation_time=0,
        tokens_per_second=0,
        total_tokens=0,
        gpu_memory_before_gb=0,
        gpu_memory_after_gb=0,
        gpu_memory_peak_gb=0,
        buffer_size_mb=buffer_gb * 1024,
        estimated_activation_size_mb=0,
        avg_gpu_util=0,
        peak_gpu_util=0,
        success=False
    )
    
    # Start GPU monitoring
    monitor = GPUMonitor(gpu_id=0)
    monitor.start()
    
    try:
        # Record initial memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result.gpu_memory_before_gb = get_gpu_memory()
        
        print(f"Loading model... (Initial GPU memory: {result.gpu_memory_before_gb:.2f} GB)")
        
        # Initialize model
        llm = LLM(
            model=model_name,
            worker_cls="vllm.v1.worker.gpu_worker_capture.WorkerCapture",
            enforce_eager=True,  # Required for hooks
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,  # Use 70% for larger models
            max_model_len=512,
            dtype="float16",
            download_dir="/data/models",  # Cache models
        )
        
        # Record memory after model load
        torch.cuda.synchronize()
        model_load_memory = get_gpu_memory()
        print(f"Model loaded. GPU memory: {model_load_memory:.2f} GB")
        
        # Prepare test prompts
        prompts = [
            "The future of artificial intelligence involves",
            "Climate change can be addressed through",
            "The most important scientific discovery was",
            "Economic growth depends on factors like",
            "Education systems should focus on developing",
            "Healthcare innovations include technologies such as",
            "Renewable energy sources like solar and wind",
            "Space exploration will lead humanity to",
            "Quantum computing promises to revolutionize",
            "The internet has transformed society by",
        ][:num_prompts]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        
        # Warm up
        print("Warming up...")
        _ = llm.generate(["Hello world"], SamplingParams(max_tokens=5))
        torch.cuda.synchronize()
        
        # Main generation
        print(f"Generating {num_prompts} responses with {max_tokens} tokens each...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        torch.cuda.synchronize()
        generation_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
        result.generation_time = generation_time
        result.tokens_per_second = tokens_per_second
        result.total_tokens = total_tokens
        
        # Record final memory
        result.gpu_memory_after_gb = get_gpu_memory()
        
        # Stop GPU monitoring and get stats
        gpu_stats = monitor.stop()
        result.avg_gpu_util = gpu_stats['avg_util']
        result.peak_gpu_util = gpu_stats['peak_util']
        result.gpu_memory_peak_gb = gpu_stats['peak_memory']
        
        # Check activation buffer
        try:
            import multiprocessing.shared_memory as shm
            worker_shm = shm.SharedMemory(name="vllm_capture_rank_0")
            
            # Estimate activation size based on buffer usage
            # This is approximate - actual usage depends on compression
            if compression_k:
                # With compression, estimate based on compression ratio
                bytes_per_activation = compression_k * 2 * 2  # k components * 2 bytes * safety factor
            else:
                # Without compression, full tensor size
                bytes_per_activation = 4096 * 2  # hidden_dim * 2 bytes (float16)
            
            activations_per_token = result.num_layers_captured
            total_activations = total_tokens * activations_per_token
            result.estimated_activation_size_mb = (total_activations * bytes_per_activation) / (1024**2)
            
            worker_shm.close()
            print(f"✅ Activations captured successfully")
            
        except FileNotFoundError:
            print("⚠️ No activation buffer found")
            result.estimated_activation_size_mb = 0
        
        result.success = True
        
        # Print results
        print(f"\n--- Results ---")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Throughput: {tokens_per_second:.1f} tokens/s")
        print(f"Total tokens: {total_tokens}")
        print(f"GPU Memory: {result.gpu_memory_before_gb:.2f} GB → {result.gpu_memory_after_gb:.2f} GB (peak: {result.gpu_memory_peak_gb:.2f} GB)")
        print(f"GPU Utilization: {result.avg_gpu_util:.1f}% avg, {result.peak_gpu_util:.1f}% peak")
        print(f"Estimated activation storage: {result.estimated_activation_size_mb:.1f} MB")
        
        # Show sample output
        if outputs:
            sample = outputs[0].outputs[0].text[:100]
            print(f"\nSample output: '{sample}...'")
        
        # Cleanup
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        # Try to clean up shared memory
        try:
            os.system("rm -f /dev/shm/vllm_capture_rank_0 2>/dev/null")
        except:
            pass
        
        time.sleep(3)  # Let processes clean up
        
    except Exception as e:
        monitor.stop()
        result.error = str(e)
        print(f"❌ Error: {e}")
        
        # Cleanup on error
        try:
            os.system("rm -f /dev/shm/vllm_capture_rank_0 2>/dev/null")
        except:
            pass
    
    return result

def main():
    """Run comprehensive tests with increasing layer counts."""
    
    print("="*80)
    print("Comprehensive vLLM Activation Capture Testing")
    print("="*80)
    print(f"vLLM version: {vllm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Model to test
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Will use Qwen2-7B as 8B not available
    
    # Get total layers for this model
    total_layers = get_model_layer_count(model_name)
    print(f"\nModel: {model_name}")
    print(f"Estimated total layers: {total_layers}")
    
    # Test configurations with increasing layer counts
    test_configs = [
        # Start with 3 layers
        (list(range(3)), None),  # Layers 0,1,2 uncompressed
        (list(range(3)), 256),   # Layers 0,1,2 with SVD-256
        
        # 5 layers
        (list(range(5)), None),  # Layers 0-4 uncompressed
        (list(range(5)), 256),   # Layers 0-4 with compression
        
        # 10 layers
        (list(range(10)), None),  # Layers 0-9 uncompressed
        (list(range(10)), 256),   # Layers 0-9 with compression
        
        # 16 layers (half)
        (list(range(16)), 256),   # Layers 0-15 with compression
        
        # All layers (if memory permits)
        (None, 256),  # All layers with compression
    ]
    
    results = []
    baseline_throughput = None
    
    # First, run without capture to get baseline
    print("\n" + "="*70)
    print("BASELINE TEST (No Activation Capture)")
    print("="*70)
    
    os.environ["VLLM_CAPTURE_ENABLED"] = "0"
    
    try:
        llm = LLM(
            model=model_name,
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=512,
            dtype="float16",
            download_dir="/data/models",
        )
        
        prompts = ["The future of AI is"] * 5
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        baseline_time = time.time() - start
        
        baseline_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        baseline_throughput = baseline_tokens / baseline_time
        
        print(f"Baseline throughput: {baseline_throughput:.1f} tokens/s")
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)
        
    except Exception as e:
        print(f"Baseline test failed: {e}")
        baseline_throughput = None
    
    # Run layer scaling tests
    for layer_indices, compression_k in test_configs:
        result = test_layer_configuration(
            model_name=model_name,
            layer_indices=layer_indices,
            compression_k=compression_k,
            num_prompts=5,
            max_tokens=50
        )
        results.append(result)
        
        # Stop if we're running out of memory
        if not result.success and "out of memory" in str(result.error).lower():
            print("\n⚠️ Out of memory - stopping tests")
            break
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    if baseline_throughput:
        print(f"\nBaseline Performance (No Capture): {baseline_throughput:.1f} tokens/s")
    
    print(f"\n{'Layers':<10} {'Compression':<12} {'Throughput':<15} {'Overhead':<12} {'Memory (GB)':<15} {'Act. Size (MB)':<15} {'GPU Util':<12}")
    print("-"*110)
    
    for result in results:
        if result.success:
            overhead = ((baseline_throughput - result.tokens_per_second) / baseline_throughput * 100) if baseline_throughput else 0
            memory_str = f"{result.gpu_memory_after_gb:.1f} (Δ{result.gpu_memory_after_gb - result.gpu_memory_before_gb:+.1f})"
            gpu_util_str = f"{result.avg_gpu_util:.0f}%/{result.peak_gpu_util:.0f}%"
            
            print(f"{result.num_layers_captured:<10} "
                  f"{'None' if not result.compression else f'SVD-{result.compression}':<12} "
                  f"{result.tokens_per_second:<15.1f} "
                  f"{overhead:<12.1f}% "
                  f"{memory_str:<15} "
                  f"{result.estimated_activation_size_mb:<15.1f} "
                  f"{gpu_util_str:<12}")
        else:
            print(f"{result.num_layers_captured:<10} "
                  f"{'None' if not result.compression else f'SVD-{result.compression}':<12} "
                  f"{'FAILED':<15} "
                  f"{'-':<12} "
                  f"{'-':<15} "
                  f"{'-':<15} "
                  f"{'-':<12}")
    
    # Memory scaling analysis
    print("\n" + "="*80)
    print("MEMORY SCALING ANALYSIS")
    print("="*80)
    
    successful_results = [r for r in results if r.success]
    
    if successful_results:
        # Group by compression
        uncompressed = [r for r in successful_results if not r.compression]
        compressed = [r for r in successful_results if r.compression]
        
        if uncompressed:
            print("\nUncompressed Activation Storage:")
            for r in uncompressed:
                mb_per_layer = r.estimated_activation_size_mb / r.num_layers_captured if r.num_layers_captured > 0 else 0
                print(f"  {r.num_layers_captured} layers: {r.estimated_activation_size_mb:.1f} MB total, {mb_per_layer:.1f} MB/layer")
        
        if compressed:
            print(f"\nCompressed Activation Storage (SVD):")
            for r in compressed:
                mb_per_layer = r.estimated_activation_size_mb / r.num_layers_captured if r.num_layers_captured > 0 else 0
                print(f"  {r.num_layers_captured} layers: {r.estimated_activation_size_mb:.1f} MB total, {mb_per_layer:.1f} MB/layer")
        
        # Extrapolation for 100 agents, 500 timesteps
        print("\n" + "="*80)
        print("EXTRAPOLATION FOR MULTI-AGENT SCENARIOS")
        print("="*80)
        
        print("\nFor 100 agents, 500 timesteps (50,000 total inferences):")
        
        for r in successful_results[:3]:  # Show first few configs
            total_storage_gb = (r.estimated_activation_size_mb * 50000 / r.total_tokens) / 1024
            total_time_hours = (r.generation_time * 50000 / (r.total_tokens / r.tokens_per_second)) / 3600
            
            comp_str = "Uncompressed" if not r.compression else f"SVD-{r.compression}"
            print(f"\n  {r.num_layers_captured} layers, {comp_str}:")
            print(f"    Storage needed: {total_storage_gb:.1f} GB")
            print(f"    Estimated time: {total_time_hours:.1f} hours")
            print(f"    GPU memory required: {r.gpu_memory_peak_gb:.1f} GB")
    
    # Save results to JSON
    results_dict = [asdict(r) for r in results]
    with open('/proj/vllm-capture/layer_scaling_results.json', 'w') as f:
        json.dump({
            'model': model_name,
            'baseline_throughput': baseline_throughput,
            'results': results_dict
        }, f, indent=2)
    
    print(f"\n✅ Results saved to layer_scaling_results.json")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()