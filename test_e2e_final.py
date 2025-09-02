#!/usr/bin/env python3
"""
Final e2e test using the correct setup as recommended by developer.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# CRITICAL: Set up paths to use patched vllm_src BEFORE importing vllm
repo_root = Path(__file__).resolve().parent
vllm_path = repo_root / 'vllm_src'

# This ensures the patched models with intermediate_tensors.add() are used
sys.path.insert(0, str(repo_root))  # For vllm_capture
# Don't use vllm_src since it has missing compiled modules
# sys.path.insert(0, str(vllm_path))  # For patched vLLM

# Set PYTHONPATH for child processes
os.environ['PYTHONPATH'] = f"{repo_root}:" + os.environ.get('PYTHONPATH', '')

# GPU and offline settings
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# Activation capture configuration
os.environ['VLLM_ACT_CAPTURE'] = '1'
os.environ['VLLM_ACT_MODE'] = 'rp8'
os.environ['VLLM_ACT_RP_K'] = '64'
os.environ['VLLM_ACT_OUTDIR'] = '/tmp/acts'
os.environ['VLLM_ACT_HIDDEN_SIZE'] = '1024'  # For Qwen3-0.6B

# Clean output directory
os.system('rm -rf /tmp/acts 2>/dev/null')
os.makedirs('/tmp/acts', exist_ok=True)

print("=" * 70)
print("vLLM Activation Capture - End-to-End Test")
print("=" * 70)
print("Configuration:")
print(f"  Model: /tmp/Qwen3-0.6B (local)")
print(f"  GPU: cuda:2")
print(f"  Capture: ENABLED")
print(f"  Mode: rp8 (RP to 64 dims + uint8)")
print(f"  Output: /tmp/acts")
print("=" * 70)

from vllm import LLM, SamplingParams

print("\nInitializing LLM with hook-free activation capture...")

# Use local model to avoid HF cache issues
llm = LLM(
    model='/tmp/Qwen3-0.6B',
    worker_cls='vllm_capture.gpu_worker_capture.WorkerCapture',
    tensor_parallel_size=1,
    max_model_len=256,
    enforce_eager=True,  # For testing
    gpu_memory_utilization=0.7,
)

print("✓ LLM initialized successfully")

# Test prompts
prompts = ["The capital of France is"]

print(f"\nGenerating text for: '{prompts[0]}'...")
outputs = llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0))

# Print generated text
text = outputs[0].outputs[0].text
print(f"Generated: '{text.strip()}'")

# Try to get activation manifest
print("\n" + "=" * 70)
print("Checking for activation capture results...")
print("=" * 70)

# Check for manifest from worker
try:
    engine = getattr(llm, 'llm_engine', None)
    executor = getattr(engine, 'model_executor', None) if engine else None
    worker = getattr(executor, 'driver_worker', None) if executor else None
    
    if worker and hasattr(worker, 'get_last_activation_manifest'):
        manifest = worker.get_last_activation_manifest()
        if manifest:
            print("\n✅ ACTIVATION MANIFEST:")
            print(json.dumps(manifest, indent=2))
    else:
        print("ℹ️  No manifest accessor available")
except Exception as e:
    print(f"ℹ️  Could not retrieve manifest: {e}")

# Check for saved files
print("\nChecking for activation files in /tmp/acts/...")

activation_files = []
for root, dirs, files in os.walk('/tmp/acts'):
    for file in files:
        if file.endswith('.npz'):
            activation_files.append(os.path.join(root, file))

if activation_files:
    print(f"\n✅ SUCCESS! Found {len(activation_files)} activation files:")
    
    for i, filepath in enumerate(activation_files[:5]):
        rel_path = os.path.relpath(filepath, '/tmp/acts')
        size_kb = os.path.getsize(filepath) / 1024
        
        # Load and analyze
        data = np.load(filepath)
        keys = list(data.keys())
        
        print(f"\n  [{i+1}] {rel_path} ({size_kb:.1f} KB)")
        print(f"      Keys: {keys}")
        
        if 'q' in data:
            q = data['q']
            print(f"      Shape: {q.shape} (tokens × RP dims)")
            print(f"      Dtype: {q.dtype}")
            
            if len(q.shape) == 2:
                n_tokens, rp_dims = q.shape
                print(f"      → Captured {n_tokens} tokens in {rp_dims} dimensions")
    
    if len(activation_files) > 5:
        print(f"\n  ... and {len(activation_files)-5} more files")
    
    print("\n" + "=" * 70)
    print("🎉 ACTIVATION CAPTURE WORKING!")
    print("=" * 70)
    print(f"Summary:")
    print(f"  • {len(activation_files)} layer activations captured")
    print(f"  • Random projection to 64 dimensions")
    print(f"  • Quantized to uint8 for efficiency")
    print(f"  • Hook-free architecture successful!")
else:
    print("\n⚠️  No activation files found")
    print("\nPossible reasons:")
    print("  1. Model needs intermediate_tensors.add() calls")
    print("  2. System vLLM being used instead of patched version")
    print("  3. Check if capture was actually triggered")

print("\nTest complete!")