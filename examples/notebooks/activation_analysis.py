#!/usr/bin/env python3
"""
Example notebook-style script for activation analysis.

This can be run as a Python script or converted to Jupyter notebook.
Demonstrates loading, analyzing, and visualizing captured activations.
"""

# %% [markdown]
# # vLLM Activation Analysis Example
# 
# This notebook demonstrates how to load and analyze activations captured from vLLM models.

# %% Import required libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm_capture.analysis import (
    ActivationLoader, 
    quick_load,
    load_sequence_activations,
    compare_tokens
)

# %% [markdown]
# ## 1. Quick Load Example
# 
# The simplest way to load activations for interactive exploration:

# %% Quick load activations
# Replace with your actual output directory
output_dir = "/tmp/acts"

# Load first request, layer 10, reconstructed
try:
    acts = quick_load(output_dir, layer=10, reconstruct=True)
    print(f"Loaded activation shape: {acts.shape}")
    print(f"Min: {acts.min():.3f}, Max: {acts.max():.3f}, Mean: {acts.mean():.3f}")
except Exception as e:
    print(f"Note: {e}")
    print("Run a generation with activation capture first to create data")
    # Create dummy data for demonstration
    acts = np.random.randn(5, 64).astype(np.float32)
    print("Using dummy data for demonstration")

# %% [markdown]
# ## 2. Using ActivationLoader for Detailed Analysis

# %% Initialize loader
loader = ActivationLoader(output_dir, mode="rp8")

# List available requests
requests = loader.list_requests()
if requests:
    print(f"Found {len(requests)} captured requests:")
    for req in requests[:5]:
        print(f"  - {req}")
else:
    print("No captured requests found. Using example data.")
    # Create example structure
    requests = ["req_example"]

# %% [markdown]
# ## 3. Load and Analyze a Full Request

# %% Load all activations for a request
if requests:
    request_id = requests[0]
    print(f"\nAnalyzing request: {request_id}")
    
    try:
        # Load all sequences and layers
        all_activations = loader.load_request(request_id, reconstruct=True)
        
        # Get first sequence
        seq_id = min(all_activations.keys()) if all_activations else 0
        sequence_acts = all_activations.get(seq_id, {})
        
        print(f"Sequence {seq_id}:")
        print(f"  Layers captured: {sorted(sequence_acts.keys())}")
        
        # Compute statistics for each layer
        if sequence_acts:
            stats = loader.compute_layer_statistics(sequence_acts)
            
            print("\nLayer Statistics:")
            for layer_id in sorted(stats.keys())[:5]:
                s = stats[layer_id]
                print(f"  Layer {layer_id:2d}: "
                      f"mean={s['mean']:6.3f}, "
                      f"std={s['std']:6.3f}, "
                      f"norm={s['norm']:8.2f}")
    except Exception as e:
        print(f"Could not load real data: {e}")
        # Create example data
        sequence_acts = {
            0: np.random.randn(5, 64),
            1: np.random.randn(5, 64),
            2: np.random.randn(5, 64),
        }

# %% [markdown]
# ## 4. Visualize Activation Patterns

# %% Create visualization
if sequence_acts and len(sequence_acts) > 0:
    # Select a layer to visualize
    layer_id = min(sequence_acts.keys())
    layer_acts = sequence_acts[layer_id]
    
    # Ensure it's numpy array
    if hasattr(layer_acts, 'cpu'):
        layer_acts = layer_acts.cpu().numpy()
    elif isinstance(layer_acts, dict) and 'q' in layer_acts:
        # Still compressed, reconstruct
        from vllm_capture.analysis.load import reconstruct_activations
        layer_acts = reconstruct_activations(layer_acts, mode='rp8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Heatmap of activations
    ax = axes[0, 0]
    im = ax.imshow(layer_acts, aspect='auto', cmap='coolwarm')
    ax.set_title(f'Layer {layer_id} Activations Heatmap')
    ax.set_xlabel('Activation Dimension')
    ax.set_ylabel('Token Position')
    plt.colorbar(im, ax=ax)
    
    # 2. Token norms over positions
    ax = axes[0, 1]
    token_norms = np.linalg.norm(layer_acts, axis=1)
    ax.plot(token_norms, 'o-')
    ax.set_title('Token Activation Norms')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    
    # 3. Dimension statistics
    ax = axes[1, 0]
    dim_means = np.mean(layer_acts, axis=0)
    dim_stds = np.std(layer_acts, axis=0)
    x = np.arange(len(dim_means))
    ax.bar(x[::2], dim_means[::2], width=1.5, alpha=0.7, label='Mean')
    ax.errorbar(x[::2], dim_means[::2], yerr=dim_stds[::2], 
                fmt='none', color='black', alpha=0.5)
    ax.set_title('Activation Statistics per Dimension (subset)')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.legend()
    
    # 4. Cosine similarity matrix between tokens
    ax = axes[1, 1]
    # Compute pairwise cosine similarities
    norm_acts = layer_acts / (np.linalg.norm(layer_acts, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = norm_acts @ norm_acts.T
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Token-to-Token Cosine Similarity')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Token Position')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nVisualization Summary:")
    print(f"  Layer {layer_id} shape: {layer_acts.shape}")
    print(f"  Mean activation: {np.mean(layer_acts):.3f}")
    print(f"  Std activation: {np.std(layer_acts):.3f}")
    print(f"  Max token norm: {np.max(token_norms):.3f}")
    print(f"  Min token norm: {np.min(token_norms):.3f}")

# %% [markdown]
# ## 5. Compare Activations Across Layers

# %% Layer comparison
if sequence_acts and len(sequence_acts) > 1:
    layer_ids = sorted(sequence_acts.keys())
    
    # Compute similarity between consecutive layers
    similarities = []
    for i in range(len(layer_ids) - 1):
        layer1 = sequence_acts[layer_ids[i]]
        layer2 = sequence_acts[layer_ids[i + 1]]
        
        # Ensure numpy arrays
        if hasattr(layer1, 'cpu'):
            layer1 = layer1.cpu().numpy()
        if hasattr(layer2, 'cpu'):
            layer2 = layer2.cpu().numpy()
        
        # Flatten and compute cosine similarity
        sim = np.dot(layer1.flatten(), layer2.flatten()) / (
            np.linalg.norm(layer1.flatten()) * np.linalg.norm(layer2.flatten())
        )
        similarities.append(sim)
    
    # Plot layer-to-layer similarities
    if similarities:
        plt.figure(figsize=(10, 5))
        plt.plot(similarities, 'o-')
        plt.title('Cosine Similarity Between Consecutive Layers')
        plt.xlabel('Layer Transition')
        plt.ylabel('Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        # Add layer labels
        xticks = [f"{layer_ids[i]}→{layer_ids[i+1]}" 
                  for i in range(len(similarities))]
        plt.xticks(range(len(similarities)), xticks, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Average layer-to-layer similarity: {np.mean(similarities):.3f}")
        print(f"Min similarity: {np.min(similarities):.3f} at transition {xticks[np.argmin(similarities)]}")
        print(f"Max similarity: {np.max(similarities):.3f} at transition {xticks[np.argmax(similarities)]}")

# %% [markdown]
# ## 6. Token Evolution Across Layers

# %% Track specific token across layers
if sequence_acts and len(sequence_acts) > 2:
    # Track first token (position 0) across all layers
    token_position = 0
    
    token_evolution = []
    layer_ids = sorted(sequence_acts.keys())
    
    for layer_id in layer_ids:
        layer_acts = sequence_acts[layer_id]
        if hasattr(layer_acts, 'cpu'):
            layer_acts = layer_acts.cpu().numpy()
        
        if token_position < len(layer_acts):
            token_evolution.append(layer_acts[token_position])
    
    if token_evolution:
        token_evolution = np.array(token_evolution)
        
        # Plot evolution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap of token evolution
        ax = axes[0]
        im = ax.imshow(token_evolution.T, aspect='auto', cmap='coolwarm')
        ax.set_title(f'Token {token_position} Evolution Across Layers')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Dimension')
        plt.colorbar(im, ax=ax)
        
        # Norm evolution
        ax = axes[1]
        norms = np.linalg.norm(token_evolution, axis=1)
        ax.plot(layer_ids[:len(norms)], norms, 'o-')
        ax.set_title(f'Token {token_position} Norm Evolution')
        ax.set_xlabel('Layer')
        ax.set_ylabel('L2 Norm')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nToken {token_position} Evolution:")
        print(f"  Initial norm: {norms[0]:.3f}")
        print(f"  Final norm: {norms[-1]:.3f}")
        print(f"  Max norm at layer {layer_ids[np.argmax(norms)]}: {np.max(norms):.3f}")

# %% [markdown]
# ## 7. Summary and Insights
# 
# This notebook demonstrated:
# 
# 1. **Loading activations**: Using `quick_load()` and `ActivationLoader`
# 2. **Statistical analysis**: Computing means, stds, norms across layers
# 3. **Visualization**: Heatmaps, similarity matrices, evolution plots
# 4. **Layer comparison**: Tracking how activations change through the network
# 5. **Token analysis**: Following individual tokens through layers
# 
# Key observations from activation analysis can reveal:
# - Which layers are most active for certain inputs
# - How information flows through the network
# - Potential bottlenecks or redundancies
# - Patterns that correlate with specific behaviors

print("\n" + "="*60)
print("Analysis complete! Key takeaways:")
print("- Activations can be efficiently loaded and analyzed")
print("- Compression preserves important patterns")
print("- Layer-wise analysis reveals network dynamics")
print("- Token tracking shows information flow")
print("="*60)