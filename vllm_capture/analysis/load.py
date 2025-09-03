"""
Helper functions for loading and analyzing captured activations.

This module provides utilities to:
- Load activation manifests and files
- Reconstruct activations from compressed format
- Convert between numpy and torch tensors
- Analyze activation patterns
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_activation_manifest(manifest_path: Union[str, Path]) -> Dict:
    """
    Load an activation manifest JSON file.
    
    Args:
        manifest_path: Path to manifest.json file
        
    Returns:
        Dictionary containing manifest data with request info and file paths
    """
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_activation_file(
    npz_path: Union[str, Path],
    return_torch: bool = False,
    device: Optional[str] = None
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """
    Load a single activation .npz file.
    
    Args:
        npz_path: Path to .npz file
        return_torch: If True, convert to PyTorch tensors
        device: Device to place tensors on (if return_torch=True)
        
    Returns:
        Dictionary with activation data (q, scale, zero for compressed,
        or indices, values, scale, zero for sparse)
    """
    data = np.load(npz_path)
    
    result = {}
    for key in data.keys():
        array = data[key]
        
        if return_torch and TORCH_AVAILABLE:
            tensor = torch.from_numpy(array)
            if device:
                tensor = tensor.to(device)
            result[key] = tensor
        else:
            result[key] = array
    
    return result


def reconstruct_activations(
    compressed_data: Dict[str, Union[np.ndarray, 'torch.Tensor']],
    mode: str = 'rp8'
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Reconstruct full activations from compressed format.
    
    Args:
        compressed_data: Dictionary with compressed activation data
        mode: Compression mode ('rp8', 'full8', 'topk8')
        
    Returns:
        Reconstructed activation tensor
    """
    if mode in ['rp8', 'full8']:
        # Dense compressed format
        q = compressed_data['q']
        scale = compressed_data['scale']
        zero = compressed_data['zero']
        
        if TORCH_AVAILABLE and isinstance(q, torch.Tensor):
            # PyTorch reconstruction
            reconstructed = q.float() * scale + zero
        else:
            # NumPy reconstruction
            reconstructed = q.astype(np.float32) * scale + zero
            
    elif mode == 'topk8':
        # Sparse format reconstruction
        indices = compressed_data['indices']
        values = compressed_data['values']
        scale = compressed_data['scale']
        zero = compressed_data['zero']
        
        # Reconstruct values
        reconstructed_values = values.astype(np.float32) * scale + zero
        
        # Create sparse tensor (simplified - would need full dimension info)
        # This is a placeholder - actual implementation would need hidden_size
        raise NotImplementedError("Top-k sparse reconstruction requires hidden_size info")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return reconstructed


def load_sequence_activations(
    output_dir: Union[str, Path],
    request_id: str,
    sequence_id: Union[int, str] = 0,
    layers: Optional[List[int]] = None,
    return_torch: bool = False,
    device: Optional[str] = None
) -> Dict[int, Dict]:
    """
    Load all activations for a specific sequence.
    
    Args:
        output_dir: Base output directory
        request_id: Request ID
        sequence_id: Sequence ID within request
        layers: Specific layers to load (None = all)
        return_torch: Convert to PyTorch tensors
        device: Device for tensors
        
    Returns:
        Dictionary mapping layer_id to activation data
    """
    output_dir = Path(output_dir)
    request_dir = output_dir / request_id
    
    if not request_dir.exists():
        raise FileNotFoundError(f"Request directory not found: {request_dir}")
    
    activations = {}
    
    # Find all files for this sequence
    pattern = f"seq{sequence_id}_layer*.npz"
    for npz_file in request_dir.glob(pattern):
        # Extract layer number from filename
        # Format: seqX_layerY_rankZ.npz
        parts = npz_file.stem.split('_')
        layer_id = int(parts[1].replace('layer', ''))
        
        if layers is None or layer_id in layers:
            activations[layer_id] = load_activation_file(
                npz_file, return_torch, device
            )
    
    return activations


class ActivationLoader:
    """
    High-level interface for loading and analyzing activations.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        mode: str = 'rp8',
        device: Optional[str] = None
    ):
        """
        Initialize activation loader.
        
        Args:
            output_dir: Base directory with activation files
            mode: Compression mode used
            device: Device for PyTorch tensors
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.device = device
        
    def list_requests(self) -> List[str]:
        """List all request IDs with saved activations."""
        requests = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith('req_'):
                requests.append(path.name)
        return sorted(requests)
    
    def load_request(
        self,
        request_id: str,
        reconstruct: bool = False
    ) -> Dict[int, Dict[int, Union[np.ndarray, 'torch.Tensor']]]:
        """
        Load all activations for a request.
        
        Args:
            request_id: Request ID to load
            reconstruct: If True, reconstruct from compressed format
            
        Returns:
            Nested dict: {sequence_id: {layer_id: activation_data}}
        """
        request_dir = self.output_dir / request_id
        if not request_dir.exists():
            raise FileNotFoundError(f"Request not found: {request_id}")
        
        # Group files by sequence
        sequences = {}
        for npz_file in request_dir.glob("*.npz"):
            # Parse filename: seqX_layerY_rankZ.npz
            parts = npz_file.stem.split('_')
            seq_id = int(parts[0].replace('seq', ''))
            layer_id = int(parts[1].replace('layer', ''))
            
            if seq_id not in sequences:
                sequences[seq_id] = {}
            
            data = load_activation_file(
                npz_file,
                return_torch=TORCH_AVAILABLE and self.device is not None,
                device=self.device
            )
            
            if reconstruct:
                data = reconstruct_activations(data, self.mode)
            
            sequences[seq_id][layer_id] = data
        
        return sequences
    
    def compute_similarity(
        self,
        acts1: Union[np.ndarray, 'torch.Tensor'],
        acts2: Union[np.ndarray, 'torch.Tensor'],
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two activation tensors.
        
        Args:
            acts1, acts2: Activation tensors to compare
            metric: Similarity metric ('cosine', 'l2', 'correlation')
            
        Returns:
            Similarity score
        """
        if TORCH_AVAILABLE and isinstance(acts1, torch.Tensor):
            # PyTorch computation
            acts1_flat = acts1.flatten()
            acts2_flat = acts2.flatten()
            
            if metric == 'cosine':
                similarity = torch.nn.functional.cosine_similarity(
                    acts1_flat.unsqueeze(0),
                    acts2_flat.unsqueeze(0)
                ).item()
            elif metric == 'l2':
                similarity = -torch.norm(acts1_flat - acts2_flat).item()
            elif metric == 'correlation':
                # Pearson correlation
                acts1_centered = acts1_flat - acts1_flat.mean()
                acts2_centered = acts2_flat - acts2_flat.mean()
                similarity = (acts1_centered @ acts2_centered) / (
                    acts1_centered.norm() * acts2_centered.norm()
                )
                similarity = similarity.item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            # NumPy computation
            acts1_flat = acts1.flatten()
            acts2_flat = acts2.flatten()
            
            if metric == 'cosine':
                similarity = np.dot(acts1_flat, acts2_flat) / (
                    np.linalg.norm(acts1_flat) * np.linalg.norm(acts2_flat)
                )
            elif metric == 'l2':
                similarity = -np.linalg.norm(acts1_flat - acts2_flat)
            elif metric == 'correlation':
                similarity = np.corrcoef(acts1_flat, acts2_flat)[0, 1]
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return float(similarity)
    
    def create_token_timeline(
        self,
        sequence_activations: Dict[int, Dict],
        layer: int = 0
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Create a timeline of activations across tokens for a specific layer.
        
        Args:
            sequence_activations: Activations for one sequence
            layer: Layer index to analyze
            
        Returns:
            Array/tensor of shape (n_tokens, activation_dim)
        """
        if layer not in sequence_activations:
            raise ValueError(f"Layer {layer} not found in activations")
        
        layer_data = sequence_activations[layer]
        
        # For compressed data, reconstruct first
        if 'q' in layer_data:
            timeline = reconstruct_activations(layer_data, self.mode)
        else:
            timeline = layer_data
        
        return timeline
    
    def compute_layer_statistics(
        self,
        sequence_activations: Dict[int, Dict]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for each layer's activations.
        
        Args:
            sequence_activations: Activations for one sequence
            
        Returns:
            Dictionary mapping layer_id to statistics
        """
        stats = {}
        
        for layer_id, layer_data in sequence_activations.items():
            # Reconstruct if compressed
            if 'q' in layer_data:
                acts = reconstruct_activations(layer_data, self.mode)
            else:
                acts = layer_data
            
            if TORCH_AVAILABLE and isinstance(acts, torch.Tensor):
                stats[layer_id] = {
                    'mean': acts.mean().item(),
                    'std': acts.std().item(),
                    'min': acts.min().item(),
                    'max': acts.max().item(),
                    'norm': acts.norm().item(),
                }
            else:
                stats[layer_id] = {
                    'mean': float(np.mean(acts)),
                    'std': float(np.std(acts)),
                    'min': float(np.min(acts)),
                    'max': float(np.max(acts)),
                    'norm': float(np.linalg.norm(acts)),
                }
        
        return stats


# Convenience functions for quick analysis
def quick_load(
    output_dir: str,
    request_id: Optional[str] = None,
    layer: int = 0,
    reconstruct: bool = True
) -> Union[np.ndarray, Dict]:
    """
    Quick load function for interactive use.
    
    Args:
        output_dir: Base output directory
        request_id: Specific request (None = first found)
        layer: Layer to load
        reconstruct: Whether to reconstruct from compressed
        
    Returns:
        Activation data or dict of all data
    """
    loader = ActivationLoader(output_dir)
    
    if request_id is None:
        requests = loader.list_requests()
        if not requests:
            raise ValueError("No requests found in output directory")
        request_id = requests[0]
        print(f"Loading first request: {request_id}")
    
    sequences = loader.load_request(request_id, reconstruct=reconstruct)
    
    # Return first sequence, specified layer
    if sequences:
        seq_id = min(sequences.keys())
        if layer in sequences[seq_id]:
            return sequences[seq_id][layer]
        else:
            print(f"Layer {layer} not found, returning all layers")
            return sequences[seq_id]
    
    return sequences


def compare_tokens(
    acts1: Union[np.ndarray, 'torch.Tensor'],
    acts2: Union[np.ndarray, 'torch.Tensor'],
    token_idx1: int = 0,
    token_idx2: int = 0,
    metric: str = 'cosine'
) -> float:
    """
    Compare activations between specific tokens.
    
    Args:
        acts1, acts2: Activation tensors (tokens × dims)
        token_idx1, token_idx2: Token indices to compare
        metric: Similarity metric
        
    Returns:
        Similarity score
    """
    if len(acts1.shape) != 2 or len(acts2.shape) != 2:
        raise ValueError("Expected 2D tensors (tokens × dims)")
    
    token1 = acts1[token_idx1]
    token2 = acts2[token_idx2]
    
    loader = ActivationLoader('.')  # Dummy for method access
    return loader.compute_similarity(token1, token2, metric)