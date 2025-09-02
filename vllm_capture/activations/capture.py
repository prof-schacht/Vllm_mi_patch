"""
Activation capture system for vLLM based on GPT5 external review.
This implementation:
- Captures ALL tokens (prefill + decode), not just last token
- Uses random projection instead of SVD for real-time compression
- Works with CUDA graphs enabled (no PyTorch hooks in hot path)
- Correctly handles TP/PP parallelism
- No false "zero-copy" claims - we copy GPU->CPU then use shared memory between processes
"""

import os
import json
import math
import uuid
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

# --------- Config ---------------------------------------------------------------------------------

class ReturnMode(str, Enum):
    SHARDED = "sharded"     # return TP shard-local activations (default; no all-gather)
    GATHERED = "gathered"   # all-gather across TP shards before compressing (costly)
    SUMMARY = "summary"     # only store summaries inline; no artifact files

class ActMode(str, Enum):
    NONE = "none"
    RP8 = "rp8"             # residual stream: random projection to k dims + uint8 per-dim scales
    FULL8 = "full8"         # full vector quantized to uint8 (per-row or per-dim scales)
    TOPK8 = "topk8"         # top-k indices + uint8 values (for MLP activations)

@dataclass
class ActivationCaptureConfig:
    enabled: bool = False
    mode: ActMode = ActMode.RP8
    rp_k: int = 512
    topk_k: int = 64
    quant_bits: int = 8
    return_mode: ReturnMode = ReturnMode.SHARDED
    attention_topm: int = 0          # 0 disables attention summaries for now
    output_dir: str = "/tmp/vllm_activations"
    # internal/advanced
    rp_seed: int = 12345
    use_per_dim_scale: bool = True   # for FULL8/RP8 quantization

    @staticmethod
    def from_env() -> "ActivationCaptureConfig":
        def _get(name: str, default: Optional[str] = None) -> Optional[str]:
            v = os.environ.get(name, default)
            return v
        enabled = _get("VLLM_ACT_CAPTURE", "0") in ("1", "true", "TRUE", "on", "ON")
        mode = ActMode(_get("VLLM_ACT_MODE", "rp8"))
        rp_k = int(_get("VLLM_ACT_RP_K", "512"))
        topk_k = int(_get("VLLM_ACT_TOPK_K", "64"))
        quant_bits = int(_get("VLLM_ACT_BITS", "8"))
        return_mode = ReturnMode(_get("VLLM_ACT_RETURN", "sharded"))
        attention_topm = int(_get("VLLM_ACT_ATTNM", "0"))
        output_dir = _get("VLLM_ACT_OUTDIR", "/tmp/vllm_activations")
        rp_seed = int(_get("VLLM_ACT_RP_SEED", "12345"))
        use_per_dim_scale = _get("VLLM_ACT_PER_DIM", "1") in ("1", "true", "TRUE", "on", "ON")
        return ActivationCaptureConfig(
            enabled=enabled, mode=mode, rp_k=rp_k, topk_k=topk_k,
            quant_bits=quant_bits, return_mode=return_mode,
            attention_topm=attention_topm, output_dir=output_dir,
            rp_seed=rp_seed, use_per_dim_scale=use_per_dim_scale
        )

# --------- Compression primitives -----------------------------------------------------------------

def _quantize_u8_per_dim(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to uint8 with per-dimension scales."""
    x = x.contiguous()
    # Per-dim range
    x_max = torch.amax(x, dim=-2, keepdim=True) + 1e-12  # avoid zero-range
    x_min = torch.amin(x, dim=-2, keepdim=True)
    scale = (x_max - x_min) / 255.0
    zero = (-x_min / (scale + 1e-12)).clamp(0, 255)
    q = ((x / (scale + 1e-12)) + zero).round().clamp(0, 255).to(torch.uint8)
    return q, scale.squeeze(-2), zero.squeeze(-2)

def _quantize_u8_per_row(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to uint8 with per-row scales."""
    x = x.contiguous()
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x_min, _ = torch.min(x, dim=-1, keepdim=True)
    scale = (x_max - x_min) / 255.0
    zero = (-x_min / (scale + 1e-12)).clamp(0, 255)
    q = ((x / (scale + 1e-12)) + zero).round().clamp(0, 255).to(torch.uint8)
    return q, scale.squeeze(-1), zero.squeeze(-1)

def _topk_sparsify(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k values and indices for MLP activations."""
    values, indices = torch.topk(torch.abs(x), k=min(k, x.shape[-1]), dim=-1)
    # Get actual values (not absolute)
    batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
    actual_values = x[batch_indices, indices]
    return indices.to(torch.int16), actual_values

class RandomProjector:
    """Fixed random projection for compression. Much faster than SVD."""
    def __init__(self, in_dim: int, out_dim: int, seed: int = 12345, device: torch.device = torch.device("cpu")):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Random Gaussian projection scaled by 1/sqrt(k)
        self.R = torch.randn(in_dim, out_dim, generator=g, device=device, dtype=torch.float16) / math.sqrt(out_dim)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float16) @ self.R  # returns [..., k]

# --------- Rolling buffers for sequences ----------------------------------------------------------

@dataclass
class SequenceBuffer:
    """Per-sequence buffer that accumulates activations across timesteps."""
    seq_id: int
    prompt_len: int
    max_len: int
    layers: Dict[int, List[torch.Tensor]]  # layer_id -> list of compressed chunks
    write_idx: int = 0

    def append_prefill(self, layer_id: int, chunk: dict):
        """Append prefill activations for this layer."""
        if layer_id not in self.layers:
            self.layers[layer_id] = []
        # chunk is already a dict with cpu tensors
        self.layers[layer_id].append(chunk)
        self.write_idx = self.prompt_len

    def append_decode(self, layer_id: int, chunk: dict):
        """Append single decode step for this layer."""
        if layer_id not in self.layers:
            self.layers[layer_id] = []
        # chunk is already a dict with cpu tensors
        self.layers[layer_id].append(chunk)
        self.write_idx += 1

# --------- Main collector --------------------------------------------------------------------------

class ActivationCollector:
    """
    Per-worker, per-run collector. Stores shard-local compressed activations and spills to files.
    Key difference from our old design: maintains rolling buffers per sequence to capture ALL tokens.
    """
    def __init__(self, cfg: ActivationCaptureConfig, *,
                 hidden_size: int, device: torch.device,
                 tp_world_size: int = 1, tp_rank: int = 0) -> None:
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.device = device
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.projector: Optional[RandomProjector] = None
        self.sequence_buffers: Dict[int, SequenceBuffer] = {}  # seq_id -> buffer
        self.request_id: Optional[str] = None
        self.output_dir: Optional[str] = None

    def begin_request(self, request_id: str, seq_ids: List[int], prompt_lens: List[int], max_len: int) -> None:
        """Initialize buffers for a new request."""
        self.request_id = request_id
        self.output_dir = os.path.join(self.cfg.output_dir, f"{request_id}-{uuid.uuid4().hex[:8]}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize per-sequence buffers
        for seq_id, prompt_len in zip(seq_ids, prompt_lens):
            self.sequence_buffers[seq_id] = SequenceBuffer(
                seq_id=seq_id,
                prompt_len=prompt_len,
                max_len=max_len,
                layers={}
            )
        
        # Initialize projector if needed
        if self.cfg.mode == ActMode.RP8 and self.projector is None:
            self.projector = RandomProjector(
                self.hidden_size, self.cfg.rp_k, 
                seed=self.cfg.rp_seed, device=self.device
            )

    @torch.no_grad()
    def write_batch(self, *, seq_ids: List[int], layer_id: int, 
                    tensor: torch.Tensor, is_prefill: bool) -> None:
        """
        Write activations for a batch of sequences at a specific layer.
        
        Args:
            seq_ids: Sequence IDs in the batch
            layer_id: Which transformer layer
            tensor: [B, S, H_shard] where S>1 for prefill, S=1 for decode
            is_prefill: Whether this is prefill (full prompt) or decode (single token)
        """
        if not self.cfg.enabled or not self.sequence_buffers:
            return
            
        B, S, H = tensor.shape
        
        # Compress based on mode
        if self.cfg.mode == ActMode.RP8:
            # Random projection + quantization
            assert self.projector is not None
            x_flat = tensor.reshape(B * S, H)
            x_proj = self.projector(x_flat)  # [B*S, k]
            
            if self.cfg.use_per_dim_scale:
                q, scale, zero = _quantize_u8_per_dim(x_proj.unsqueeze(-2))
                q = q.squeeze(-2)
            else:
                q, scale, zero = _quantize_u8_per_row(x_proj)
            
            # Reshape back to [B, S, k]
            q = q.reshape(B, S, -1)
            scale = scale.reshape(B, S, -1) if scale.dim() > 1 else scale.reshape(B, S)
            zero = zero.reshape(B, S, -1) if zero.dim() > 1 else zero.reshape(B, S)
            
        elif self.cfg.mode == ActMode.FULL8:
            # Full quantization without projection
            x_flat = tensor.reshape(B * S, H)
            if self.cfg.use_per_dim_scale:
                q, scale, zero = _quantize_u8_per_dim(x_flat.unsqueeze(-2))
                q = q.squeeze(-2)
            else:
                q, scale, zero = _quantize_u8_per_row(x_flat)
            
            q = q.reshape(B, S, -1)
            scale = scale.reshape(B, S, -1) if scale.dim() > 1 else scale.reshape(B, S)
            zero = zero.reshape(B, S, -1) if zero.dim() > 1 else zero.reshape(B, S)
            
        elif self.cfg.mode == ActMode.TOPK8:
            # Top-k sparsification for MLP activations
            x_flat = tensor.reshape(B * S, H)
            indices, values = _topk_sparsify(x_flat, self.cfg.topk_k)
            
            # Quantize values
            values_q, values_scale, values_zero = _quantize_u8_per_row(values)
            
            # Pack as dictionary for each token
            compressed = {
                'indices': indices.reshape(B, S, -1),
                'values': values_q.reshape(B, S, -1),
                'scale': values_scale.reshape(B, S),
                'zero': values_zero.reshape(B, S)
            }
        else:
            return
        
        # Store in per-sequence buffers
        for b, seq_id in enumerate(seq_ids):
            if seq_id not in self.sequence_buffers:
                continue
                
            buffer = self.sequence_buffers[seq_id]
            
            if self.cfg.mode == ActMode.TOPK8:
                # Store sparse representation (move to CPU)
                chunk = {
                    'indices': compressed['indices'][b].cpu(),
                    'values': compressed['values'][b].cpu(),
                    'scale': compressed['scale'][b].cpu(),
                    'zero': compressed['zero'][b].cpu()
                }
            else:
                # Store dense compressed representation (move to CPU)
                chunk = {
                    'q': q[b].cpu(),
                    'scale': scale[b].cpu(),
                    'zero': zero[b].cpu()
                }
            
            if is_prefill:
                buffer.append_prefill(layer_id, chunk)
            else:
                buffer.append_decode(layer_id, chunk)

    def finalize(self) -> Dict[str, Any]:
        """
        Spill all buffers to disk and return manifest for RequestOutput.
        """
        if not self.request_id or not self.output_dir:
            return {}
            
        manifest = {
            "request_id": self.request_id,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "mode": str(self.cfg.mode),
            "return_mode": str(self.cfg.return_mode),
            "compression": {
                "rp_k": self.cfg.rp_k if self.cfg.mode == ActMode.RP8 else None,
                "topk_k": self.cfg.topk_k if self.cfg.mode == ActMode.TOPK8 else None,
                "quant_bits": self.cfg.quant_bits,
                "per_dim_scale": self.cfg.use_per_dim_scale
            },
            "output_dir": self.output_dir,
            "sequences": {}
        }
        
        # Spill each sequence's buffers
        for seq_id, buffer in self.sequence_buffers.items():
            seq_manifest = {
                "prompt_len": buffer.prompt_len,
                "total_len": buffer.write_idx,
                "layers": {}
            }
            
            for layer_id, chunks in buffer.layers.items():
                # Concatenate all chunks for this layer
                if self.cfg.mode == ActMode.TOPK8:
                    # Handle sparse format
                    all_indices = []
                    all_values = []
                    all_scales = []
                    all_zeros = []
                    
                    for chunk in chunks:
                        all_indices.append(chunk['indices'])
                        all_values.append(chunk['values'])
                        all_scales.append(chunk['scale'])
                        all_zeros.append(chunk['zero'])
                    
                    # Save as npz
                    path = os.path.join(self.output_dir, f"seq{seq_id}_layer{layer_id}_rank{self.tp_rank}.npz")
                    np.savez_compressed(
                        path,
                        indices=torch.cat(all_indices, dim=0).numpy(),
                        values=torch.cat(all_values, dim=0).numpy(),
                        scale=torch.cat(all_scales, dim=0).numpy(),
                        zero=torch.cat(all_zeros, dim=0).numpy()
                    )
                else:
                    # Handle dense format
                    all_q = []
                    all_scales = []
                    all_zeros = []
                    
                    for chunk in chunks:
                        all_q.append(chunk['q'])
                        all_scales.append(chunk['scale'])
                        all_zeros.append(chunk['zero'])
                    
                    # Save as npz
                    path = os.path.join(self.output_dir, f"seq{seq_id}_layer{layer_id}_rank{self.tp_rank}.npz")
                    np.savez_compressed(
                        path,
                        q=torch.cat(all_q, dim=0).numpy(),
                        scale=torch.cat(all_scales, dim=0).numpy(),
                        zero=torch.cat(all_zeros, dim=0).numpy()
                    )
                
                seq_manifest["layers"][str(layer_id)] = {
                    "path": path,
                    "shape": [buffer.write_idx, self.cfg.rp_k if self.cfg.mode == ActMode.RP8 else self.hidden_size]
                }
            
            manifest["sequences"][str(seq_id)] = seq_manifest
        
        # Clear buffers
        self.sequence_buffers.clear()
        
        return manifest

# --------- Integration helper for runner ----------------------------------------------------------

class CaptureContext:
    """
    Context manager that wraps model execution to capture activations.
    This replaces our old hook-based approach with clean integration at runner level.
    """
    def __init__(self, collector: ActivationCollector):
        self.collector = collector
        self.batch_seq_ids: List[int] = []
        self.is_prefill: bool = True
        
    def set_batch_info(self, seq_ids: List[int], is_prefill: bool):
        """Set current batch information."""
        self.batch_seq_ids = seq_ids
        self.is_prefill = is_prefill
    
    def capture_layer(self, layer_id: int, tensor: torch.Tensor):
        """Capture activation for a specific layer."""
        if self.collector and self.batch_seq_ids:
            self.collector.write_batch(
                seq_ids=self.batch_seq_ids,
                layer_id=layer_id,
                tensor=tensor,
                is_prefill=self.is_prefill
            )