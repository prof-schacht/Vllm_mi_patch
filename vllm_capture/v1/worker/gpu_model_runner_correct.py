"""
Corrected GPU Model Runner for activation capture based on GPT5 review.
This implementation:
- NO PyTorch hooks (keeps CUDA graphs enabled)
- Uses intermediate_tensors pattern that vLLM already has
- Captures ALL tokens (prefill and decode)
- Works with the runner level, not fighting the architecture
"""

import os
import uuid
from typing import Dict, List, Optional, Any

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from ..._compat import VllmConfig
from vllm.sequence import IntermediateTensors

from ...activations.capture import (
    ActivationCaptureConfig,
    ActivationCollector,
    CaptureContext
)


class IntermediateTensorCollector:
    """
    Shim object that models call to report layer outputs.
    This is the key innovation from the review - we pass this to the model
    and it calls our add() method for each layer, no hooks needed!
    """
    def __init__(self, capture_context: CaptureContext):
        self.context = capture_context
        
    def add(self, layer_id: int, tensor: torch.Tensor):
        """Called by model for each layer output."""
        print(f"[DEBUG] IntermediateTensorCollector.add called: layer_id={layer_id}, tensor shape={tensor.shape}")
        self.context.capture_layer(layer_id, tensor)


class GPUModelRunnerCorrect(GPUModelRunner):
    """
    Corrected model runner that captures activations WITHOUT hooks.
    Based on GPT5 review recommendations.
    """
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        
        # Initialize activation capture configuration
        self._act_cfg = ActivationCaptureConfig.from_env()
        
        # Override from config if present
        if hasattr(vllm_config, 'activation'):
            ac = vllm_config.activation
            if hasattr(ac, 'enabled'):
                self._act_cfg.enabled = ac.enabled
            if hasattr(ac, 'mode'):
                self._act_cfg.mode = ac.mode
            if hasattr(ac, 'rp_k'):
                self._act_cfg.rp_k = ac.rp_k
            if hasattr(ac, 'return_mode'):
                self._act_cfg.return_mode = ac.return_mode
            if hasattr(ac, 'output_dir'):
                self._act_cfg.output_dir = ac.output_dir
        
        self._act_collector: Optional[ActivationCollector] = None
        self._capture_context: Optional[CaptureContext] = None
        self._last_manifest: Optional[Dict[str, Any]] = None
        
    def _resolve_hidden_size(self, model) -> Optional[int]:
        """Robustly resolve model hidden size from multiple sources."""
        # 1) Manual override via environment variable
        env = os.environ.get("VLLM_ACT_HIDDEN_SIZE")
        if env and env.isdigit():
            return int(env)
        
        # 2) Try common fields on vLLM model_config, HF config, and the model itself
        candidates = ["hidden_size", "d_model", "n_embd", "hidden_dim", 
                     "model_dim", "dim", "embed_dim"]
        
        for src in [getattr(self, "model_config", None), 
                   getattr(model, "config", None), 
                   model]:
            if src is None:
                continue
            for name in candidates:
                v = getattr(src, name, None)
                if isinstance(v, int) and v > 0:
                    return int(v)
        
        # 3) Try HF config dict (works even if attrs aren't set)
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "to_dict"):
            d = cfg.to_dict()
            for name in candidates:
                v = d.get(name)
                if isinstance(v, int) and v > 0:
                    return int(v)
        
        # 4) Infer from common module shapes
        def _get(obj, path):
            cur = obj
            for p in path.split("."):
                if not hasattr(cur, p):
                    return None
                cur = getattr(cur, p)
            return cur
        
        # Check embeddings
        for p in ["model.embed_tokens", "transformer.wte", "model.embed_in"]:
            emb = _get(model, p)
            if emb is not None and hasattr(emb, "embedding_dim"):
                return int(emb.embedding_dim)
        
        # Check lm_head
        for p in ["lm_head", "model.lm_head"]:
            head = _get(model, p)
            if head is not None and hasattr(head, "in_features"):
                return int(head.in_features)
        
        # Check attention q_proj in_features
        for p in ["model.layers.0.self_attn.q_proj", 
                 "model.model.layers.0.self_attn.q_proj",
                 "transformer.h.0.attn.c_attn"]:
            q = _get(model, p)
            if q is not None and hasattr(q, "in_features"):
                return int(q.in_features)
        
        return None
    
    def load_model(self, **kwargs):
        """Load model and initialize activation collector if enabled."""
        model = super().load_model(**kwargs)
        
        # Initialize collector if capture is enabled
        if self._act_cfg.enabled:
            # Get model hidden size using robust resolver
            hidden_size = self._resolve_hidden_size(model)
            
            if hidden_size:
                # Get TP info if available
                tp_world_size = getattr(self, 'tp_world_size', 1)
                tp_rank = getattr(self, 'tp_rank', 0)
                
                self._act_collector = ActivationCollector(
                    self._act_cfg,
                    hidden_size=hidden_size,
                    device=self.device,
                    tp_world_size=tp_world_size,
                    tp_rank=tp_rank
                )
                
                self._capture_context = CaptureContext(self._act_collector)
                
                print(f"[GPUModelRunnerCorrect] Activation capture ENABLED")
                print(f"  Mode: {self._act_cfg.mode}")
                print(f"  Hidden size: {hidden_size}")
                print(f"  Output dir: {self._act_cfg.output_dir}")
            else:
                print("[GPUModelRunnerCorrect] Warning: Could not determine hidden size, capture disabled")
                self._act_cfg.enabled = False
        
        return model
    
    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> Any:
        """
        Execute model with activation capture.
        
        This is the KEY METHOD where we integrate cleanly with vLLM.
        We check if the model supports intermediate_tensors and if so,
        pass our collector. No hooks, no hacks!
        """
        
        if not self._act_cfg.enabled or not self._capture_context:
            # Capture disabled, run normally
            return super().execute_model(scheduler_output, intermediate_tensors)
        
        # Extract batch information - use simple integer IDs
        batch_seq_ids = []
        prompt_lens = []
        is_prefill = False
        seq_counter = 0
        
        # Determine if this is prefill or decode based on scheduler output
        if hasattr(scheduler_output, 'scheduled_new_reqs'):
            # v1 uses scheduled_new_reqs for new sequences (prefill)
            if scheduler_output.scheduled_new_reqs:
                is_prefill = True
                for req_data in scheduler_output.scheduled_new_reqs:
                    # Use simple integer IDs
                    batch_seq_ids.append(seq_counter)
                    seq_counter += 1
                    # The attribute name varies across vLLM versions
                    # Try different possible attribute names
                    prompt_len = None
                    for attr in ['prompt_token_ids_len', 'num_prompt_tokens', 'prompt_len']:
                        if hasattr(req_data, attr):
                            prompt_len = getattr(req_data, attr)
                            break
                    
                    # If not found, try to get from scheduler output
                    if prompt_len is None and hasattr(scheduler_output, 'num_scheduled_tokens'):
                        prompt_len = scheduler_output.num_scheduled_tokens.get(req_data.req_id, 0)
                    
                    if prompt_len is None:
                        prompt_len = 0  # Default fallback
                    
                    prompt_lens.append(prompt_len)
        
        if hasattr(scheduler_output, 'scheduled_cached_reqs'):
            # Decode phase - continuing sequences
            if scheduler_output.scheduled_cached_reqs.req_ids:
                for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                    # Use simple integer IDs
                    batch_seq_ids.append(seq_counter)
                    seq_counter += 1
                    prompt_lens.append(0)  # Not needed for decode
        
        # Initialize request if this is the first prefill
        if is_prefill and batch_seq_ids and prompt_lens:
            request_id = f"req_{uuid.uuid4().hex[:8]}"
            max_len = getattr(self.model_config, 'max_model_len', 2048)
            self._act_collector.begin_request(
                request_id=request_id,
                seq_ids=batch_seq_ids,  # Already integers
                prompt_lens=prompt_lens,
                max_len=max_len
            )
        
        # Update capture context with batch info
        self._capture_context.set_batch_info(batch_seq_ids, is_prefill)
        
        # Check if model supports intermediate_tensors parameter
        model_forward = getattr(self.model, 'forward', None)
        print(f"[DEBUG GPU_RUNNER] model type: {type(self.model)}, has forward: {model_forward is not None}")
        if model_forward:
            import inspect
            sig = inspect.signature(model_forward)
            supports_intermediate = 'intermediate_tensors' in sig.parameters
            print(f"[DEBUG GPU_RUNNER] Model forward signature params: {list(sig.parameters.keys())}")
            print(f"[DEBUG GPU_RUNNER] Supports intermediate_tensors: {supports_intermediate}")
        else:
            supports_intermediate = False
            print(f"[DEBUG GPU_RUNNER] No forward method found")
        
        if supports_intermediate:
            # Create our collector
            collector_shim = IntermediateTensorCollector(self._capture_context)
            print(f"[DEBUG GPU_RUNNER] Created collector_shim, will inject via wrapper")
            
            # CRITICAL FIX: Inject collector via model.forward wrapper
            # This ensures it reaches the model even when parent doesn't pass it
            original_forward = self.model.forward
            
            def forward_wrapper(*args, **kwargs):
                # Check if intermediate_tensors was passed
                intermediate_tensors = kwargs.get('intermediate_tensors', None)
                
                if intermediate_tensors is None:
                    # Parent didn't pass one (common for non-PP), inject ours
                    print(f"[DEBUG WRAPPER] Injecting collector_shim as intermediate_tensors")
                    kwargs['intermediate_tensors'] = collector_shim
                else:
                    print(f"[DEBUG WRAPPER] Parent already provided intermediate_tensors: {type(intermediate_tensors)}")
                
                # Call original forward with potentially modified kwargs
                return original_forward(*args, **kwargs)
            
            # Temporarily replace model.forward
            self.model.forward = forward_wrapper
            
            try:
                # Call parent's execute_model (it won't pass intermediate_tensors for non-PP)
                print(f"[DEBUG GPU_RUNNER] Calling parent execute_model with wrapper in place")
                output = super().execute_model(scheduler_output, intermediate_tensors=None)
            finally:
                # Always restore original forward
                self.model.forward = original_forward
                print(f"[DEBUG GPU_RUNNER] Restored original model.forward")
        else:
            # Model doesn't support intermediate_tensors yet
            # Run normally but warn user
            if self._act_cfg.enabled:
                print("[GPUModelRunnerCorrect] Warning: Model doesn't support intermediate_tensors")
                print("  Activation capture requires model modification (see docs)")
            output = super().execute_model(scheduler_output, intermediate_tensors=None)
        
        # Finalize if this was the last decode step
        # (In production, detect this from scheduler_output.finished_req_ids)
        if hasattr(scheduler_output, 'finished_req_ids') and scheduler_output.finished_req_ids:
            self._last_manifest = self._act_collector.finalize()
        
        return output
    
    def get_activation_manifest(self) -> Optional[Dict[str, Any]]:
        """Get the last activation manifest for completed request."""
        return self._last_manifest