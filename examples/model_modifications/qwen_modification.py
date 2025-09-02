"""
Example of how to modify Qwen model to support intermediate_tensors.
Based on GPT5 review - this is the ONLY modification needed to the model!

The key insight: Instead of using PyTorch hooks, we add ONE line per layer
that calls intermediate_tensors.add(). This keeps CUDA graphs enabled and
captures ALL tokens (prefill and decode).
"""

# This shows the modification to vllm/model_executor/models/qwen2.py
# (or qwen3.py depending on your vLLM version)

"""
Original forward method in Qwen model:

def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    use_cache: bool = True,
    **kwargs,
):
    # ... input preparation ...
    
    for layer_id, block in enumerate(self.layers):
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[layer_id] if past_key_values else None,
            use_cache=use_cache,
        )
    
    # ... rest of forward ...
    return outputs
"""

# MODIFIED VERSION:
def forward_with_capture(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    use_cache: bool = True,
    intermediate_tensors: Optional[Any] = None,  # <-- ADD THIS PARAMETER
    **kwargs,
):
    """
    Modified forward that supports activation capture.
    The ONLY changes:
    1. Add intermediate_tensors parameter
    2. Add ONE line after each layer to report activations
    """
    
    # ... input preparation (unchanged) ...
    hidden_states = self.embed_tokens(input_ids)
    
    for layer_id, block in enumerate(self.layers):
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[layer_id] if past_key_values else None,
            use_cache=use_cache,
        )
        
        # ===== THIS IS THE ONLY ADDITION =====
        # Capture block output (residual stream after layer)
        if intermediate_tensors is not None and hasattr(intermediate_tensors, 'add'):
            try:
                intermediate_tensors.add(layer_id, hidden_states)
            except Exception:
                pass  # Fail silently if collector has issues
        # =====================================
    
    # ... rest of forward (unchanged) ...
    hidden_states = self.norm(hidden_states)
    logits = self.lm_head(hidden_states)
    
    return logits


# Example patch file for automated application:
QWEN_MODEL_PATCH = """
--- a/vllm/model_executor/models/qwen2.py
+++ b/vllm/model_executor/models/qwen2.py
@@ -245,6 +245,7 @@ class Qwen2Model(nn.Module):
         position_ids: Optional[torch.LongTensor] = None,
         past_key_values: Optional[Any] = None,
         use_cache: bool = True,
+        intermediate_tensors: Optional[Any] = None,
         **kwargs,
     ):
         hidden_states = self.embed_tokens(input_ids)
@@ -257,6 +258,11 @@ class Qwen2Model(nn.Module):
                 past_key_value=past_key_values[layer_id] if past_key_values else None,
                 use_cache=use_cache,
             )
+            # Activation capture tap
+            if intermediate_tensors is not None and hasattr(intermediate_tensors, 'add'):
+                try:
+                    intermediate_tensors.add(layer_id, hidden_states)
+                except Exception:
+                    pass
         
         hidden_states = self.norm(hidden_states)
         return hidden_states
"""


# For other models (Llama, Mistral, etc), the modification is identical:
# 1. Add intermediate_tensors parameter to forward()
# 2. Add intermediate_tensors.add(layer_id, hidden_states) after each layer

# That's it! No hooks, no complex modifications, just one line per layer.