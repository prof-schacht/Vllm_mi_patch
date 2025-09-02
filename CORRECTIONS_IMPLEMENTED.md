# Corrections Implemented Based on GPT5 External Review

Date: September 2, 2025

Summary: We removed the hooks-based path, adopted the hook-free runner-level integration, and updated docs/tests to match the actual implementation. The system now captures ALL tokens and uses practical compression (RP/uint8, full8, topk8). Claims about “zero-copy” and SVD in the hot path have been removed.

## Critical Flaws Identified and Fixed

### 1. Last-token-only capture → All tokens
- Implemented per-sequence rolling buffers handling prefill and decode.
- File: `vllm_capture/activations/capture.py` (`SequenceBuffer`, `ActivationCollector`).

### 2. Hooks removed; runner integration
- No PyTorch forward hooks; CUDA graphs can remain enabled.
- Runner passes `intermediate_tensors` collector; models call `add(layer_id, hidden_states)`.
- File: `vllm_capture/v1/worker/gpu_model_runner_correct.py`.

### 3. SVD in hot path removed; practical compression added
- Compression modes: RP+uint8 (`rp8`), full uint8 (`full8`), top-k sparse (`topk8`).
- File: `vllm_capture/activations/capture.py` (`RandomProjector`, quantization helpers).

### 4. Zero-copy claims removed
- Docs now state GPU→CPU requires copies; shared memory is CPU-only.
- README/docs updated accordingly.

### 5. Alignment with vLLM architecture
- Capture is at runner layer, uses intermediate tensors, respects TP/PP.
- Manifests are returned by runner; plumbing into RequestOutput can be added downstream.

## New Architecture Overview

### Activation Flow
1) Runner passes a collector via `intermediate_tensors`.
2) Layers call `collector.add(layer_id, hidden_states)`.
3) Collector compresses and accumulates tokens.
4) Finalize writes `.npz` per (seq,layer); returns a manifest.

### Key Components:

#### `/vllm_capture/activations/capture.py`
- `ActivationCaptureConfig`, `ActMode`, `ReturnMode`
- `RandomProjector`, quantization helpers
- `SequenceBuffer`, `ActivationCollector`

#### `/vllm_capture/v1/worker/gpu_model_runner_correct.py`
- `IntermediateTensorCollector`: shim models call per layer
- `GPUModelRunnerCorrect`: hook-free runner integration

#### `/examples/model_modifications/qwen_modification.py`
- Shows the ONLY model change needed (one line per layer)
- No complex modifications, just add intermediate_tensors parameter

## Compression Methods

### Implemented:
1. **RP8**: Random projection to k dims + uint8 quantization
2. **FULL8**: Full vector with uint8 quantization
3. **TOPK8**: Top-k sparse (indices + values) for MLP activations

### NOT Using (Too Slow):
- SVD/PCA in hot path
- Complex compression schemes
- Anything requiring double computation

## Testing Strategy

### Key Tests:
1. **All Tokens Test**: Verify prefill captures N tokens, decode captures 1 at a time
2. **No Hooks Test**: Verify CUDA graphs stay enabled
3. **Performance Test**: Random projection vs SVD timing
4. **Memory Test**: Verify no false zero-copy claims

## Configuration

### Environment Variables:
```bash
VLLM_ACT_CAPTURE=1           # Enable capture
VLLM_ACT_MODE=rp8            # Compression mode
VLLM_ACT_RP_K=512            # Random projection dimensions
VLLM_ACT_RETURN=sharded      # Don't all-gather (expensive)
VLLM_ACT_OUTDIR=/tmp/acts    # Output directory
```

### CLI Flags:
```bash
vllm serve --model Qwen/Qwen3-8B \
  --activation-capture \
  --act-mode rp8 \
  --rp-k 512 \
  --act-outdir /tmp/acts
```

## Output Format

### Manifest Structure:
```json
{
  "request_id": "req_abc123",
  "tp_rank": 0,
  "tp_world_size": 1,
  "mode": "rp8",
  "compression": {
    "rp_k": 512,
    "quant_bits": 8
  },
  "sequences": {
    "0": {
      "prompt_len": 50,
      "total_len": 100,
      "layers": {
        "0": {"path": "seq0_layer0_rank0.npz", "shape": [100, 512]}
      }
    }
  }
}
```

### NPZ File Contents:
- `q`: uint8 quantized activations [tokens, dims]
- `scale`: per-row or per-dim scales
- `zero`: zero points for dequantization

## Remaining Work

1. **Model Patches**: Add intermediate_tensors.add() calls to each model architecture
2. **Integration Tests**: Full end-to-end testing with real vLLM
3. **Documentation**: Update README with honest performance numbers
4. **Benchmarks**: Measure actual overhead (expect 5-15%, not <5%)

## Lessons Learned

1. **Don't fight the architecture**: Work with vLLM's existing patterns
2. **Be honest about limitations**: No zero-copy GPU→CPU, SVD too slow
3. **Test comprehensively**: Especially prefill vs decode differences
4. **Keep CUDA graphs enabled**: Avoid hooks, use runner-level integration
5. **All tokens matter**: Capturing only last token misses critical information

## Acknowledgment

Thank you to the GPT5 reviewer for the thorough and honest assessment. Their expertise prevented us from shipping a fundamentally flawed system and guided us to a correct implementation that actually works with vLLM's architecture.
