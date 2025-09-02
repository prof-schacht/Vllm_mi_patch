# vLLM Modifications (Hook-Free Integration)

## Overview

This documents the hook-free activation capture integration for vLLM. The design attaches a lightweight collector via the runner’s `intermediate_tensors` parameter so models can report per-layer hidden states without PyTorch forward hooks.

## Key Changes

- Runner: `vllm_capture/v1/worker/gpu_model_runner_correct.py`
  - Passes an `IntermediateTensorCollector` into model forward when capture is enabled.
  - Uses `ActivationCollector` to accumulate ALL tokens (prefill + decode) and spill compressed artifacts to `.npz`.
- Capture/Compression: `vllm_capture/activations/capture.py`
  - Modes: `rp8` (random projection + uint8), `full8` (uint8 quant), `topk8` (indices + uint8 values).
  - Per-sequence rolling buffers; shard-local by default under TP.
- Worker: `vllm_capture/gpu_worker_capture.py`
  - Installs `GPUModelRunnerCorrect` (no hooks).

## Integration Flow

1. Runner checks `VLLM_ACT_*` env vars; if enabled, instantiates `ActivationCollector`.
2. Runner forwards a collector shim (`IntermediateTensorCollector`) into the model’s forward.
3. Each transformer layer calls `collector.add(layer_id, hidden_states)`.
4. Collector compresses and accumulates tokens across prefill and decode.
5. On sequence completion, the collector finalizes and writes `.npz` per (seq, layer). A manifest describes shapes, paths, and compression.

## Environment Variables

- `VLLM_ACT_CAPTURE`: enable capture (0/1)
- `VLLM_ACT_MODE`: `rp8` | `full8` | `topk8`
- `VLLM_ACT_RP_K`: RP output dims (rp8)
- `VLLM_ACT_RETURN`: `sharded` | `gathered`
- `VLLM_ACT_OUTDIR`: output directory for artifacts

## Notes

- No PyTorch forward hooks; CUDA graphs can remain enabled.
- There is no zero‑copy from GPU→CPU; host copies occur before writing artifacts.
- Returning manifests in `RequestOutput.metrics.extras["activation_manifest"]` requires patching the vLLM worker where outputs are constructed. This repository includes the runner and collector; plumbing into public outputs can be added downstream.
