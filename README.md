# vLLM Activation Capture System (Hook-Free)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/cuda-12.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Hook-free activation capture for vLLM that records ALL tokens (prefill and decode) using runner-level integration via `intermediate_tensors`. No PyTorch forward hooks, no double compute.

## 🎯 Key Features

- **Hook-Free**: Integrates at the runner level; CUDA graphs can remain enabled
- **All Tokens**: Captures prefill and decode tokens without last-token-only bugs
- **Practical Compression**: Random projection + uint8, full uint8, or top‑k sparse
- **Shard-Aware**: Works with tensor parallel; returns shard-local by default
- **Portable Artifacts**: Saves per-(sequence,layer) `.npz` with quantized activations

## 📊 Notes on Performance

- Overhead depends on compression mode, layers captured, and TP/PP topology.
- Random projection to 256–512 dims with uint8 quantization is typically fast enough for online use.
- We do not claim “zero-copy GPU→CPU” or unrealistically low overhead numbers.

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import os
from vllm import LLM, SamplingParams

# Configure activation capture (hook-free)
os.environ["VLLM_ACT_CAPTURE"] = "1"     # enable
os.environ["VLLM_ACT_MODE"] = "rp8"      # rp8|full8|topk8
os.environ["VLLM_ACT_RP_K"] = "512"      # RP output dims (rp8)
os.environ["VLLM_ACT_OUTDIR"] = "/tmp/acts"

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    worker_cls="vllm_capture.gpu_worker_capture.WorkerCapture",
    tensor_parallel_size=1,
)

outputs = llm.generate(["The future of AI is"], SamplingParams(max_tokens=32))

text = outputs[0].outputs[0].text
print(text)

# If you patched vLLM to attach manifests to RequestOutput.metrics,
# access them under outputs[...].metrics.extras["activation_manifest"].
```

## 📁 Repository Structure

```
vllm-activation-capture/
├── vllm_capture/              # Core implementation
│   ├── activations/            # Core capture + compression
│   │   └── capture.py
│   ├── v1/worker/
│   │   └── gpu_model_runner_correct.py
│   └── gpu_worker_capture.py   # Worker that uses the corrected runner
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── benchmarks/            # Performance benchmarks
├── examples/                  # Usage examples
│   ├── notebooks/             # Jupyter notebooks
│   └── scripts/               # Python scripts
├── results/                   # Output directory
│   ├── activations/           # Saved tensors
│   └── benchmarks/            # Performance data
└── docs/                      # Documentation
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VLLM_ACT_CAPTURE` | Enable activation capture | "0" | "1" |
| `VLLM_ACT_MODE` | `rp8` (RP+u8), `full8`, `topk8` | `rp8` | `rp8` |
| `VLLM_ACT_RP_K` | RP output dims (rp8) | `512` | `256` |
| `VLLM_ACT_RETURN` | `sharded` or `gathered` | `sharded` | `gathered` |
| `VLLM_ACT_OUTDIR` | Output directory for `.npz` | `/tmp/vllm_activations` | `/data/acts` |

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance benchmarks
python tests/benchmarks/benchmark_layer_scaling.py
```

## 📈 Notes on Scaling

Artifact sizes scale with tokens × layers × compressed dims. Use `rp8` (e.g., k=256–512) or `topk8` for sparse MLP capture to keep sizes manageable.

## 🔬 How It Works

### Technical Overview

1. **Runner Integration**: The runner passes an `intermediate_tensors` collector into model forward.
2. **Layer Taps**: Each transformer block calls `collector.add(layer_id, hidden_states)`.
3. **Compression**: Runner-side compression (RP+u8, full8, topk8) on GPU; then spill to `.npz`.
4. **Manifests**: A per-request manifest references the saved artifacts and compression metadata.

### Key Modifications to vLLM

- `GPUModelRunnerCorrect`: Runner that integrates capture via `intermediate_tensors`
- `ActivationCollector`: Token-accurate per-sequence rolling buffers + spill to `.npz`
- `WorkerCapture`: Worker that installs the corrected runner

## 📚 Examples

### Mode Examples
```python
os.environ["VLLM_ACT_MODE"] = "rp8"     # Residual stream RP to k dims + u8
os.environ["VLLM_ACT_RP_K"] = "256"     # Use 256 dims

os.environ["VLLM_ACT_MODE"] = "full8"  # Full vector u8 per-row/per-dim scale
os.environ["VLLM_ACT_MODE"] = "topk8"   # MLP top-k indices + u8 values
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm)
- Inspired by mechanistic interpretability research
- Developed for AI safety and control research

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{vllm_activation_capture,
  title = {vLLM Activation Capture System},
  author = {Research Team},
  year = {2025},
  url = {https://github.com/yourusername/vllm-activation-capture}
}
```

## 🔗 Links

- [Documentation](docs/)
- [Example Notebooks](examples/notebooks/)
- [Performance Analysis](docs/performance_analysis.md)
- [API Reference](docs/api_reference.md)

---

Notes:
- There is no zero-copy from GPU→CPU; copies happen before artifacts are written.
- Disable capture for max throughput. Enable only when needed.
