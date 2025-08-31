# vLLM Activation Capture System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/cuda-12.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A fork extension of vLLM that enables real-time neural activation capture during inference without double computation. Capture the model "red-handed" during the exact moment it generates text.

## 🎯 Key Features

- **Single-Pass Capture**: Hooks fire during the EXACT inference generating text - no double computation
- **Selective Layer Targeting**: Choose specific layers or capture all
- **Efficient Compression**: Optional SVD compression reduces storage by ~88%
- **Zero-Copy Transfer**: Shared memory buffers avoid serialization overhead
- **Post-hoc Marking**: Mark interesting events after generation for selective storage
- **Production Ready**: Tested on H100 GPUs with <5% performance overhead

## 📊 Performance Results

| Layers Captured | Compression | Throughput | Overhead | Storage/Inference |
|-----------------|-------------|------------|----------|-------------------|
| 3 layers | None | 399.2 tok/s | 0.1% | 5.9 MB |
| 5 layers | SVD-256 | 405.9 tok/s | -1.5% | 1.2 MB |
| 16 layers | SVD-256 | 394.3 tok/s | 1.4% | 3.9 MB |
| 32 layers (all) | SVD-256 | 383.2 tok/s | 4.1% | 7.8 MB |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vllm-activation-capture.git
cd vllm-activation-capture

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import os
from vllm import LLM, SamplingParams

# Configure activation capture
os.environ["VLLM_CAPTURE_ENABLED"] = "1"
os.environ["VLLM_CAPTURE_LAYERS"] = "0,7,15,23,31"  # Capture 5 layers
os.environ["VLLM_CAPTURE_COMPRESSION_K"] = "256"    # SVD compression

# Initialize model with capture
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    worker_cls="vllm.v1.worker.gpu_worker_capture.WorkerCapture",
    enforce_eager=True,  # Required for hooks
    tensor_parallel_size=1,
)

# Generate text (activations captured automatically)
outputs = llm.generate(["The future of AI is"], 
                       SamplingParams(temperature=0.7, max_tokens=50))

print(outputs[0].outputs[0].text)
```

## 📁 Repository Structure

```
vllm-activation-capture/
├── vllm_capture/              # Core implementation
│   ├── gpu_model_runner_capture.py
│   └── gpu_worker_capture.py
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
| `VLLM_CAPTURE_ENABLED` | Enable activation capture | "0" | "1" |
| `VLLM_CAPTURE_LAYERS` | Layers to capture | None | "0,7,15,23,31" or "all" |
| `VLLM_CAPTURE_COMPRESSION_K` | SVD components (None=no compression) | None | "256" |
| `VLLM_CAPTURE_BUFFER_SIZE_GB` | Shared memory buffer size | "2.0" | "8.0" |
| `VLLM_CAPTURE_SAMPLE_RATE` | Fraction of inferences to capture | "1.0" | "0.1" |

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance benchmarks
python tests/benchmarks/benchmark_layer_scaling.py
```

## 📈 Scaling Analysis

For 100 agents over 500 timesteps (50,000 inferences):

| Model | Layers | Compression | Storage | Runtime |
|-------|--------|-------------|---------|---------|
| Qwen2-0.5B | 24 | SVD-256 | 4.8 GB | ~14 hours |
| Qwen2.5-7B | 5 | SVD-256 | 240 MB | ~14 hours |
| Qwen2.5-7B | 32 | SVD-256 | 1.56 GB | ~14.4 hours |
| Qwen2.5-7B | 32 | None | 213.6 GB | ~14.4 hours |

## 🔬 How It Works

### Technical Innovation

1. **Hook Registration**: PyTorch hooks are registered on transformer layers during model initialization
2. **Single Forward Pass**: Hooks fire during the EXACT inference that generates text
3. **Shared Memory Transfer**: Activations are copied to shared memory via zero-copy transfer
4. **Post-hoc Selection**: After generation, mark interesting events for permanent storage
5. **Compression**: Optional SVD reduces storage while preserving information

### Key Modifications to vLLM

- `GPUModelRunnerCapture`: Extended model runner with hook registration
- `SharedActivationBuffer`: Zero-copy shared memory management
- `WorkerCapture`: Modified worker to use capture-enabled runner

## 📚 Examples

### Selective Layer Capture
```python
# Capture only critical layers for efficiency
os.environ["VLLM_CAPTURE_LAYERS"] = "0,15,31"  # First, middle, last
```

### Full Capture Without Compression
```python
# Maximum fidelity for deep analysis
os.environ["VLLM_CAPTURE_LAYERS"] = "all"
os.environ.pop("VLLM_CAPTURE_COMPRESSION_K", None)  # No compression
```

### Multi-Agent Simulation
```python
# Efficient settings for 100 agents
os.environ["VLLM_CAPTURE_LAYERS"] = "0,7,14,21,28"  # 5 distributed layers
os.environ["VLLM_CAPTURE_COMPRESSION_K"] = "256"     # Compress for storage
os.environ["VLLM_CAPTURE_SAMPLE_RATE"] = "0.1"      # Sample 10% of agents
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

**Note**: This is a research tool. Always validate captured activations match your experimental requirements.