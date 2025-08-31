"""
Setup script for vLLM Activation Capture System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="vllm-activation-capture",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Real-time neural activation capture for vLLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vllm-activation-capture",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "numpy>=1.24.0,<2.0.0",
        "vllm>=0.10.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vllm-capture-test=vllm_capture.cli:test_capture",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)