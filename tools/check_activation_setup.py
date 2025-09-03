#!/usr/bin/env python3
"""
Verification script for vLLM activation capture setup.

This script checks that everything is properly configured for activation capture:
- vLLM is installed and accessible
- Model files are patched
- Custom worker can be imported
- Runs a minimal test to verify activation capture works

Usage:
    python tools/check_activation_setup.py
    python tools/check_activation_setup.py --full-test
    python tools/check_activation_setup.py --model qwen2
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'
CHECK = '✓'
CROSS = '✗'
INFO = 'ℹ'
WARN = '⚠'

class SetupChecker:
    """Comprehensive checker for vLLM activation capture setup."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checks_passed = []
        self.checks_failed = []
        self.checks_warning = []
        
    def print_header(self, title: str):
        """Print a section header."""
        print(f"\n{BOLD}{title}{RESET}")
        print("=" * 60)
    
    def print_result(self, success: bool, message: str, details: str = "", warning: bool = False):
        """Print a check result with formatting."""
        if warning:
            symbol = f"{YELLOW}{WARN}{RESET}"
            self.checks_warning.append(message)
        elif success:
            symbol = f"{GREEN}{CHECK}{RESET}"
            self.checks_passed.append(message)
        else:
            symbol = f"{RED}{CROSS}{RESET}"
            self.checks_failed.append(message)
        
        print(f"  {symbol} {message}")
        if details and self.verbose:
            print(f"    {details}")
    
    def check_python_version(self) -> bool:
        """Check Python version is 3.10+."""
        version = sys.version_info
        success = version.major == 3 and version.minor >= 10
        
        self.print_result(
            success,
            f"Python version: {version.major}.{version.minor}.{version.micro}",
            f"Required: 3.10+" if not success else ""
        )
        return success
    
    def check_vllm_installation(self) -> Optional[Path]:
        """Check if vLLM is installed and get its path."""
        try:
            import vllm
            vllm_path = Path(vllm.__file__).parent
            vllm_version = getattr(vllm, '__version__', 'unknown')
            
            self.print_result(
                True,
                f"vLLM installed: v{vllm_version}",
                f"Path: {vllm_path}"
            )
            return vllm_path
        except ImportError as e:
            self.print_result(
                False,
                "vLLM not installed",
                "Install with: pip install vllm"
            )
            return None
    
    def check_cuda_availability(self) -> bool:
        """Check CUDA availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                
                self.print_result(
                    True,
                    f"CUDA available: {device_count} GPU(s)",
                    f"Primary GPU: {device_name}, CUDA {cuda_version}"
                )
            else:
                self.print_result(
                    False,
                    "CUDA not available",
                    "GPU required for vLLM inference"
                )
            
            return cuda_available
        except ImportError:
            self.print_result(
                False,
                "PyTorch not installed",
                "Install with: pip install torch"
            )
            return False
    
    def check_worker_import(self) -> bool:
        """Check if custom worker can be imported."""
        try:
            # Add parent directory to path if needed
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from vllm_capture.gpu_worker_capture import WorkerCapture
            
            self.print_result(
                True,
                "Custom WorkerCapture importable",
                "vllm_capture.gpu_worker_capture.WorkerCapture"
            )
            return True
        except ImportError as e:
            self.print_result(
                False,
                "Cannot import WorkerCapture",
                f"Error: {e}\nTry: pip install -e ."
            )
            return False
    
    def check_model_patches(self, vllm_path: Path, specific_model: Optional[str] = None) -> Dict[str, bool]:
        """Check which model files are patched."""
        models_path = vllm_path / 'model_executor' / 'models'
        
        models_to_check = {
            'qwen2': 'qwen2.py',
            'qwen3': 'qwen3.py', 
            'llama': 'llama.py',
            'opt': 'opt.py',
            'mistral': 'mistral.py',
            'gpt2': 'gpt2.py',
        }
        
        if specific_model:
            models_to_check = {specific_model: models_to_check.get(specific_model, f"{specific_model}.py")}
        
        results = {}
        any_patched = False
        
        for model_name, filename in models_to_check.items():
            filepath = models_path / filename
            
            if not filepath.exists():
                results[model_name] = None
                continue
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            is_patched = 'Activation capture support (added by patch_vllm_models.py)' in content
            results[model_name] = is_patched
            
            if is_patched:
                any_patched = True
        
        # Print results
        if specific_model:
            if results.get(specific_model) is None:
                self.print_result(
                    False,
                    f"Model {specific_model} not found",
                    f"File not found: {models_path / models_to_check[specific_model]}"
                )
            elif results[specific_model]:
                self.print_result(
                    True,
                    f"Model {specific_model} is patched",
                    ""
                )
            else:
                self.print_result(
                    False,
                    f"Model {specific_model} not patched",
                    "Run: python tools/patch_vllm_models.py --model " + specific_model
                )
        else:
            patched_models = [m for m, p in results.items() if p]
            unpatched_models = [m for m, p in results.items() if p is False]
            missing_models = [m for m, p in results.items() if p is None]
            
            if patched_models:
                self.print_result(
                    True,
                    f"Patched models: {', '.join(patched_models)}",
                    ""
                )
            
            if unpatched_models:
                self.print_result(
                    False,
                    f"Unpatched models: {', '.join(unpatched_models)}",
                    "Run: python tools/patch_vllm_models.py --model all",
                    warning=True
                )
            
            if missing_models and self.verbose:
                print(f"    {BLUE}{INFO}{RESET} Missing models: {', '.join(missing_models)}")
        
        return results
    
    def check_environment_variables(self) -> bool:
        """Check if environment variables are properly set."""
        required_vars = {
            'VLLM_ACT_CAPTURE': '1',
            'VLLM_ACT_MODE': ['rp8', 'full8', 'topk8'],
            'VLLM_ACT_OUTDIR': None,  # Just needs to be set
        }
        
        optional_vars = {
            'VLLM_ACT_RP_K': None,
            'VLLM_ACT_HIDDEN_SIZE': None,
        }
        
        all_good = True
        
        for var, expected in required_vars.items():
            value = os.environ.get(var)
            
            if value is None:
                self.print_result(
                    False,
                    f"Environment variable {var} not set",
                    f"Set with: export {var}=..." ,
                    warning=True
                )
                all_good = False
            elif isinstance(expected, list):
                if value in expected:
                    self.print_result(True, f"{var}={value}", "")
                else:
                    self.print_result(
                        False,
                        f"{var}={value} (invalid)",
                        f"Expected one of: {', '.join(expected)}",
                        warning=True
                    )
                    all_good = False
            elif expected is not None:
                if value == expected:
                    self.print_result(True, f"{var}={value}", "")
                else:
                    self.print_result(
                        False,
                        f"{var}={value} (expected {expected})",
                        "",
                        warning=True
                    )
                    all_good = False
            else:
                self.print_result(True, f"{var}={value}", "")
        
        # Check optional vars if verbose
        if self.verbose:
            for var, _ in optional_vars.items():
                value = os.environ.get(var)
                if value:
                    print(f"    {BLUE}{INFO}{RESET} {var}={value}")
        
        return all_good
    
    def run_minimal_test(self, model_name: str = "gpt2") -> bool:
        """Run a minimal generation test to verify activation capture works."""
        self.print_header("Running Minimal Test")
        
        # Set up temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="vllm_act_test_")
        os.environ['VLLM_ACT_CAPTURE'] = '1'
        os.environ['VLLM_ACT_MODE'] = 'rp8'
        os.environ['VLLM_ACT_RP_K'] = '64'
        os.environ['VLLM_ACT_OUTDIR'] = temp_dir
        
        print(f"  Output directory: {temp_dir}")
        
        try:
            from vllm import LLM, SamplingParams
            
            print(f"  Loading model: {model_name}...")
            
            # Try to use a small model for testing
            model_path = model_name
            if model_name == "gpt2":
                model_path = "gpt2"  # Use HuggingFace model ID
            
            llm = LLM(
                model=model_path,
                worker_cls='vllm_capture.gpu_worker_capture.WorkerCapture',
                max_model_len=128,
                enforce_eager=True,
                gpu_memory_utilization=0.5,
            )
            
            print(f"  Generating text...")
            outputs = llm.generate(
                ["Hello world"],
                SamplingParams(max_tokens=10, temperature=0)
            )
            
            generated_text = outputs[0].outputs[0].text
            print(f"  Generated: '{generated_text.strip()}'")
            
            # Check for activation files
            npz_files = list(Path(temp_dir).rglob("*.npz"))
            
            if npz_files:
                self.print_result(
                    True,
                    f"Activation capture working! {len(npz_files)} files created",
                    f"First file: {npz_files[0].name}"
                )
                
                # Verify file content
                data = np.load(npz_files[0])
                if 'q' in data:
                    shape = data['q'].shape
                    print(f"    Activation shape: {shape} (tokens × RP dims)")
                
                return True
            else:
                self.print_result(
                    False,
                    "No activation files created",
                    "Check if model is properly patched"
                )
                return False
                
        except Exception as e:
            self.print_result(
                False,
                f"Test failed: {type(e).__name__}",
                str(e)
            )
            return False
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def print_summary(self):
        """Print a summary of all checks."""
        self.print_header("Summary")
        
        total = len(self.checks_passed) + len(self.checks_failed) + len(self.checks_warning)
        
        if self.checks_passed:
            print(f"  {GREEN}{CHECK}{RESET} Passed: {len(self.checks_passed)}/{total}")
        
        if self.checks_warning:
            print(f"  {YELLOW}{WARN}{RESET} Warnings: {len(self.checks_warning)}/{total}")
        
        if self.checks_failed:
            print(f"  {RED}{CROSS}{RESET} Failed: {len(self.checks_failed)}/{total}")
            print(f"\n  {BOLD}Failed checks:{RESET}")
            for check in self.checks_failed:
                print(f"    - {check}")
        
        if not self.checks_failed:
            print(f"\n{GREEN}{BOLD}✅ All critical checks passed!{RESET}")
            print("Your vLLM activation capture setup is ready to use.")
        else:
            print(f"\n{RED}{BOLD}❌ Some checks failed.{RESET}")
            print("Please fix the issues above before using activation capture.")
    
    def run_all_checks(self, full_test: bool = False, model: Optional[str] = None):
        """Run all verification checks."""
        print(f"{BOLD}vLLM Activation Capture Setup Verification{RESET}")
        print("=" * 60)
        
        # System checks
        self.print_header("System Requirements")
        self.check_python_version()
        self.check_cuda_availability()
        
        # vLLM checks
        self.print_header("vLLM Installation")
        vllm_path = self.check_vllm_installation()
        
        if vllm_path:
            # Model patches
            self.print_header("Model Patches")
            self.check_model_patches(vllm_path, model)
        
        # Import checks
        self.print_header("Import Checks")
        self.check_worker_import()
        
        # Environment checks
        self.print_header("Environment Variables")
        self.check_environment_variables()
        
        # Run test if requested
        if full_test and not self.checks_failed:
            if model:
                self.run_minimal_test(model)
            else:
                # Try to find a patched model for testing
                print(f"\n{YELLOW}{WARN}{RESET} Skipping test (use --full-test --model <name>)")
        
        # Summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Verify vLLM activation capture setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python tools/check_activation_setup.py
  
  # Run with full test (generates text and captures activations)
  python tools/check_activation_setup.py --full-test --model gpt2
  
  # Check specific model
  python tools/check_activation_setup.py --model qwen2
  
  # Quiet mode (less verbose)
  python tools/check_activation_setup.py --quiet
        """
    )
    
    parser.add_argument(
        '--full-test',
        action='store_true',
        help='Run a full test including generation and activation capture'
    )
    parser.add_argument(
        '--model',
        help='Specific model to check/test'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    checker = SetupChecker(verbose=not args.quiet)
    checker.run_all_checks(full_test=args.full_test, model=args.model)
    
    # Exit with error if checks failed
    if checker.checks_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()