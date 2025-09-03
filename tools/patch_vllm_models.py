#!/usr/bin/env python3
"""
One-shot patcher for vLLM model files to enable activation capture.

This script modifies installed vLLM model files to add intermediate_tensors.add()
calls after each transformer layer, enabling hook-free activation capture.

Usage:
    python tools/patch_vllm_models.py --model qwen2
    python tools/patch_vllm_models.py --model all --backup
    python tools/patch_vllm_models.py --check
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import importlib.util

# ANSI color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ModelPatcher:
    """Patches vLLM model files to add activation capture support."""
    
    # Model configurations: (file_name, class_name, layer_loop_pattern)
    SUPPORTED_MODELS = {
        'qwen2': {
            'file': 'qwen2.py',
            'classes': ['Qwen2Model'],
            'pattern': r'(for idx, layer in enumerate\(\s*self\.layers\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states, residual = layer\(.*?\))',
            'insert_after': 'hidden_states, residual = layer(',
            'indent_level': 3,
        },
        'qwen3': {
            'file': 'qwen3.py',  # Often same as qwen2
            'classes': ['Qwen3Model', 'Qwen2Model'],  # May reuse Qwen2Model
            'pattern': r'(for idx, layer in enumerate\(\s*self\.layers\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states, residual = layer\(.*?\))',
            'insert_after': 'hidden_states, residual = layer(',
            'indent_level': 3,
        },
        'llama': {
            'file': 'llama.py',
            'classes': ['LlamaModel'],
            'pattern': r'(for idx, layer in enumerate\(\s*self\.layers\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states, residual = layer\(.*?\))',
            'insert_after': 'hidden_states, residual = layer(',
            'indent_level': 3,
        },
        'opt': {
            'file': 'opt.py',
            'classes': ['OPTModel'],
            'pattern': r'(for idx, layer in enumerate\(layers\):.*?)(hidden_states = layer\(.*?\))',
            'insert_after': 'hidden_states = layer(',
            'indent_level': 3,
        },
        'mistral': {
            'file': 'mistral.py',
            'classes': ['MistralModel'],
            'pattern': r'(for idx, layer in enumerate\(\s*self\.layers\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states, residual = layer\(.*?\))',
            'insert_after': 'hidden_states, residual = layer(',
            'indent_level': 3,
        },
        'mixtral': {
            'file': 'mixtral.py',
            'classes': ['MixtralModel'],
            'pattern': r'(for idx, layer in enumerate\(\s*self\.layers\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states, residual = layer\(.*?\))',
            'insert_after': 'hidden_states, residual = layer(',
            'indent_level': 3,
        },
        'gpt2': {
            'file': 'gpt2.py',
            'classes': ['GPT2Model'],
            'pattern': r'(for idx, layer in enumerate\(self\.h\):.*?)(hidden_states = layer\(.*?\))',
            'insert_after': 'hidden_states = layer(',
            'indent_level': 3,
        },
        'falcon': {
            'file': 'falcon.py',
            'classes': ['FalconModel'],
            'pattern': r'(for idx, layer in enumerate\(\s*self\.h\[self\.start_layer:self\.end_layer\]\):.*?)(hidden_states = layer\(.*?\))',
            'insert_after': 'hidden_states = layer(',
            'indent_level': 3,
        },
    }
    
    # The activation capture code to insert
    ACTIVATION_CAPTURE_CODE = """
            # Activation capture support (added by patch_vllm_models.py)
            if intermediate_tensors is not None and hasattr(intermediate_tensors, 'add'):
                intermediate_tensors.add({layer_idx}, hidden_states)"""
    
    def __init__(self, backup: bool = True, verbose: bool = True):
        self.backup = backup
        self.verbose = verbose
        self.vllm_path = self._find_vllm_path()
        self.models_path = self.vllm_path / 'model_executor' / 'models'
        
    def _find_vllm_path(self) -> Path:
        """Find the vLLM installation path."""
        try:
            import vllm
            vllm_path = Path(vllm.__file__).parent
            if self.verbose:
                print(f"{GREEN}✓{RESET} Found vLLM at: {vllm_path}")
            return vllm_path
        except ImportError:
            print(f"{RED}✗{RESET} vLLM not installed. Please install with: pip install vllm")
            sys.exit(1)
            
    def _backup_file(self, filepath: Path) -> Optional[Path]:
        """Create a backup of the file before patching."""
        if not self.backup:
            return None
            
        backup_path = filepath.with_suffix('.py.backup')
        
        # Don't overwrite existing backups
        if backup_path.exists():
            if self.verbose:
                print(f"  {YELLOW}⚠{RESET} Backup already exists: {backup_path.name}")
            return backup_path
            
        shutil.copy2(filepath, backup_path)
        if self.verbose:
            print(f"  {GREEN}✓{RESET} Created backup: {backup_path.name}")
        return backup_path
    
    def _is_already_patched(self, content: str) -> bool:
        """Check if the file is already patched."""
        return 'Activation capture support (added by patch_vllm_models.py)' in content
    
    def _patch_content(self, content: str, model_name: str, config: dict) -> Tuple[str, int]:
        """Patch the content of a model file."""
        if self._is_already_patched(content):
            return content, 0
        
        patches_made = 0
        lines = content.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            
            # Look for the layer loop and the layer call
            if 'for idx, layer in enumerate' in line:
                # Found the loop, now look for the layer call in next few lines
                j = i + 1
                while j < min(i + 10, len(lines)):
                    if config['insert_after'] in lines[j]:
                        # Found the layer call, insert our code after it
                        new_lines.append(lines[j])
                        
                        # Determine the layer index expression
                        if 'self.start_layer:self.end_layer' in content:
                            layer_idx = 'idx + self.start_layer'
                        else:
                            layer_idx = 'idx'
                        
                        # Add the activation capture code with proper indentation
                        indent = '    ' * config['indent_level']
                        capture_code = self.ACTIVATION_CAPTURE_CODE.format(layer_idx=layer_idx)
                        for code_line in capture_code.split('\n')[1:]:  # Skip first empty line
                            new_lines.append(code_line)
                        
                        patches_made += 1
                        
                        # Add remaining lines after the insertion point
                        for k in range(j + 1, len(lines)):
                            new_lines.append(lines[k])
                        
                        # Break out of all loops
                        i = len(lines)
                        break
                    j += 1
            i += 1
        
        return '\n'.join(new_lines) if patches_made > 0 else content, patches_made
    
    def patch_model(self, model_name: str) -> bool:
        """Patch a specific model file."""
        if model_name not in self.SUPPORTED_MODELS:
            print(f"{RED}✗{RESET} Model '{model_name}' not supported")
            return False
        
        config = self.SUPPORTED_MODELS[model_name]
        filepath = self.models_path / config['file']
        
        if not filepath.exists():
            if self.verbose:
                print(f"{YELLOW}⚠{RESET} Model file not found: {filepath}")
            return False
        
        print(f"\n{BOLD}Patching {model_name}:{RESET}")
        
        # Read the file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if self._is_already_patched(content):
            print(f"  {BLUE}ℹ{RESET} Already patched")
            return True
        
        # Create backup
        self._backup_file(filepath)
        
        # Patch the content
        patched_content, patches_made = self._patch_content(content, model_name, config)
        
        if patches_made == 0:
            print(f"  {YELLOW}⚠{RESET} No patches needed (pattern not found)")
            return False
        
        # Write the patched content
        with open(filepath, 'w') as f:
            f.write(patched_content)
        
        print(f"  {GREEN}✓{RESET} Patched successfully ({patches_made} insertion{'s' if patches_made > 1 else ''})")
        return True
    
    def patch_all(self) -> dict:
        """Patch all supported models."""
        results = {}
        for model_name in self.SUPPORTED_MODELS:
            results[model_name] = self.patch_model(model_name)
        return results
    
    def check_patches(self) -> dict:
        """Check which models are patched."""
        results = {}
        print(f"\n{BOLD}Checking patch status:{RESET}")
        
        for model_name, config in self.SUPPORTED_MODELS.items():
            filepath = self.models_path / config['file']
            
            if not filepath.exists():
                results[model_name] = 'not_found'
                print(f"  {model_name:12} {YELLOW}[ NOT FOUND ]{RESET}")
                continue
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            if self._is_already_patched(content):
                results[model_name] = 'patched'
                print(f"  {model_name:12} {GREEN}[ PATCHED ]{RESET}")
            else:
                results[model_name] = 'not_patched'
                print(f"  {model_name:12} {RED}[ NOT PATCHED ]{RESET}")
        
        return results
    
    def verify_import(self) -> bool:
        """Verify that patched models can be imported."""
        print(f"\n{BOLD}Verifying imports:{RESET}")
        
        try:
            # Try importing a common model
            spec = importlib.util.spec_from_file_location(
                "test_model",
                self.models_path / "qwen2.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"  {GREEN}✓{RESET} Models can be imported successfully")
                return True
        except Exception as e:
            print(f"  {RED}✗{RESET} Import verification failed: {e}")
            return False
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Patch vLLM model files for activation capture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch a specific model
  python tools/patch_vllm_models.py --model qwen2
  
  # Patch all supported models
  python tools/patch_vllm_models.py --model all
  
  # Check patch status
  python tools/patch_vllm_models.py --check
  
  # Patch without creating backups
  python tools/patch_vllm_models.py --model llama --no-backup
        """
    )
    
    parser.add_argument(
        '--model',
        choices=list(ModelPatcher.SUPPORTED_MODELS.keys()) + ['all'],
        help='Model to patch (or "all" for all supported models)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check which models are already patched'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Require either --model or --check
    if not args.model and not args.check:
        parser.print_help()
        sys.exit(1)
    
    patcher = ModelPatcher(
        backup=not args.no_backup,
        verbose=not args.quiet
    )
    
    print(f"{BOLD}vLLM Model Patcher for Activation Capture{RESET}")
    print(f"{'=' * 50}")
    
    if args.check:
        patcher.check_patches()
    elif args.model == 'all':
        results = patcher.patch_all()
        
        # Summary
        print(f"\n{BOLD}Summary:{RESET}")
        patched = sum(1 for v in results.values() if v)
        print(f"  {GREEN}✓{RESET} {patched}/{len(results)} models patched successfully")
        
        # Verify imports
        patcher.verify_import()
    else:
        success = patcher.patch_model(args.model)
        if success:
            print(f"\n{GREEN}✓{RESET} Patching complete!")
            patcher.verify_import()
        else:
            print(f"\n{YELLOW}⚠{RESET} Patching incomplete")
            sys.exit(1)


if __name__ == '__main__':
    main()