#!/usr/bin/env python3
"""
Unpatcher for vLLM model files - reverses patches applied by patch_vllm_models.py

Usage:
    python tools/unpatch_vllm_models.py --model qwen2
    python tools/unpatch_vllm_models.py --model all
    python tools/unpatch_vllm_models.py --restore-from-backup
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ModelUnpatcher:
    """Removes activation capture patches from vLLM model files."""
    
    SUPPORTED_MODELS = {
        'qwen2': 'qwen2.py',
        'qwen3': 'qwen3.py',
        'llama': 'llama.py',
        'opt': 'opt.py',
        'mistral': 'mistral.py',
        'mixtral': 'mixtral.py',
        'gpt2': 'gpt2.py',
        'falcon': 'falcon.py',
    }
    
    def __init__(self, verbose: bool = True):
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
            print(f"{RED}✗{RESET} vLLM not installed")
            sys.exit(1)
    
    def _remove_patch_from_content(self, content: str) -> tuple[str, int]:
        """Remove patch code from file content."""
        lines = content.split('\n')
        new_lines = []
        removed_count = 0
        i = 0
        
        while i < len(lines):
            # Look for our patch marker
            if 'Activation capture support (added by patch_vllm_models.py)' in lines[i]:
                # Skip the comment line and the following lines until we find a line
                # that doesn't belong to our patch
                removed_count += 1
                
                # Skip lines that are part of our patch
                while i < len(lines) and (
                    'Activation capture support' in lines[i] or
                    'intermediate_tensors.add(' in lines[i] or
                    (i + 1 < len(lines) and 'intermediate_tensors.add(' in lines[i + 1])
                ):
                    i += 1
                continue
            
            new_lines.append(lines[i])
            i += 1
        
        return '\n'.join(new_lines), removed_count
    
    def unpatch_model(self, model_name: str) -> bool:
        """Remove patches from a specific model file."""
        if model_name not in self.SUPPORTED_MODELS:
            print(f"{RED}✗{RESET} Model '{model_name}' not supported")
            return False
        
        filepath = self.models_path / self.SUPPORTED_MODELS[model_name]
        
        if not filepath.exists():
            if self.verbose:
                print(f"{YELLOW}⚠{RESET} Model file not found: {filepath}")
            return False
        
        print(f"\n{BOLD}Unpatching {model_name}:{RESET}")
        
        # Read the file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if patched
        if 'Activation capture support (added by patch_vllm_models.py)' not in content:
            print(f"  {BLUE}ℹ{RESET} Not patched, nothing to remove")
            return True
        
        # Remove patches
        cleaned_content, removed_count = self._remove_patch_from_content(content)
        
        if removed_count > 0:
            # Write the cleaned content
            with open(filepath, 'w') as f:
                f.write(cleaned_content)
            
            print(f"  {GREEN}✓{RESET} Removed {removed_count} patch{'es' if removed_count > 1 else ''}")
            return True
        else:
            print(f"  {YELLOW}⚠{RESET} No patches found to remove")
            return False
    
    def unpatch_all(self) -> dict:
        """Unpatch all supported models."""
        results = {}
        for model_name in self.SUPPORTED_MODELS:
            results[model_name] = self.unpatch_model(model_name)
        return results
    
    def restore_from_backup(self, model_name: Optional[str] = None) -> int:
        """Restore model files from backups."""
        restored_count = 0
        
        if model_name:
            # Restore specific model
            if model_name not in self.SUPPORTED_MODELS:
                print(f"{RED}✗{RESET} Model '{model_name}' not supported")
                return 0
            
            models_to_restore = [model_name]
        else:
            # Restore all models
            models_to_restore = list(self.SUPPORTED_MODELS.keys())
        
        print(f"\n{BOLD}Restoring from backups:{RESET}")
        
        for model in models_to_restore:
            filepath = self.models_path / self.SUPPORTED_MODELS[model]
            backup_path = filepath.with_suffix('.py.backup')
            
            if backup_path.exists():
                shutil.copy2(backup_path, filepath)
                print(f"  {GREEN}✓{RESET} Restored {model} from backup")
                restored_count += 1
            else:
                if self.verbose:
                    print(f"  {YELLOW}⚠{RESET} No backup found for {model}")
        
        return restored_count
    
    def list_backups(self) -> list:
        """List all available backup files."""
        backups = []
        print(f"\n{BOLD}Available backups:{RESET}")
        
        for model_name, filename in self.SUPPORTED_MODELS.items():
            backup_path = self.models_path / f"{filename}.backup"
            if backup_path.exists():
                backups.append(model_name)
                size = backup_path.stat().st_size / 1024  # KB
                print(f"  {model_name:12} {GREEN}[{size:.1f} KB]{RESET}")
        
        if not backups:
            print(f"  {YELLOW}No backups found{RESET}")
        
        return backups


def main():
    parser = argparse.ArgumentParser(
        description='Remove activation capture patches from vLLM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unpatch a specific model
  python tools/unpatch_vllm_models.py --model qwen2
  
  # Unpatch all models
  python tools/unpatch_vllm_models.py --model all
  
  # Restore from backup files
  python tools/unpatch_vllm_models.py --restore-from-backup
  
  # List available backups
  python tools/unpatch_vllm_models.py --list-backups
        """
    )
    
    parser.add_argument(
        '--model',
        choices=list(ModelUnpatcher.SUPPORTED_MODELS.keys()) + ['all'],
        help='Model to unpatch (or "all" for all models)'
    )
    parser.add_argument(
        '--restore-from-backup',
        action='store_true',
        help='Restore original files from backups'
    )
    parser.add_argument(
        '--list-backups',
        action='store_true',
        help='List available backup files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    if not args.model and not args.restore_from_backup and not args.list_backups:
        parser.print_help()
        sys.exit(1)
    
    unpatcher = ModelUnpatcher(verbose=not args.quiet)
    
    print(f"{BOLD}vLLM Model Unpatcher{RESET}")
    print(f"{'=' * 50}")
    
    if args.list_backups:
        unpatcher.list_backups()
    elif args.restore_from_backup:
        restored = unpatcher.restore_from_backup(args.model if args.model != 'all' else None)
        print(f"\n{GREEN}✓{RESET} Restored {restored} file{'s' if restored != 1 else ''} from backup")
    elif args.model == 'all':
        results = unpatcher.unpatch_all()
        
        # Summary
        print(f"\n{BOLD}Summary:{RESET}")
        unpatched = sum(1 for v in results.values() if v)
        print(f"  {GREEN}✓{RESET} {unpatched}/{len(results)} models unpatched successfully")
    else:
        success = unpatcher.unpatch_model(args.model)
        if success:
            print(f"\n{GREEN}✓{RESET} Unpatching complete!")
        else:
            print(f"\n{YELLOW}⚠{RESET} Unpatching incomplete")


if __name__ == '__main__':
    main()