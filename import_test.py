#!/usr/bin/env python
# Test script to verify imports

import os
import sys
import torch
from typing import List, Optional, Tuple, Union

# Add the parent directory to the path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
    # Import each module and check if it works
    try:
        from transformers.activations import ACT2FN
        print("✓ ACT2FN imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ACT2FN: {e}")
    
    try:
        from transformers.cache_utils import Cache, DynamicCache, StaticCache
        print("✓ Cache, DynamicCache, StaticCache imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cache_utils: {e}")
    
    try:
        from transformers.generation.utils import GenerationMixin
        print("✓ GenerationMixin imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GenerationMixin: {e}")
    
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        print("✓ AttentionMaskConverter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AttentionMaskConverter: {e}")
    
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
        print("✓ FlashAttentionKwargs imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FlashAttentionKwargs: {e}")
    
    try:
        from transformers.modeling_outputs import (
            BaseModelOutputWithPast,
            CausalLMOutputWithPast,
            QuestionAnsweringModelOutput,
            SequenceClassifierOutputWithPast,
            TokenClassifierOutput,
        )
        print("✓ modeling_outputs classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import modeling_outputs: {e}")
    
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        print("✓ ROPE_INIT_FUNCTIONS imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ROPE_INIT_FUNCTIONS: {e}")
    
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
        print("✓ ALL_ATTENTION_FUNCTIONS, PreTrainedModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import modeling_utils: {e}")
    
    try:
        from transformers.processing_utils import Unpack
        print("✓ Unpack imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Unpack: {e}")
        print("Trying alternative import for Unpack...")
        try:
            from typing import Unpack
            print("✓ Unpack imported from typing successfully")
        except ImportError:
            try:
                from typing_extensions import Unpack
                print("✓ Unpack imported from typing_extensions successfully")
            except ImportError as e2:
                print(f"✗ Failed to import Unpack from typing_extensions: {e2}")
    
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        print("✓ ALL_LAYERNORM_LAYERS imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ALL_LAYERNORM_LAYERS: {e}")
    
    try:
        from transformers.utils.generic import LossKwargs
        print("✓ LossKwargs imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LossKwargs: {e}")
    
    try:
        from transformers.utils import (
            add_code_sample_docstrings,
            add_start_docstrings,
            add_start_docstrings_to_model_forward,
            logging,
            replace_return_docstrings,
        )
        print("✓ utils functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils functions: {e}")
    
    try:
        from transformers.utils.deprecation import deprecate_kwarg
        print("✓ deprecate_kwarg imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import deprecate_kwarg: {e}")
    
    # Check for local imports - these might fail depending on the directory structure
    print("\nChecking local imports (these might fail if not in the correct directory):")
    try:
        # Try to find the configuration_llama.py file
        import glob
        llama_config_files = glob.glob("**/configuration_llama.py", recursive=True)
        if llama_config_files:
            print(f"Found potential LlamaConfig files at: {llama_config_files}")
            # Add the directory to path
            for file_path in llama_config_files:
                dir_path = os.path.dirname(os.path.abspath(file_path))
                if dir_path not in sys.path:
                    sys.path.append(dir_path)
                    print(f"Added {dir_path} to sys.path")
        
        try:
            from configuration_llama import LlamaConfig
            print("✓ LlamaConfig imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import LlamaConfig: {e}")
            try:
                from transformers import LlamaConfig
                print("✓ LlamaConfig imported from transformers successfully")
            except ImportError as e2:
                print(f"✗ Failed to import LlamaConfig from transformers: {e2}")
    except Exception as e:
        print(f"Error while trying to find LlamaConfig: {e}")
    
    try:
        # Try to find the utils.py file with PaperNorm
        utils_files = glob.glob("**/utils.py", recursive=True)
        if utils_files:
            print(f"Found potential utils.py files at: {utils_files}")
            # We need to check each one for PaperNorm
            for file_path in utils_files:
                try:
                    dir_path = os.path.dirname(os.path.abspath(file_path))
                    if dir_path not in sys.path:
                        sys.path.append(dir_path)
                        print(f"Added {dir_path} to sys.path")
                    
                    # Try to import from this specific utils file
                    sys.path.insert(0, dir_path)
                    try:
                        from utils import PaperNorm
                        print(f"✓ PaperNorm imported successfully from {file_path}")
                        # Remove the directory from the beginning of sys.path
                        sys.path.pop(0)
                        break
                    except (ImportError, AttributeError):
                        # Remove the directory from the beginning of sys.path
                        sys.path.pop(0)
                        continue
                except Exception as e:
                    print(f"Error checking {file_path}: {e}")
            else:
                print("✗ Could not find PaperNorm in any utils.py file")
        else:
            print("No utils.py files found")
    except Exception as e:
        print(f"Error while trying to find PaperNorm: {e}")

    print("\nImport test completed.")

if __name__ == "__main__":
    test_imports() 