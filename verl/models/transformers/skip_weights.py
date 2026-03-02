# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monkey patch to skip weight loading for testing purposes.
This is useful when you have model configuration but not the complete weight files.
"""

import os
import torch
import warnings
from typing import Dict, Any
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel


def patch_skip_weight_loading():
    """
    Patch transformers and verl to skip weight loading.
    This creates dummy weights instead of loading from files.
    """
    
    # Check if we should skip weight loading
    skip_loading = os.environ.get("VERL_SKIP_WEIGHT_LOADING", "false").lower() == "true"
    if not skip_loading:
        return
    
    print("⚠️  Patching to skip weight loading (VERL_SKIP_WEIGHT_LOADING=true)")
    
    # Store original from_pretrained method
    original_from_pretrained = HFPreTrainedModel.from_pretrained
    
    # Create patched version
    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        print(f"⚠️  Skipping weight loading for: {pretrained_model_name_or_path}")
        print(f"⚠️  Creating model with random weights for testing")
        
        # Get config
        config = kwargs.get('config')
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Create model with random weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cls(config)
            
            # Initialize weights with small random values
            for param in model.parameters():
                if param.requires_grad:
                    param.data.normal_(mean=0.0, std=0.02)
            
            # Move to appropriate device
            device = kwargs.get('device', None)
            if device is not None:
                model = model.to(device)
            
            # Set dtype if specified
            torch_dtype = kwargs.get('torch_dtype', None)
            if torch_dtype is not None:
                model = model.to(torch_dtype)
        
        return model
    
    # Apply the patch
    HFPreTrainedModel.from_pretrained = patched_from_pretrained
    print("✓ Patched PreTrainedModel.from_pretrained to skip weight loading")
    
    # Also patch verl's _load_hf_model function
    try:
        import verl.utils.model as verl_model_utils
        
        original_load_hf_model = verl_model_utils._load_hf_model
        
        def patched_load_hf_model(config, model_config, is_value_model):
            """Patched version that creates dummy weights"""
            print(f"⚠️  Skipping HF weight loading in verl (model path: {config.model.path})")
            
            from accelerate import init_empty_weights
            from megatron.core import parallel_state as mpu
            
            from verl.models.mcore.saver import _megatron_calc_global_rank
            
            architectures = getattr(model_config, "architectures", [])
            
            # Get auto class
            auto_cls = verl_model_utils.get_hf_auto_model_class(model_config)
            
            local_model_path = config.model.path
            print(f"⚠️  Creating dummy model for: {local_model_path}")
            
            src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=mpu.get_context_parallel_rank())
            cpu_init_weights = lambda: torch.device("cpu")
            init_context = init_empty_weights if torch.distributed.get_rank() != src_rank else cpu_init_weights
            
            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create model with random weights
                model = auto_cls(model_config)
                
                # Initialize with small random weights
                for param in model.parameters():
                    if param.requires_grad:
                        param.data.normal_(mean=0.0, std=0.02)
                
                # Create dummy state dict
                state_dict = model.state_dict()
            
            return architectures, model, state_dict, is_value_model
        
        # Apply the patch
        verl_model_utils._load_hf_model = patched_load_hf_model
        print("✓ Patched verl._load_hf_model to skip weight loading")
        
    except ImportError as e:
        print(f"⚠️  Could not patch verl utils: {e}")
    
    print("✅ All weight loading patches applied. Models will be created with random weights.")


def create_dummy_state_dict(model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    """
    Create a dummy state dictionary for testing.
    """
    state_dict = {}
    for name, param in model.named_parameters():
        # Create small random weights
        state_dict[name] = torch.randn_like(param) * 0.02
    
    return state_dict


def apply_all_patches():
    """
    Apply all patches for testing without weight files.
    """
    patch_skip_weight_loading()
    
    # Additional patches can be added here
    print("✅ All testing patches applied")


if __name__ == "__main__":
    # Test the patches
    apply_all_patches()