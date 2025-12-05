import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import math
import os
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

class SafetyModel(nn.Module):
    def __init__(self, unet, train_method='xattn', rank=4, lora_up_std=0.0002, lora_down_std=0.02):
        super().__init__()
        self.unet = unet
        self.train_method = train_method
        self.rank = rank
        self.lora_up_std = lora_up_std
        self.lora_down_std = lora_down_std
        
        print(f"\n=== Initializing SafetyModel ===")
        print(f"Training method: {train_method}")
        if train_method == 'full':
            print("Will train LoRA parameters for all attention layers")
        elif train_method == 'xattn':
            print("Will only train LoRA parameters for cross-attention layers (attn2)")
        elif train_method == 'noxattn':
            print("Will only train LoRA parameters for self-attention layers (attn1)")
        elif train_method == 'xattn-strict':
            print("Will only train LoRA parameters for specific modules in cross-attention layers")
        
        # Inject LoRA layers
        self._setup_lora_layers()
        
        # Save original requires_grad state of parameters
        self.original_grad_state = {}
        for name, param in self.unet.named_parameters():
            self.original_grad_state[name] = param.requires_grad
            param.requires_grad = False
    
    def _setup_lora_layers(self):
        """Inject LoRA layers"""
        print("\n=== Starting LoRA layer injection ===")
        
        # Get attention processor configuration
        lora_attn_procs = {}
        trainable_layers = []
        frozen_layers = []
        
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            # Create LoRA processor for all layers
            lora_attn_proc = LoRAAttnProcessor(
                hidden_size=hidden_size,

                cross_attention_dim=cross_attention_dim,
                rank=self.rank,
            )

            # Initialize LoRA weights
            for param_name, param in lora_attn_proc.named_parameters():
                if 'up.weight' in param_name:
                    nn.init.normal_(param, mean=0.0, std=self.lora_up_std)
                elif 'down.weight' in param_name:
                    nn.init.normal_(param, mean=0.0, std=self.lora_down_std)
            
            lora_attn_procs[name] = lora_attn_proc
            
            # Set requires_grad based on training method
            should_train = self._should_train_layer(name)
            if should_train:
                trainable_layers.append(name)
            else:
                frozen_layers.append(name)
        
        # Set attention processors
        self.unet.set_attn_processor(lora_attn_procs)
        
        # Create parameter manager
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        
        print(f"\n=== LoRA layer injection completed ===")
        print(f"- Total layers: {len(lora_attn_procs)}")
        print(f"- Trainable layers: {len(trainable_layers)}")
        print(f"- Frozen layers: {len(frozen_layers)}")
    
    def get_lora_params(self):
        params = []
        up_count = 0
        down_count = 0
        
        for name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, LoRAAttnProcessor) and self._should_train_layer(name):
                for pname, param in attn_proc.named_parameters():
                    param.requires_grad = True  # Force unfreeze
                    params.append(param)
                    
                    # Count up and down matrices
                    if 'up.weight' in pname:
                        up_count += 1
                    elif 'down.weight' in pname:
                        down_count += 1
                        

        print(f"\n=== LoRA parameter statistics ===")
        print(f"Total parameter count: {len(params)}")
        print(f"Up matrix count: {up_count}")
        print(f"Down matrix count: {down_count}")
        print(f"Parameters per layer: {len(params) // up_count if up_count > 0 else 0}")
        
        return params
    
    def update_lora_params(self, params):
        """Update LoRA parameters, maintaining computation graph connection"""
        try:
            param_idx = 0
            stats = {
                'up': {'total': 0, 'zero': 0},
                'down': {'total': 0, 'zero': 0}
            }
            zero_params = {
                'up': [],
                'down': []
            }
            
            for name, attn_proc in self.unet.attn_processors.items():
                if isinstance(attn_proc, LoRAAttnProcessor) and self._should_train_layer(name):
                    for param_name, _ in attn_proc.named_parameters():
                        param = params[param_idx]
                        
                        # Ensure parameter is nn.Parameter type
                        if not isinstance(param, nn.Parameter):
                            param = nn.Parameter(param.detach().clone(), requires_grad=True)
                        
                        # Collect statistics
                        param_type = 'up' if 'up.weight' in param_name else 'down'
                        stats[param_type]['total'] += 1
                        if torch.all(param == 0):
                            stats[param_type]['zero'] += 1
                            zero_params[param_type].append(f"{name} - {param_name}")
                        
                        # Update parameters
                        if 'to_q_lora' in param_name:
                            if 'up.weight' in param_name:
                                attn_proc.to_q_lora.up.weight = param
                            elif 'down.weight' in param_name:
                                attn_proc.to_q_lora.down.weight = param
                        elif 'to_k_lora' in param_name:
                            if 'up.weight' in param_name:
                                attn_proc.to_k_lora.up.weight = param
                            elif 'down.weight' in param_name:
                                attn_proc.to_k_lora.down.weight = param
                        elif 'to_v_lora' in param_name:
                            if 'up.weight' in param_name:
                                attn_proc.to_v_lora.up.weight = param
                            elif 'down.weight' in param_name:
                                attn_proc.to_v_lora.down.weight = param
                        elif 'to_out_lora' in param_name:
                            if 'up.weight' in param_name:
                                attn_proc.to_out_lora.up.weight = param
                            elif 'down.weight' in param_name:
                                attn_proc.to_out_lora.down.weight = param
                        
                        param_idx += 1
            

            print(f"\nSuccessfully updated {param_idx} LoRA parameters")
            
        except Exception as e:
            print(f"Error updating LoRA parameters: {str(e)}")
            print(f"Current parameter index: {param_idx}")
            print(f"Total parameter count: {len(params)}")
            raise e
    

    def _should_train_layer(self, layer_name):
        """Determine whether this layer should be trained"""
        if self.train_method == 'full':
            return True
        elif self.train_method == 'xattn':
            return 'attn2' in layer_name  # Cross-attention
        elif self.train_method == 'noxattn':
            return 'attn1' in layer_name  # Self-attention
        elif self.train_method == 'xattn-strict':
            return 'attn2' in layer_name
        return False
    
    def merge_and_save_lora_weights(self, save_path=None):
        """Merge LoRA weights into base model, considering scaling factor"""
        print("\n=== Starting to merge LoRA weights into base model ===")
        merged_count = 0
        
        for name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, LoRAAttnProcessor):
                # Get original layer
                if name.endswith("attn1.processor"):
                    base_layer = self.unet.get_submodule(name.replace(".processor", ""))
                else:
                    base_layer = self.unet.get_submodule(name.replace(".processor", ""))
                
                # Merge weights
                with torch.no_grad():
                    # Ensure all tensors are on the same device
                    device = base_layer.to_q.weight.device
                    
                    # Get LoRA scaling factor (usually 1.0, but we explicitly get it for safety)
                    scale = getattr(attn_proc, 'scale', 1.0)
                    
                    # Calculate and merge LoRA weights for each attention head
                    # Query
                    q_weight = (attn_proc.to_q_lora.up.weight @ attn_proc.to_q_lora.down.weight).to(device) * scale
                    base_layer.to_q.weight.data += q_weight
                    
                    # Key
                    k_weight = (attn_proc.to_k_lora.up.weight @ attn_proc.to_k_lora.down.weight).to(device) * scale
                    base_layer.to_k.weight.data += k_weight
                    
                    # Value
                    v_weight = (attn_proc.to_v_lora.up.weight @ attn_proc.to_v_lora.down.weight).to(device) * scale
                    base_layer.to_v.weight.data += v_weight
                    
                    # Output
                    out_weight = (attn_proc.to_out_lora.up.weight @ attn_proc.to_out_lora.down.weight).to(device) * scale
                    base_layer.to_out[0].weight.data += out_weight
                    
                    merged_count += 1
                    
                    # Print debug information
                    print(f"Merging layer {name}:")
                    print(f"- Query weight range: [{q_weight.min():.6f}, {q_weight.max():.6f}]")
                    print(f"- Key weight range: [{k_weight.min():.6f}, {k_weight.max():.6f}]")
                    print(f"- Value weight range: [{v_weight.min():.6f}, {v_weight.max():.6f}]")
                    print(f"- Output weight range: [{out_weight.min():.6f}, {out_weight.max():.6f}]")
        
        print(f"\nSuccessfully merged {merged_count} LoRA layer weights")
        
        # Important: Remove all LoRA processors, restore to default processors
        print("\nRemoving LoRA processors...")
        self.unet.set_default_attn_processor()
        
        if save_path:
            print(f"\nSaving merged model to {save_path}")
            # Save state_dict
            torch.save(self.unet.state_dict(), save_path)
            print("Model saving completed")
        
        return merged_count
    
    def __enter__(self):
        """Context manager entry: Set layers that need to be trained"""
        # Freeze all parameters
        for param in self.unet.parameters():
            param.requires_grad = False
            
        # Only enable LoRA parameters that need to be trained
        for name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, LoRAAttnProcessor):
                should_train = self._should_train_layer(name)
                for param in attn_proc.parameters():
                    param.requires_grad = should_train
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Restore training state of all layers"""
        # Restore original requires_grad state
        for name, param in self.unet.named_parameters():
            if name in self.original_grad_state:
                param.requires_grad = self.original_grad_state[name]
    
    def merge_lora_to_unet(self, lora_params):
        """Merge LoRA parameters into original UNet for full parameter finetuning
        
        Args:
            lora_params: LoRA parameter list (not used for now, directly use LoRA weights in current UNet)
            
        Returns:
            merged_unet: Merged UNet model
        """
        print("\n=== Starting to merge LoRA parameters into UNet ===")
        
        # Create deep copy of UNet
        import copy
        merged_unet = copy.deepcopy(self.unet)
        
        # Remove all LoRA processors, restore to default processors
        merged_unet.set_default_attn_processor()
        
        merged_count = 0
        
        # Iterate over all attention layers, merge LoRA weights
        for name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, LoRAAttnProcessor) and self._should_train_layer(name):
                # Get original layer
                if name.endswith("attn1.processor"):
                    base_layer = merged_unet.get_submodule(name.replace(".processor", ""))
                else:
                    base_layer = merged_unet.get_submodule(name.replace(".processor", ""))
                
                # Merge weights
                with torch.no_grad():
                    # Ensure all tensors are on the same device
                    device = base_layer.to_q.weight.device
                    
                    # Get LoRA scaling factor (usually 1.0, but we explicitly get it for safety)
                    scale = getattr(attn_proc, 'scale', 1.0)
                    
                    # Calculate and merge LoRA weights for each attention head
                    # Query
                    q_weight = (attn_proc.to_q_lora.up.weight @ attn_proc.to_q_lora.down.weight).to(device) * scale
                    base_layer.to_q.weight.data += q_weight
                    
                    # Key
                    k_weight = (attn_proc.to_k_lora.up.weight @ attn_proc.to_k_lora.down.weight).to(device) * scale
                    base_layer.to_k.weight.data += k_weight
                    
                    # Value
                    v_weight = (attn_proc.to_v_lora.up.weight @ attn_proc.to_v_lora.down.weight).to(device) * scale
                    base_layer.to_v.weight.data += v_weight
                    
                    # Output
                    out_weight = (attn_proc.to_out_lora.up.weight @ attn_proc.to_out_lora.down.weight).to(device) * scale
                    base_layer.to_out[0].weight.data += out_weight
                    
                    merged_count += 1
                    
                    # Print debug information
                    # print(f"Merging layer {name}:")
                    # print(f"- Query weight range: [{q_weight.min():.6f}, {q_weight.max():.6f}]")
                    # print(f"- Key weight range: [{k_weight.min():.6f}, {k_weight.max():.6f}]")
                    # print(f"- Value weight range: [{v_weight.min():.6f}, {v_weight.max():.6f}]")
                    # print(f"- Output weight range: [{out_weight.min():.6f}, {out_weight.max():.6f}]")
        
        print(f"Successfully merged {merged_count} LoRA layer weights")
        return merged_unet
    
    def extract_lora_from_unet(self, original_unet, updated_unet):
        """Extract LoRA parameters from updated UNet, directly update existing LoRA processor parameters
        
        Args:
            original_unet: Original UNet model
            updated_unet: Updated UNet model
            
        Returns:
            lora_params: Updated LoRA parameter list (exactly matching the order and shape returned by get_lora_params())
        """
        print("\n=== Starting to extract LoRA parameters from updated UNet ===")
        
        lora_params = []
        extracted_count = 0
        
        # Traverse all attention layers and parameters in exactly the same order as get_lora_params
        for name, attn_proc in self.unet.attn_processors.items():
            if isinstance(attn_proc, LoRAAttnProcessor) and self._should_train_layer(name):
                # Get original layer and updated layer
                if name.endswith("attn1.processor"):
                    base_name = name.replace(".processor", "")
                    original_layer = original_unet.get_submodule(base_name)
                    updated_layer = updated_unet.get_submodule(base_name)
                else:
                    base_name = name.replace(".processor", "")
                    original_layer = original_unet.get_submodule(base_name)
                    updated_layer = updated_unet.get_submodule(base_name)
                
                # Calculate weight differences
                device = original_layer.to_q.weight.device
                
                # Calculate weight differences
                q_diff = updated_layer.to_q.weight - original_layer.to_q.weight
                k_diff = updated_layer.to_k.weight - original_layer.to_k.weight
                v_diff = updated_layer.to_v.weight - original_layer.to_v.weight
                out_diff = updated_layer.to_out[0].weight - original_layer.to_out[0].weight
                
                # Use SVD decomposition to extract LoRA parameters
                def extract_lora_weights(diff_weight, rank=4):
                    """Extract LoRA weights using SVD decomposition"""
                    U, S, V = torch.svd(diff_weight)
                    # Only keep the first rank singular values
                    U = U[:, :rank]
                    S = S[:rank]
                    V = V[:, :rank]
                    
                    # Calculate LoRA weights
                    lora_down = V.t()
                    lora_up = U @ torch.diag(S)
                    
                    return lora_up, lora_down
                
                # Extract LoRA parameters for each attention head
                q_up, q_down = extract_lora_weights(q_diff, self.rank)
                k_up, k_down = extract_lora_weights(k_diff, self.rank)
                v_up, v_down = extract_lora_weights(v_diff, self.rank)
                out_up, out_down = extract_lora_weights(out_diff, self.rank)
                
                # Directly update LoRA processor parameters, ensuring shapes match exactly
                with torch.no_grad():
                    # Update to_q_lora parameters
                    attn_proc.to_q_lora.up.weight.data = q_up.to(device)
                    attn_proc.to_q_lora.down.weight.data = q_down.to(device)
                    
                    # Update to_k_lora parameters
                    attn_proc.to_k_lora.up.weight.data = k_up.to(device)
                    attn_proc.to_k_lora.down.weight.data = k_down.to(device)
                    
                    # Update to_v_lora parameters
                    attn_proc.to_v_lora.up.weight.data = v_up.to(device)
                    attn_proc.to_v_lora.down.weight.data = v_down.to(device)
                    
                    # Update to_out_lora parameters
                    attn_proc.to_out_lora.up.weight.data = out_up.to(device)
                    attn_proc.to_out_lora.down.weight.data = out_down.to(device)
                
                # Collect parameters in exactly the same order as named_parameters() in get_lora_params
                # Order: to_q_lora.up.weight, to_q_lora.down.weight, to_k_lora.up.weight, to_k_lora.down.weight, 
                #       to_v_lora.up.weight, to_v_lora.down.weight, to_out_lora.up.weight, to_out_lora.down.weight
                for pname, param in attn_proc.named_parameters():
                    lora_params.append(param)
                
                extracted_count += 1
        
        print(f"Successfully extracted {extracted_count} LoRA layer parameters")
        return lora_params

    def forward(self, *args, **kwargs):
        """Forward pass, directly use UNet's forward"""
        return self.unet(*args, **kwargs) 