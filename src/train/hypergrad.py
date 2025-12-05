import torch
from typing import List, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad as torch_grad
import random

def get_outer_gradients(outer_loss: Tensor, params: List[Tensor], 
                       hparams: List[Tensor], retain_graph=True) -> Tuple[List[Tensor], List[Tensor]]:
    """Calculate gradients of outer loss with respect to parameters and hyperparameters"""
    print("\n=== Outer gradient calculation status ===")
    print(f"Outer loss value: {outer_loss.item():.6f}")
    print(f"Outer loss requires gradient: {outer_loss.requires_grad}")
    
    if outer_loss.requires_grad:
        outer_loss.backward()
        
        print("\nGradients of parameters (θ*):")
        for i, p in enumerate(params):
            if p.grad is not None:
                # print(f"Parameter {i} gradient norm: {p.grad.norm().item():.6f}")
                continue
            else:
                print(f"Parameter {i} has no gradient")
                
        print("\nGradients of hyperparameters (θ):")
        for i, p in enumerate(hparams):
            if p.grad is not None:
                # print(f"Hyperparameter {i} gradient norm: {p.grad.norm().item():.6f}")
                continue
            else:
                print(f"Hyperparameter {i} has no gradient")
    else:
        print("Warning: Outer loss does not require gradient, cannot calculate gradients")
    
    return [p.grad if p.grad is not None else torch.zeros_like(p) for p in params], [p.grad if p.grad is not None else torch.zeros_like(p) for p in hparams]

def cat_list_to_tensor(list_tx: List[Tensor]) -> Tensor:
    """Concatenate a list of tensors into a single vector"""
    return torch.cat([x.reshape([-1]) for x in list_tx])


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int,
                inner_optimizer: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                max_grad_norm: float = 1.0,
                max_retries: int = 5) -> Tuple[List[Tensor], float, dict]:
    """Calculate hypergradient using fixed-point iteration method
    
    Args:
        params: Model parameters
        hparams: Hyperparameters
        K: Number of fixed-point iterations
        inner_optimizer: Inner optimizer
        outer_loss: Outer loss function
        max_grad_norm: Maximum gradient norm limit
        max_retries: Maximum number of retries
        
    Returns:
        grads: Calculated gradients
        current_outer_loss: Current outer loss value
        grad_stats: Gradient statistics
    """
    
    # Initialize gradient statistics dictionary
    grad_stats = {
        'up_grads_mean': 0.0,
        'up_grads_max': 0.0,
        'up_grads_min': 0.0,
        'down_grads_mean': 0.0,
        'down_grads_max': 0.0,
        'down_grads_min': 0.0,
        'total_grad_norm': 0.0,
        'grad_update_skipped': False,
        'grad_clipped': False,
        'retry_count': 0,
        'convergence_failed': False
    }

    # Clone parameters and set requires_grad
    params = [nn.Parameter(w.detach().clone(), requires_grad=True) for w in params]
    hparams = [nn.Parameter(w.detach().clone(), requires_grad=True) for w in hparams]
    
    # Calculate outer loss and direct gradients
    print("\n=== Starting hypergradient calculation ===")
    o_loss = outer_loss(params, hparams)
    current_outer_loss = o_loss.item()
    
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)
    
    # Print statistics for grad_outer_w and grad_outer_hparams
    print("\n=== Direct gradient statistics ===")
    # grad_outer_w statistics
    w_norms = [g.norm().item() for g in grad_outer_w if g is not None]
    if w_norms:
        print("grad_outer_w (direct parameter gradients):")
        print(f"- Average norm: {sum(w_norms) / len(w_norms):.6f}")
        print(f"- Maximum norm: {max(w_norms):.6f}")
    
    # grad_outer_hparams statistics
    h_norms = [g.norm().item() for g in grad_outer_hparams if g is not None]
    if h_norms:
        print("\ngrad_outer_hparams (direct hyperparameter gradients):")
        print(f"- Average norm: {sum(h_norms) / len(h_norms):.6f}")
        print(f"- Maximum norm: {max(h_norms):.6f}")
    
    # Retry mechanism
    current_lr = 1.0  # Initial learning rate
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            print(f"\n=== Attempt {retry_count + 1}, learning rate: {current_lr:.6f} ===")
            
            # Create inner optimizer with current learning rate
            if hasattr(inner_optimizer, 'step_size'):
                # If it's GradientDescent, update step_size
                if callable(inner_optimizer.step_size):
                    original_step_size = inner_optimizer.step_size
                    inner_optimizer.step_size = lambda x: current_lr
                else:
                    original_step_size = inner_optimizer.step_size
                    inner_optimizer.step_size = current_lr
            elif hasattr(inner_optimizer, 'lr'):
                # If it's Adam, update lr
                original_lr = inner_optimizer.lr
                inner_optimizer.lr = current_lr
            
            # Perform one inner optimization update
            w_mapped = inner_optimizer(params, hparams, create_graph=True)
            
            # Fixed-point iteration
            vs = [torch.zeros_like(w) for w in params]
            vs_vec = cat_list_to_tensor(vs)
            
            print("\n=== Starting fixed-point iteration ===")
            jacobian_ratio_failed = False
            
            for k in range(K):
                vs_prev_vec = vs_vec
                
                # Record vs norms before Jacobian-vector product
                vs_norms_before = []
                for v in vs:
                    if v is not None:
                        vs_norms_before.append(v.norm().item())
                
                # Calculate Jacobian-vector product
                vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
                
                # Calculate average change ratio
                ratios = []
                for v_after, norm_before in zip(vs, vs_norms_before):
                    if v_after is not None and norm_before > 0:
                        norm_after = v_after.norm().item()
                        ratios.append(norm_after / norm_before)
                
                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)
                    print(f"\n[Iteration {k+1}] Average amplification ratio of Jacobian-vector product: {avg_ratio:.6f}")
                    
                    # Check if amplification ratio is greater than 1
                    if avg_ratio > 1.0:
                        print(f"Warning: Average amplification ratio of Jacobian-vector product ({avg_ratio:.6f}) is greater than 1, iteration does not converge")
                        jacobian_ratio_failed = True
                        break
                
                # Update vs
                vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
                vs_vec = cat_list_to_tensor(vs)
                
                # Calculate residual
                residual = float(torch.norm(vs_vec - vs_prev_vec))
                print(f"[Fixed-point iteration step {k+1}/{K}] Residual: {residual:.16f}")
                
                if residual < 1e-10:
                    print(f"[Fixed-point iteration] Converged at step {k+1}")
                    break
            
            # If Jacobian-vector product amplification ratio is greater than 1, retry
            if jacobian_ratio_failed:
                retry_count += 1
                current_lr *= 0.5
                print(f"Jacobian-vector product amplification ratio too large, retry {retry_count}, new learning rate: {current_lr:.6f}")
                continue
            
            # Record vs statistics before calculating final gradients
            print("\n=== Final vs statistics (before final gradient calculation) ===")
            vs_norms = [v.norm().item() for v in vs if v is not None]
            if vs_norms:
                print("vs norms:")
                print(f"- Average norm: {sum(vs_norms) / len(vs_norms):.6f}")
                print(f"- Maximum norm: {max(vs_norms):.6f}")
            
            # Calculate final gradients
            try:
                # Record vs state before torch_grad
                vs_before = [v.clone() if v is not None else None for v in vs]
                
                # Calculate torch_grad
                grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
                
                # Compare changes
                print("\n=== Jacobian transpose effect ===")
                ratios = []
                for v_before, grad in zip(vs_before, grads):
                    if v_before is not None and grad is not None:
                        ratio = grad.norm().item() / v_before.norm().item()
                        ratios.append(ratio)
                
                if ratios:
                    print(f"Average amplification ratio of vs after Jacobian transpose: {sum(ratios) / len(ratios):.6f}")
                    print(f"Maximum amplification ratio: {max(ratios):.6f}")
                
                print(f"Number of non-zero or non-none elements in grads: {sum(1 for g in grads if g is not None and torch.any(g != 0))}")
                grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]
                
                # Calculate gradient statistics
                up_grads = []
                down_grads = []
                up_indices = []
                down_indices = []
                
                # First collect and classify gradients
                for i, g in enumerate(grads):
                    if g is not None:
                        grad_norm = g.norm().item()
                        if g.shape[0] > g.shape[1]:  # up matrix
                            up_grads.append(g)
                            up_indices.append(i)
                        else:  # down matrix
                            down_grads.append(g)
                            down_indices.append(i)

                # Update gradient statistics
                if up_grads:
                    grad_stats['up_grads_mean'] = sum(g.norm().item() for g in up_grads) / len(up_grads)
                    grad_stats['up_grads_max'] = max(g.norm().item() for g in up_grads)
                    grad_stats['up_grads_min'] = min(g.norm().item() for g in up_grads)
                
                if down_grads:
                    grad_stats['down_grads_mean'] = sum(g.norm().item() for g in down_grads) / len(down_grads)
                    grad_stats['down_grads_max'] = max(g.norm().item() for g in down_grads)
                    grad_stats['down_grads_min'] = min(g.norm().item() for g in down_grads)

                # Calculate total gradient norm
                total_grad_norm = torch.sqrt(sum(g.norm()**2 for g in grads if g is not None))
                grad_stats['total_grad_norm'] = total_grad_norm.item()
                grad_stats['retry_count'] = retry_count

                # Check if total gradient norm exceeds limit
                if total_grad_norm > max_grad_norm:
                    retry_count += 1
                    current_lr *= 0.5
                    print(f"Total gradient norm ({total_grad_norm:.6f}) exceeds limit ({max_grad_norm:.6f}), retry {retry_count}, new learning rate: {current_lr:.6f}")
                    continue

                # Successfully calculated gradients
                print(f"Successfully calculated hypergradient, total gradient norm: {total_grad_norm:.6f}")
                return grads, current_outer_loss, grad_stats
                
            except Exception as e:
                print(f"Final gradient calculation error: {str(e)}")
                retry_count += 1
                current_lr *= 0.5
                print(f"Gradient calculation error, retry {retry_count}, new learning rate: {current_lr:.6f}")
                continue
                
        except Exception as e:
            print(f"Fixed-point iteration error: {str(e)}")
            retry_count += 1
            current_lr *= 0.5
            print(f"Fixed-point iteration error, retry {retry_count}, new learning rate: {current_lr:.6f}")
            continue
    
    # Exceeded maximum retry count, return direct gradients
    print(f"Exceeded maximum retry count ({max_retries}), returning direct gradients grad_outer_hparams")
    grad_stats['grad_update_skipped'] = True
    grad_stats['convergence_failed'] = True
    grad_stats['retry_count'] = retry_count
    
    return grad_outer_hparams, current_outer_loss, grad_stats

class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult=1):
        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_loss(self, params, hparams):
        self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1)
        self.step_size = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph=False):
        params = [p if isinstance(p, nn.Parameter) else nn.Parameter(p, requires_grad=True) for p in params]
        loss = self.get_loss(params, hparams)
        
        # Calculate step size
        sz = self.step_size(hparams) if callable(self.step_size) else self.step_size
        
        # Don't need to save computation graph during inner optimization, only needed in fixed_point
        grads = torch.autograd.grad(
            loss, 
            params,
            create_graph=create_graph,  # Don't save computation graph during inner optimization
            allow_unused=False,
            retain_graph=create_graph   # Don't retain computation graph during inner optimization
        )
        new_params = []
        for p, g in zip(params, grads):
            new_params.append(p - sz * g)
            
        return new_params



class BiLevelOptimizer:
    def __init__(self,
                 params: List[Tensor],
                 hparams: List[Tensor],
                 inner_optimizers: List[DifferentiableOptimizer],
                 outer_optimizer: torch.optim.Optimizer,
                 inner_train_loaders: dict,
                 set_current_loader,  # Function to set data loader
                 K: int = 5,
                 device: str = 'cuda',
                 max_grad_norm: float = 1.0,
                 meta_rate: float = 1.0,  # meta_rate parameter
                 prior_loaders: dict = None,  # prior_loaders parameter
                 inner_sample_k: int = None,  # Number of inner configurations to randomly sample per step
                 safety_model=None,  # safety_model parameter
                 noise_scheduler=None):  # noise_scheduler parameter
        self.params = params
        self.hparams = hparams
        self.inner_optimizers = inner_optimizers
        self.outer_optimizer = outer_optimizer
        self.inner_train_loaders = inner_train_loaders
        self.set_current_loader = set_current_loader
        self.current_inner_idx = 0
        self.K = K
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.meta_rate = meta_rate
        self.prior_loaders = prior_loaders  # Save prior_loaders
        self.accumulated_grads = None
        self.accumulated_loss = 0.0
        self.accumulated_stats = []
        self.inner_sample_k = inner_sample_k
        self.safety_model = safety_model  # Save safety_model
        self.noise_scheduler = noise_scheduler  # Save noise_scheduler

    def reset_accumulation(self):
        """Reset gradient accumulator"""
        self.accumulated_grads = None
        self.accumulated_loss = 0.0
        self.accumulated_stats = []

    def accumulate_grads(self, grads, loss, stats):
        """Accumulate gradients and statistics"""
        if self.accumulated_grads is None:
            self.accumulated_grads = [torch.zeros_like(g) if g is not None else None for g in grads]
        
        # Accumulate gradients
        for i, g in enumerate(grads):
            if g is not None and self.accumulated_grads[i] is not None:
                self.accumulated_grads[i] += g
        
        # Accumulate loss
        self.accumulated_loss += loss
        
        # Accumulate statistics
        self.accumulated_stats.append(stats)

    def step(self, inner_loss, outer_loss, process):
        try:
            # Reset accumulator
            self.reset_accumulation()
            
            # Select inner configuration indices to use in this step
            all_indices = list(range(len(self.inner_optimizers)))
            if self.inner_sample_k is not None and self.inner_sample_k > 0:
                sample_k = min(self.inner_sample_k, len(all_indices))
                selected_indices = random.sample(all_indices, sample_k)
                print(f"\nRandomly sampling {sample_k}/{len(all_indices)} configurations for inner optimization in this step: {selected_indices}")
            else:
                selected_indices = all_indices

            # Iterate over selected inner configurations
            for config_idx in selected_indices:
                # 1. Get current inner optimizer and data loader
                current_optimizer = self.inner_optimizers[config_idx]
                current_loader = self.inner_train_loaders[current_optimizer.config_name]
                
                # Set current data loader
                print(f"\nUsing configuration {config_idx+1}/{len(self.inner_optimizers)}: {current_optimizer.config_name}")
                self.set_current_loader(current_loader)

                # 2. Copy hyperparameters for optimization
                opt_params = [p.detach().clone().requires_grad_(True) for p in self.hparams]
                
                # 3. Inner optimization
                # Calculate the number of inner optimization steps to use
                default_inner_steps = 30
                if hasattr(current_optimizer, 'inner_steps'):
                    current_inner_steps = current_optimizer.inner_steps
                    print(f"\nUsing fixed inner steps: {current_inner_steps}")
                else:
                    current_inner_steps = default_inner_steps
                    print(f"\nUsing default inner steps: {current_inner_steps}")
                
                # Check if using full parameter finetuning
                use_full_param = hasattr(current_optimizer, 'config') and getattr(current_optimizer.config, 'full', False)
                
                if use_full_param:
                    print(f"\n[Full parameter finetuning mode] Starting inner optimization...")
                    # Full parameter finetuning mode
                    if self.safety_model is None or self.noise_scheduler is None:
                        print("Warning: Full parameter finetuning mode requires safety_model and noise_scheduler instances, skipping for now")
                        # Continue using LoRA mode
                        for step in range(current_inner_steps):
                            print(f"\n[Inner optimization step {step+1}/{current_inner_steps}]")
                            opt_params = current_optimizer(opt_params, self.hparams, create_graph=False)
                    else:
                        # 1. Merge LoRA parameters into original UNet
                        merged_unet = self.safety_model.merge_lora_to_unet(opt_params)
                        
                        # 2. Enable gradient computation for merged UNet
                        for param in merged_unet.parameters():
                            param.requires_grad = True
                        
                        # 3. Create full parameter optimizer
                        full_param_optimizer = torch.optim.AdamW(
                            merged_unet.parameters(),
                            lr=current_optimizer.lr if hasattr(current_optimizer, 'lr') else 1e-4,
                            betas=(0.9, 0.999),
                            eps=1e-8
                        )
                        
                        # 4. Execute full parameter optimization
                        for step in range(current_inner_steps):
                            print(f"\n[Full parameter inner optimization step {step+1}/{current_inner_steps}]")
                            
                            # Get current batch data
                            batch = next(iter(current_loader))
                            latents = batch['latents']
                            timesteps = batch['timesteps'].squeeze()
                            text_embeddings = batch['text_embeddings']
                            
                            # Ensure dimensions are correct
                            if len(latents.shape) == 3:
                                latents = latents.unsqueeze(0)
                            if len(timesteps.shape) == 0:
                                timesteps = timesteps.unsqueeze(0)
                            if len(text_embeddings.shape) == 4 and text_embeddings.shape[1] == 1:
                                text_embeddings = text_embeddings.squeeze(1)
                            
                            # Forward pass
                            full_param_optimizer.zero_grad()
                            
                            # Generate noise
                            noise = torch.randn_like(latents)
                            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                            
                            # Predict noise
                            model_pred = merged_unet(
                                noisy_latents,
                                timesteps,
                                text_embeddings,
                                return_dict=False
                            )[0]
                            
                            # Calculate loss
                            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                            
                            # Backward pass
                            loss.backward()
                            full_param_optimizer.step()
                            
                            print(f"Full parameter loss: {loss.item():.6f}")
                        
                        # 5. Extract LoRA parameters from updated UNet
                        original_unet = self.safety_model.unet
                        extracted_lora_params = self.safety_model.extract_lora_from_unet(original_unet, merged_unet)

                        
                        # 6. Update opt_params to extracted LoRA parameters
                        # Now extracted_lora_params is already the updated LoRA processor parameters, with shape exactly matching original parameters
                        opt_params = [param.clone().detach().requires_grad_(True) for param in extracted_lora_params]
                        
                        # 7. Clean up memory
                        del merged_unet, full_param_optimizer
                        torch.cuda.empty_cache()
                    
                else:
                    # LoRA finetuning mode (original logic)
                    print(f"\n[LoRA finetuning mode] Starting inner optimization...")
                    for step in range(current_inner_steps):
                        print(f"\n[Inner optimization step {step+1}/{current_inner_steps}]")
                        opt_params = current_optimizer(opt_params, self.hparams, create_graph=False)
                
                # Update self.params
                for i, param in enumerate(opt_params):
                    self.params[i] = param.clone().detach().requires_grad_(True)
                
                # 4. Create gradient descent optimizer corresponding to current inner optimizer
                if isinstance(current_optimizer, DifferentiableAdam):
                    # If it's Adam optimizer, create corresponding GD optimizer
                    current_gd = GradientDescent(
                        current_optimizer.loss_f,
                        step_size=current_optimizer.lr  # Use Adam's learning rate
                    )
                else:
                    # If it's already a GD optimizer, use directly
                    current_gd = current_optimizer

                # 4.1 Construct wrapped loss with fixed L2(0.5) regularization for fixed_point stage, and create GD optimizer only for fixed_point
                def _fixed_point_inner_loss(params, hparams):
                    base_loss = current_optimizer.loss_f(params, hparams)
                    l2_reg = sum(((p - h) ** 2).sum() for p, h in zip(params, hparams))
                    # l2_reg = torch.tensor(0.0, device=params[0].device)
                    # for p, h in zip(params, hparams):
                    #     l2_reg = l2_reg + torch.norm(p - h, p=2)**2
                    return base_loss + 0.5 * l2_reg

                # Inherit original step size setting (if GD, reuse its step_size, can be scalar or callable; if Adam, use its lr)
                if isinstance(current_optimizer, GradientDescent):
                    fp_step_size = current_optimizer.step_size
                else:
                    fp_step_size = current_optimizer.lr

                current_gd_for_fixed_point = GradientDescent(
                    _fixed_point_inner_loss,
                    step_size=1
                    # step_size=fp_step_size
                )
                
                # 5. Calculate hypergradient and outer loss
                print("\nStarting hypergradient calculation...")
                print("\nUsing fixed-point iteration method to calculate hypergradient...")
                self.outer_optimizer.zero_grad()
                
                grads, current_outer_loss, grad_stats = fixed_point(
                    params=self.params,
                    hparams=self.hparams,
                    K=self.K,
                    inner_optimizer=current_gd_for_fixed_point,
                    outer_loss=outer_loss,
                    max_grad_norm=self.max_grad_norm
                )

                # Accumulate gradients and statistics
                self.accumulate_grads(grads, current_outer_loss, grad_stats)

            # Calculate average gradients and loss (normalized by number of sampled configurations)
            n_configs = len(selected_indices) if 'selected_indices' in locals() else len(self.inner_optimizers)
            if self.accumulated_grads is not None:
                # Calculate average gradients and apply meta_rate
                for i, g in enumerate(self.accumulated_grads):
                    if g is not None:
                        g = g / n_configs * self.meta_rate
                        self.hparams[i].grad = g

                # Calculate average loss
                avg_loss = self.accumulated_loss / n_configs

                # Update outer parameters
                self.outer_optimizer.step()

                # Merge statistics
                merged_stats = {}
                for key in self.accumulated_stats[0].keys():
                    if key in ['grad_update_skipped', 'grad_clipped']:
                        # For boolean fields, if any configuration is True, the final is True
                        merged_stats[key] = any(s[key] for s in self.accumulated_stats)
                    elif isinstance(self.accumulated_stats[0][key], (int, float)):
                        merged_stats[key] = sum(s[key] for s in self.accumulated_stats) / n_configs
                    else:
                        merged_stats[key] = self.accumulated_stats[0][key]  # For non-numeric types, take the first one

                return avg_loss, merged_stats
            else:
                return 0.0, {}

        except Exception as e:
            print(f"BiLevelOptimizer.step error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Error stack:\n{traceback.format_exc()}")
            return 0.0, {}

class DifferentiableAdam(DifferentiableOptimizer):
    """Differentiable Adam optimizer implementation"""
    def __init__(self, loss_f, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, inner_steps=30, config_name=""):
        super(DifferentiableAdam, self).__init__(loss_f)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.inner_steps = inner_steps
        self.config_name = config_name
        self.state = {}  # Store optimizer state

    def init_state(self, params):
        """Initialize Adam state"""
        if not self.state:
            self.state = {
                'step': 0,
                'm': [torch.zeros_like(p) for p in params],
                'v': [torch.zeros_like(p) for p in params]
            }

    def step(self, params, hparams, create_graph):
        """Execute one step of Adam optimization"""
        params = [p if isinstance(p, nn.Parameter) else nn.Parameter(p, requires_grad=True) for p in params]
        self.init_state(params)
        
        # Calculate loss and gradients
        loss = self.get_loss(params, hparams)
        # Don't need to save computation graph during inner optimization, only needed in fixed_point
        grads = torch.autograd.grad(
            loss, params,
            create_graph=False,  # Don't save computation graph during inner optimization
            allow_unused=False,
            retain_graph=False   # Don't retain computation graph during inner optimization
        )

        # Update state
        self.state['step'] += 1
        beta1, beta2 = self.betas
        step = self.state['step']
        
        # Bias correction coefficients (convert to tensor)
        bias_correction1 = torch.tensor(1 - beta1 ** step, device=params[0].device)
        bias_correction2 = torch.tensor(1 - beta2 ** step, device=params[0].device)

        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            # Update momentum
            self.state['m'][i] = beta1 * self.state['m'][i] + (1 - beta1) * g
            self.state['v'][i] = beta2 * self.state['v'][i] + (1 - beta2) * g * g
            
            # Calculate bias-corrected estimates
            m_hat = self.state['m'][i] / bias_correction1
            v_hat = self.state['v'][i] / bias_correction2
            
            # Calculate step size (using tensor operations)
            step_size = self.lr * torch.sqrt(1 - torch.tensor(beta2 ** step, device=p.device)) / (1 - torch.tensor(beta1 ** step, device=p.device))
            
            # Update parameters
            new_p = p - step_size * m_hat / (torch.sqrt(v_hat) + self.eps)
            new_params.append(new_p)

        return new_params

