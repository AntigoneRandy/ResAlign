import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
import argparse
import os
from tqdm import tqdm
import gc
import copy
import torch.nn as nn
from diffusers.models.attention_processor import LoRAAttnProcessor
from src.train.hypergrad import BiLevelOptimizer, GradientDescent, DifferentiableOptimizer, DifferentiableAdam
from diffusers.optimization import get_scheduler
import datetime
import json
import pandas as pd
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from torch.utils.data import DataLoader
# from nudenet import NudeDetector
from pathlib import Path
import csv

from src.train.models import SafetyModel
from src.train.utils import load_models, load_nsfw_dataset, compute_esd_loss, compute_diffusion_loss,get_text_embeddings, inference_with_lora, NSFWDataset

@dataclass
class InnerLoopConfig:
    """Inner loop optimization configuration class"""
    data_dir: str  # Data directory
    learning_rate: float  # Learning rate
    inner_steps: int  # Number of inner optimization steps
    batch_size: int  # Batch size
    optimizer_type: str = "gd"  # Optimizer type, default is gradient descent
    name: str = ""  # Configuration name
    finetune_method: str = "lora"  # Finetuning method, options: "lora" or "dreambooth"
    prior_preservation_data_dir: str = None  # Data directory for prior preservation, only used in dreambooth mode
    full: bool = False  # Whether to use full parameter finetuning, default is False (use LoRA)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"lr{self.learning_rate}_steps{self.inner_steps}_{self.optimizer_type}"
        # Validate finetuning method
        if self.finetune_method not in ["lora", "dreambooth"]:
            raise ValueError(f"Unsupported finetuning method: {self.finetune_method}")
        # Validate that prior preservation data directory must be provided in dreambooth mode
        if self.finetune_method == "dreambooth" and not self.prior_preservation_data_dir:
            raise ValueError("prior_preservation_data_dir must be provided in dreambooth mode")

def create_output_dir(args):
    """Create output directory and return path"""
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # Create directory name
    dir_name = f"{args.num_epochs}epochs-{timestamp}"
    # Full output path
    output_path = os.path.join(args.output_dir, dir_name)
    # Create directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save configuration parameters
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w") as f:
        # Convert arguments to dictionary
        config_dict = vars(args)
        # Special handling for InnerLoopConfig objects
        if 'inner_loop_configs' in config_dict:
            config_dict['inner_loop_configs'] = [
                {
                    'data_dir': cfg.data_dir,
                    'learning_rate': cfg.learning_rate,
                    'inner_steps': cfg.inner_steps,
                    'batch_size': cfg.batch_size,
                    'optimizer_type': cfg.optimizer_type,
                    'name': cfg.name,
                    'finetune_method': cfg.finetune_method,
                    'prior_preservation_data_dir': cfg.prior_preservation_data_dir,
                    'full': cfg.full  # Add missing full field
                }
                for cfg in config_dict['inner_loop_configs']
            ]
        json.dump(config_dict, f, indent=4)
    
    return output_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str, 
                        default="CompVis/stable-diffusion-v1-4",
                        choices=["CompVis/stable-diffusion-v1-4",
                                 "runwayml/stable-diffusion-v1-5",
                                 "lykon/dreamshaper-8"]
                        )
    parser.add_argument("--local_unet_path", type=str, default=None,
                        help="Local UNet weight path, if provided, load UNet weights from local path")
    parser.add_argument("--data_dir", type=str, default="data/imma_train")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--inner_learning_rate", type=float, default=2e-3) # The inner learning rate here controls the step size for one optimization step when using fixed-point iteration for outer optimization
    parser.add_argument("--outer_lambda", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_device", type=int, default=0,
                      help="Specify the CUDA device number to use")
    parser.add_argument("--inner_steps", type=int, default=30,
                      help="Number of inner optimization steps")
    parser.add_argument("--K", type=int, default=5,
                      help="Number of fixed-point iteration steps")
    parser.add_argument("--train_method", type=str, default="xattn",
                      choices=["xattn", "noxattn", "full", "xattn-strict"],
                      help="LoRA training method")
    parser.add_argument("--erase_concept", type=str, default="nudity",
                      help="Concept to remove")
    parser.add_argument("--erase_from", type=str, default="people",
                        help="Where to remove from")
    parser.add_argument("--negative_guidance", type=float, default=1.0,
                      help="Negative guidance strength")
    parser.add_argument("--retain_loss_weight", type=float, default=0.2,
                      help="Retain loss weight")
    parser.add_argument("--inference_interval", type=int, default=100,
                      help="Run inference every N epochs")
    parser.add_argument("--lora_up_std", type=float, default=0.002,
                      help="Standard deviation for LoRA up matrix initialization")
    parser.add_argument("--lora_down_std", type=float, default=0.008,
                      help="Standard deviation for LoRA down matrix initialization")
    parser.add_argument("--lora_rank", type=int, default=4,
                      help="LoRA rank, controls the dimension of low-rank decomposition")
    parser.add_argument("--dynamic_weights", action="store_true",
                      help="Whether to use dynamic weights")
    parser.add_argument("--final_outer_lambda", type=float, default=0.8,
                      help="outer_lambda value at the end of training")
    parser.add_argument("--final_retain_loss_weight", type=float, default=0.4,
                      help="retain_loss_weight value at the end of training")
    parser.add_argument("--weight_rampup_epochs", type=float, default=0.5,
                      help="Epoch proportion when weights start to increase, default 0.5 means weights start to increase at 50% of training progress")
    parser.add_argument("--max_grad_norm", type=float, default=0.2,
                      help="Maximum gradient norm limit, gradient updates exceeding this value will be ignored")
    parser.add_argument("--inner_configs", type=str, nargs="+", default=None,
                      help="List of inner optimization configuration file paths")
    parser.add_argument("--outer_data_dir", type=str, default="data/women_plus_man_train",
                      help="Data directory used for outer loss function calculation")
    parser.add_argument("--outer_batch_size", type=int, default=1,
                      help="Batch size for outer optimization")
    parser.add_argument("--retain_data_dir", type=str, default="data/for_medium_retain",
                      help="Dataset directory used for retention mode")
    parser.add_argument("--retain_batch_size", type=int, default=4,
                      help="Batch size for retention")
    parser.add_argument("--meta_rate", type=float, default=0.5,
                      help="meta_rate parameter")
    parser.add_argument("--loss_mode", type=str, default="ga",
                      choices=["ga", "esd"],
                      help="Outer loss function based on ga or esd")
    parser.add_argument("--val_prompt2", type=str, default="a nude beautiful woman")
    parser.add_argument("--val_prompt4", type=str, default="symmetrical oil painting of full - body women by samokhvalov")
    parser.add_argument("--inner_sample_k", type=int, default=3,
                      help="Number of inner configurations to randomly sample per step; if not set, use all configurations")
    
    args = parser.parse_args()
    
    # If no inner configuration is provided, create default configuration
    if not args.inner_configs:
        default_config = InnerLoopConfig(
            data_dir=args.data_dir,
            learning_rate=args.inner_learning_rate,
            inner_steps=args.inner_steps,
            batch_size=args.batch_size,
            optimizer_type="gd",
            finetune_method="lora"
        )
        args.inner_loop_configs = [default_config]
    else:
        # Load inner configuration from JSON files
        args.inner_loop_configs = []
        print("\nLoading inner configuration files:")
        for config_path in args.inner_configs:
            print(f"- Loading: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                print(f"  Configuration content: {config_dict}")
                # Ensure all required fields exist
                required_fields = ['data_dir', 'learning_rate', 'inner_steps', 'batch_size', 'optimizer_type']
                for field in required_fields:
                    if field not in config_dict:
                        raise ValueError(f"Configuration file {config_path} missing required field: {field}")
                
                # Create InnerLoopConfig object
                config = InnerLoopConfig(
                    data_dir=config_dict['data_dir'],
                    learning_rate=float(config_dict['learning_rate']),
                    inner_steps=int(config_dict['inner_steps']),
                    batch_size=int(config_dict['batch_size']),
                    optimizer_type=str(config_dict['optimizer_type']),
                    name=str(config_dict.get('name', '')),
                    finetune_method=str(config_dict.get('finetune_method', 'lora')),
                    prior_preservation_data_dir=config_dict.get('prior_preservation_data_dir'),
                    full=bool(config_dict.get('full', False))
                )
                print(f"  Created configuration object: {vars(config)}")
                args.inner_loop_configs.append(config)
    
    return args

def create_inner_optimizer(config: InnerLoopConfig, loss_fn) -> DifferentiableOptimizer:
    """Create inner optimizer based on configuration"""
    # If a dictionary is passed, convert it to InnerLoopConfig object first
    if isinstance(config, dict):
        config = InnerLoopConfig(**config)
    
    if config.optimizer_type.lower() == "gd":
        optimizer = GradientDescent(loss_fn, config.learning_rate)
        optimizer.inner_steps = config.inner_steps  # Add inner_steps attribute
        optimizer.config_name = config.name  # Add config_name attribute
        return optimizer
    elif config.optimizer_type.lower() == "adam":
        return DifferentiableAdam(
            loss_fn,
            lr=config.learning_rate,
            inner_steps=config.inner_steps,
            config_name=config.name
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

def create_loss_functions(safety_model, outer_train_loader, noise_scheduler, concept_embeddings, 
                       base_embeddings, empty_embeddings, tokenizer, text_encoder, vae, args, prior_loaders=None):
    """Create inner and outer loss functions
    
    Args:
        safety_model: Safety model (with LoRA)
        outer_train_loader: Outer training data loader
        noise_scheduler: Noise scheduler
        concept_embeddings: Concept embeddings
        base_embeddings: Embeddings of the concept to erase from
        empty_embeddings: Empty text embeddings
        tokenizer: Tokenizer
        text_encoder: Text encoder
        vae: VAE model
        args: Argument configuration
        prior_loaders: Dictionary of prior preservation data loaders for dreambooth mode
    """
    
    # Load retention dataset (medium_retain mode)
    retain_batch_size = args.retain_batch_size  
    retain_loss_w = args.retain_loss_weight    # Use command line arguments
    
    # Load real image dataset for medium retention mode
    if True:  # Always use medium_retain mode
        retain_dataset = NSFWDataset(
            data_dir=args.retain_data_dir,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            device=args.device,
            use_metadata_prompts=True
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=args.retain_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=retain_dataset.collate_fn,
            drop_last=True
        )
    
    # Create a completely clean UNet (without LoRA injection)
    from diffusers import UNet2DConditionModel
    original_unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet"
    ).to(args.device)
    
    # Set to evaluation mode and freeze parameters
    original_unet.eval()
    for param in original_unet.parameters():
        param.requires_grad = False
    
    # Add current epoch counter
    current_epoch = 0

    # Add current data loader
    current_train_loader = [None]  # Use list to allow modification in closure
    
    def set_current_loader(loader):
        """Set the current data loader"""
        current_train_loader[0] = loader
    
    def update_epoch():
        """Update current epoch count"""
        nonlocal current_epoch
        current_epoch += 1
        
    def get_dynamic_weights():
        """Calculate dynamic weights for current epoch"""
        if not args.dynamic_weights or current_epoch < args.num_epochs * args.weight_rampup_epochs:
            return args.outer_lambda, args.retain_loss_weight
            
        # Calculate weight increase progress (between 0 and 1)
        progress = (current_epoch - args.num_epochs * args.weight_rampup_epochs) / (args.num_epochs * (1 - args.weight_rampup_epochs))
        progress = min(max(progress, 0), 1)  # Ensure between 0-1
        
        # Use smooth sigmoid function for interpolation
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
        
        # Calculate current weights
        current_outer_lambda = args.outer_lambda + (args.final_outer_lambda - args.outer_lambda) * sigmoid(progress)
        current_retain_loss_w = args.retain_loss_weight + (args.final_retain_loss_weight - args.retain_loss_weight) * sigmoid(progress)
        
        return current_outer_lambda, current_retain_loss_w

    def create_inner_loss(config=None, prior_loaders=None):
        """Factory function to create inner loss function
        
        Args:
            config: InnerLoopConfig object containing finetuning method and other configurations
            prior_loaders: Dictionary of prior preservation data loaders for dreambooth mode
        """
        def inner_loss(params, hparams):
            try:
                # Ensure current data loader exists
                if current_train_loader[0] is None:
                    raise ValueError("Current data loader is not set!")

                # Get current batch data
                batch = next(iter(current_train_loader[0]))
                latents = batch['latents']
                timesteps = batch['timesteps'].squeeze()
                
                # Process text embeddings based on finetuning method
                if config and config.finetune_method == "dreambooth":
                    # Check if prior_loaders is available
                    if not prior_loaders or config.name not in prior_loaders:
                        raise ValueError(f"Prior loader for configuration '{config.name}' not found")
                    
                    # Force use "a photo of *s person" as prompt
                    text_input = tokenizer(
                        "a photo of *s person",
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(args.device)
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(text_input.input_ids)[0]
                else:
                    # lora mode uses original text embeddings
                    encoder_hidden_states = batch['text_embeddings']
                
                # Ensure dimensions are correct
                if len(latents.shape) == 3:
                    latents = latents.unsqueeze(0)
                if len(timesteps.shape) == 0:
                    timesteps = timesteps.unsqueeze(0)
                if len(encoder_hidden_states.shape) == 4 and encoder_hidden_states.shape[1] == 1:
                    encoder_hidden_states = encoder_hidden_states.squeeze(1)
                
                # Update LoRA parameters
                safety_model.update_lora_params(params)
                
                # Generate noise
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                model_pred = safety_model.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]
                
                # Calculate base MSE loss
                mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = mse_loss
                
                # If in dreambooth mode, add prior preservation loss
                if config and config.finetune_method == "dreambooth":
                    # Get prior preservation data
                    prior_batch = next(iter(prior_loaders[config.name]))
                    
                    prior_latents = prior_batch['latents']
                    prior_timesteps = prior_batch['timesteps']
                    prior_embeddings = prior_batch['text_embeddings']
                    
                    # Ensure dimensions are correct
                    if len(prior_latents.shape) == 3:
                        prior_latents = prior_latents.unsqueeze(0)
                    if len(prior_timesteps.shape) > 1:
                        prior_timesteps = prior_timesteps.squeeze()
                    if len(prior_timesteps.shape) == 0:
                        prior_timesteps = prior_timesteps.unsqueeze(0)
                    if len(prior_embeddings.shape) == 4 and prior_embeddings.shape[1] == 1:
                        prior_embeddings = prior_embeddings.squeeze(1)
                    
                    # Generate prior noise
                    prior_noise = torch.randn_like(prior_latents)
                    prior_noisy_latents = noise_scheduler.add_noise(prior_latents, prior_noise, prior_timesteps)
                    
                    # Predict prior noise
                    prior_pred = safety_model.unet(
                        prior_noisy_latents,
                        prior_timesteps,
                        prior_embeddings,
                        return_dict=False
                    )[0]
                    
                    # Calculate prior preservation loss
                    prior_loss = F.mse_loss(prior_pred.float(), prior_noise.float(), reduction="mean")
                    loss = loss + prior_loss
                
                return loss

            except Exception as e:
                print(f"Inner loss calculation error: {str(e)}")
                raise e
        return inner_loss
    
    def outer_loss(params, hparams=None):
        try:
            if not params or not hparams:
                return torch.tensor(0.0, device=args.device, requires_grad=True)
                
            # Get current dynamic weights
            current_outer_lambda, current_retain_loss_w = get_dynamic_weights()
            
            # Ensure parameters require gradients
            theta_star = params  # Parameters after inner optimization
            theta = hparams      # Original parameters
            
            if not any(p.requires_grad for p in theta_star) or not any(p.requires_grad for p in theta):
                print("Warning: Parameters do not have requires_grad")
                theta_star = [p.detach().requires_grad_(True) for p in theta_star]
                theta = [p.detach().requires_grad_(True) for p in theta]
            
            # Get batch data (using outer data loader)
            batch = next(iter(outer_train_loader))
            
            batch_latents = batch['latents']
            t = batch['timesteps'].squeeze()
            
            # Get text embeddings for outer batch
            if 'text_embeddings' in batch:
                outer_text_embeddings = batch['text_embeddings']
            else:
                prompts = batch['prompts']
                text_input = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(args.device)
                with torch.no_grad():
                    outer_text_embeddings = text_encoder(text_input.input_ids)[0]
            
            # Calculate regularization term for performance retention (medium_retain mode)
            retain_loss = torch.tensor(0.0, device=args.device)
            
            # Medium retention mode
            try:
                retain_batch = next(iter(retain_loader))
                
                retain_latents = retain_batch['latents']
                retain_timesteps = retain_batch['timesteps'].squeeze()  # Ensure timesteps are 1D
                retain_embeddings = retain_batch['text_embeddings']
                
                # Ensure dimensions are correct
                if len(retain_latents.shape) == 3:
                    retain_latents = retain_latents.unsqueeze(0)
                if len(retain_timesteps.shape) == 0:
                    retain_timesteps = retain_timesteps.unsqueeze(0)
                if len(retain_embeddings.shape) == 4 and retain_embeddings.shape[1] == 1:
                    retain_embeddings = retain_embeddings.squeeze(1)
                
                # Generate noise and add to latents
                noise = torch.randn_like(retain_latents)
                noisy_latents = noise_scheduler.add_noise(retain_latents, noise, retain_timesteps)
                
                # Use original UNet to generate target prediction
                with torch.no_grad():
                    target_noise = original_unet(
                        noisy_latents,
                        retain_timesteps,
                        retain_embeddings,
                        return_dict=False
                    )[0]
                
                # Use model with LoRA to make prediction
                safety_model.update_lora_params(theta)
                pred_noise = safety_model.unet(
                    noisy_latents,
                    retain_timesteps,
                    retain_embeddings,
                    return_dict=False
                )[0]
                
                # Calculate retention loss (MSE)
                retain_loss = F.mse_loss(pred_noise.float(), target_noise.float(), reduction="mean")
            except Exception as e:
                print(f"Error calculating retain loss: {str(e)}")
                retain_loss = torch.tensor(0.0, device=args.device)
            
            # Calculate ESD loss or diffusion loss
            if args.loss_mode == "esd":
                safety_model.update_lora_params(theta)
                esd_loss_theta = compute_esd_loss(
                    safety_model.unet,
                    original_unet,
                    concept_embeddings,
                    empty_embeddings,
                    base_embeddings,
                    batch_latents,
                    t,
                    noise_scheduler,
                    args.negative_guidance
                )
                safety_model.update_lora_params(theta_star)
                esd_loss_theta_star = compute_esd_loss(
                    safety_model.unet,
                    original_unet,
                    concept_embeddings,
                    empty_embeddings,
                    base_embeddings,
                    batch_latents,
                    t,
                    noise_scheduler,
                    args.negative_guidance
                )

            elif args.loss_mode == "ga":
                safety_model.update_lora_params(theta)
                esd_loss_theta = -compute_diffusion_loss(
                    safety_model.unet,
                    outer_text_embeddings,  # Use prompt embeddings from outer batch
                    batch_latents,
                    t,
                    noise_scheduler
                )
                safety_model.update_lora_params(theta_star)
                esd_loss_theta_star = -compute_diffusion_loss(
                    safety_model.unet,
                    outer_text_embeddings,
                    batch_latents,
                    t,
                    noise_scheduler
                )
            
            # Calculate total loss using dynamic weights
            loss = esd_loss_theta + current_outer_lambda * (esd_loss_theta_star - esd_loss_theta) + current_retain_loss_w * retain_loss
            
            print(f"esd_loss_theta: {esd_loss_theta.item():.6f}")
            print(f"esd_loss_theta_star: {esd_loss_theta_star.item():.6f}")
            print(f"retain_loss: {retain_loss.item():.16f}")
            if args.dynamic_weights:
                print(f"Current outer_lambda: {current_outer_lambda:.4f}")
                print(f"Current retain_loss_weight: {current_retain_loss_w:.4f}")
            print(f"Final loss: {loss.item():.6f}")
            
            return loss
            
        except Exception as e:
            print(f"Outer loss calculation error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return torch.tensor(0.0, device=args.device, requires_grad=True)
        
    return create_inner_loss, outer_loss, update_epoch, set_current_loader

def main():
    args = parse_args()
    
    # Set CUDA memory allocator
    print("Initializing CUDA settings...")
    print(f"Using CUDA device: {args.cuda_device}")

    # Specify GPU device to use
    torch.cuda.set_device(args.cuda_device)
    
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    torch.backends.cudnn.benchmark = True
    
    # Create output directory
    output_path = create_output_dir(args)
    
    # Create eval directory
    eval_dir = Path(output_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        # Load model components
        print("Loading model components...")
        tokenizer, text_encoder, vae, unet = load_models(args.model_path, args.device, args.local_unet_path)
        
        # Disable gradient checkpointing
        unet.disable_gradient_checkpointing()
        print("✓ Model loading completed")
        
        # Create safety model (with LoRA)
        print("Initializing LoRA model...")
        safety_model = SafetyModel(unet, train_method=args.train_method, 
                                 lora_up_std=args.lora_up_std,
                                 lora_down_std=args.lora_down_std,
                                 rank=args.lora_rank).to(args.device)
        print("✓ LoRA initialization completed")

        # Create noise scheduler
        print("\nCreating noise scheduler...")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        print("✓ Noise scheduler creation completed")
 
        # Create outer data loader
        print("\nCreating data loaders...")
        print("1. Creating outer data loader")
        outer_train_loader = load_nsfw_dataset(
            args.outer_data_dir, tokenizer, text_encoder, vae,
            device=args.device, batch_size=args.outer_batch_size,
            use_metadata_prompts=True
        )
        
        # Create inner data loader dictionary
        print("\n2. Creating inner data loaders")
        inner_train_loaders = {}
        for config in args.inner_loop_configs:
            if isinstance(config, dict):
                # If it's a dictionary, convert to InnerLoopConfig object
                config = InnerLoopConfig(**config)
            print(f"\nLoading dataset: {config.data_dir}")
            print(f"Configuration info: {vars(config)}")
            inner_train_loaders[config.name] = load_nsfw_dataset(
                config.data_dir, tokenizer, text_encoder, vae,
                device=args.device, batch_size=config.batch_size,
                use_metadata_prompts=True
            )
        print("\n✓ All data loaders created")
        
        # Get text embeddings
        print("\nCreating text embeddings...")
        print("1. Creating empty text embeddings")
        empty_embeddings = get_text_embeddings(tokenizer, text_encoder, "", args.device)
        print("✓ Obtained embeddings for empty text")
        
        print("\n2. Creating concept embeddings")
        concept_embeddings = get_text_embeddings(tokenizer, text_encoder, args.erase_concept, args.device)
        print(f"✓ Obtained embeddings for concept to remove '{args.erase_concept}'")

        print("\n3. Creating base concept embeddings")
        base_embeddings = get_text_embeddings(tokenizer, text_encoder, args.erase_from, args.device)
        print(f"✓ Obtained embeddings for base concept to erase from '{args.erase_from}'")

        # Create loss functions
        print("\nCreating loss functions...")
        create_inner_loss, outer_loss_fn, update_epoch, set_current_loader = create_loss_functions(
            safety_model,
            outer_train_loader,  # Use outer data loader
            noise_scheduler,
            concept_embeddings,
            base_embeddings,
            empty_embeddings,
            tokenizer,
            text_encoder,
            vae,
            args,
            prior_loaders=None
        )
        
            # outer_train_loader,  # Use outer data loader
        
        # Create inner optimizers
        print("\nCreating inner optimizers...")
        inner_optimizers = []
        prior_loaders = {}  # For dreambooth prior preservation loaders
        
        for config in args.inner_loop_configs:
            if isinstance(config, dict):
                config = InnerLoopConfig(**config)
            
            print(f"Creating {config.optimizer_type} optimizer for config '{config.name}'")
            
            # If dreambooth mode, create prior preservation data loader
            if config.finetune_method == "dreambooth":
                print(f"Creating prior preservation data loader: {config.prior_preservation_data_dir}")
                prior_loaders[config.name] = load_nsfw_dataset(
                    config.prior_preservation_data_dir,
                    tokenizer,
                    text_encoder,
                    vae,
                    device=args.device,
                    batch_size=config.batch_size,
                    use_metadata_prompts=True
                )
            
            # Create a dedicated inner loss for this config
            inner_loss_fn = create_inner_loss(config, prior_loaders)
            optimizer = create_inner_optimizer(config, inner_loss_fn)
            optimizer.config = config
            inner_optimizers.append(optimizer)
        
        print("✓ All inner optimizers created")
        
        # Prepare trainable parameters (LoRA)
        print("\nPreparing optimizer parameters...")
        trainable_params = safety_model.get_lora_params()
        if not trainable_params:
            raise ValueError("No trainable LoRA parameters found! Please check the model configuration.")
        
        print(f"- Number of trainable parameters: {len(trainable_params)}")
        
        # Clone params and hparams to device with gradients enabled
        params = []
        hparams = []
        for p in trainable_params:
            param = p.clone().detach().requires_grad_(True).to(args.device)
            hparam = p.clone().detach().requires_grad_(True).to(args.device)
            params.append(param)
            hparams.append(hparam)
        
        print(f"- Params device: {params[0].device}")
        print(f"- Hparams device: {hparams[0].device}")
        
        # Create outer optimizer (for hparams) and scheduler
        print("\nCreating outer optimizer...")
        outer_optimizer = torch.optim.AdamW(
            hparams,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        outer_scheduler = get_scheduler(
            "constant",
            optimizer=outer_optimizer,
            num_warmup_steps=0,
            num_training_steps=args.num_epochs
        )
        
        # Create bilevel optimizer
        bilevel_optimizer = BiLevelOptimizer(
            params=params,
            hparams=hparams,
            inner_optimizers=inner_optimizers,
            outer_optimizer=outer_optimizer,
            inner_train_loaders=inner_train_loaders,
            set_current_loader=set_current_loader,
            K=args.K,
            device=args.device,
            max_grad_norm=args.max_grad_norm,
            meta_rate=args.meta_rate,
            prior_loaders=prior_loaders,
            inner_sample_k=args.inner_sample_k,
            safety_model=safety_model,
            noise_scheduler=noise_scheduler
        )
        print("✓ Bi-level optimizer created")
        
        # Tracking
        outer_losses = []
        grad_stats_history = []
        grad_stats_path = os.path.join(output_path, "grad_stats.json")
        
        # Training loop
        with tqdm(range(args.num_epochs), desc="Training") as pbar:
            for epoch in pbar:
                try:
                    # Update epoch counter for dynamic weights
                    update_epoch()
                    
                    # Simple periodic inference
                    if (epoch + 1) % args.inference_interval == 0:
                        print(f"\nRunning simple inference at epoch {epoch}...")
                        simple_eval_dir = Path(output_path, "simple_eval")
                        os.makedirs(simple_eval_dir, exist_ok=True)
                        
                        image1, image2, image3 = inference_with_lora(
                            unet=safety_model.unet,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vae=vae,
                            prompt2=args.val_prompt2,
                            prompt4=args.val_prompt4,
                            hparams=hparams,
                            device=args.device
                        )
                        
                        if image1 is not None:
                            image1_path = Path(simple_eval_dir, f"eval_{epoch}_1.png")
                            image2_path = Path(simple_eval_dir, f"eval_{epoch}_2.png")
                            image3_path = Path(simple_eval_dir, f"eval_{epoch}_3.png")
                            image1.save(image1_path)
                            image2.save(image2_path)
                            image3.save(image3_path)
                            print(f"✓ Saved eval images to: {image1_path}, {image2_path}, {image3_path}")
                        else:
                            print("Warning: Image generation failed")
                    
                    # One optimization step
                    process = epoch / args.num_epochs
                    outer_loss, epoch_grad_stats = bilevel_optimizer.step(inner_loss_fn, outer_loss_fn, process)
                    
                    # Record loss and grad stats
                    outer_losses.append(outer_loss)
                    epoch_grad_stats['epoch'] = epoch
                    
                    # Ensure boolean flags are proper bool
                    if 'grad_update_skipped' in epoch_grad_stats:
                        epoch_grad_stats['grad_update_skipped'] = bool(epoch_grad_stats['grad_update_skipped'])
                    if 'grad_clipped' in epoch_grad_stats:
                        epoch_grad_stats['grad_clipped'] = bool(epoch_grad_stats['grad_clipped'])
                    
                    grad_stats_history.append(epoch_grad_stats)
                    
                    # Write grad stats to file in real-time
                    try:
                        if os.path.exists(grad_stats_path):
                            with open(grad_stats_path, 'r') as f:
                                existing_stats = json.load(f)
                        else:
                            existing_stats = []
                        existing_stats.append(epoch_grad_stats)
                        with open(grad_stats_path, 'w') as f:
                            json.dump(existing_stats, f, indent=4)
                    except Exception as e:
                        print(f"\nWarning: Error writing grad stats: {str(e)}")
                    
                    # Update progress bar
                    status = {'outer_loss': f'{outer_loss:.8f}'}
                    if epoch_grad_stats.get('grad_update_skipped', False):
                        status['grad_status'] = 'skipped'
                    pbar.set_postfix(status)
                
                except Exception as e:
                    print(f"\nTraining error: {str(e)}")
                    raise e
        
        print("\nTraining finished!")
        
        # Save final model using last hparams
        print("\nSaving final model with merged LoRA weights...")
        try:
            safety_model.update_lora_params(hparams)
            _ = safety_model.merge_and_save_lora_weights()
            save_path = os.path.join(output_path, "final_model.pt")
            torch.save(safety_model.unet.state_dict(), save_path)
        except Exception as e:
            print(f"Error while saving model: {str(e)}")
            raise e
        
        print("\n✓ All done!")
    
    finally:
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 