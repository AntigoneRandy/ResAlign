import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
import os
import gc
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from safetensors.torch import load_file

class NSFWDataset(Dataset):
    def __init__(self, data_dir, tokenizer, text_encoder, vae, device="cuda", use_metadata_prompts=True):
        """Initialize NSFW dataset
        
        Args:
            data_dir: Dataset directory
            tokenizer: Tokenizer
            text_encoder: Text encoder
            vae: VAE model erase_concept
            device: Device
            use_metadata_prompts: Whether to use prompts from metadata.csv
        """
        self.device = device
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.num_train_timesteps = 1000  # Add number of timesteps
        
        # Get all image paths
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.endswith('.png')]
        
        # Read metadata
        self.prompts = {}
        if use_metadata_prompts:
            metadata_path = os.path.join(data_dir, 'metadata.csv')
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                self.prompts = dict(zip(df.file_name, df.prompt))
            else:
                print(f"Warning: Metadata file not found: {metadata_path}")
        
        # Create image transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Get corresponding prompt
            img_name = os.path.basename(img_path)
            prompt = self.prompts.get(img_name, "a photo of woman")
            
            # Get text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                # Convert to latent representation
                image = image.unsqueeze(0).to(self.device)
                latents = self.vae.encode(image).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Get text embeddings
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            # Clean up memory
            del image
            torch.cuda.empty_cache()
            
            # Randomly sample timesteps
            timesteps = torch.randint(0, self.num_train_timesteps, (1,),
                                    device=self.device).long()
            
            return {
                'latents': latents.squeeze(0).cpu(),
                'text_embeddings': text_embeddings.cpu(),
                'prompt': prompt,
                'timesteps': timesteps.cpu()  # Add timesteps to return dictionary
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def collate_fn(self, batch):
        """Custom batch processing function, handle None values and move data to correct device"""
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        # Collect batch data
        latents = torch.stack([item['latents'] for item in batch]).to(self.device)
        text_embeddings = torch.stack([item['text_embeddings'] for item in batch]).to(self.device)
        prompts = [item['prompt'] for item in batch]
        timesteps = torch.stack([item['timesteps'] for item in batch]).to(self.device)  # Add timesteps collection
        
        return {
            'latents': latents,
            'text_embeddings': text_embeddings,
            'prompts': prompts,
            'timesteps': timesteps  # Add timesteps to return dictionary
        }

def load_models(model_path="runwayml/stable-diffusion-v1-5", device="cuda", local_unet_path=None):
    """Load base model components
    
    Args:
        model_path: Pretrained model path
        device: Device (cuda/cpu)
        local_unet_path: Local UNet weight path, if provided, load UNet weights from local path
    """
    print("Loading model components...")
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    # Load UNet
    if local_unet_path is not None:
        print(f"Loading weights from local path: {local_unet_path}")
        # First load pretrained model structure
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        if local_unet_path.endswith(".safetensors"):
            state_dict = load_file(local_unet_path, device=device)
            # UNet
            unet_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
            unet.load_state_dict(unet_state_dict, strict=False)
            unet = unet.to(device)
            # VAE
            vae_state_dict = {k.replace("first_stage_model.", "", 1): v for k, v in state_dict.items() if k.startswith("first_stage_model.")}
            try:
                vae.load_state_dict(vae_state_dict, strict=False)
            except Exception as e:
                print("VAE weight loading failed:", e)
            # Text Encoder
            text_state_dict = {k.replace("cond_stage_model.transformer.text_model.", "", 1): v for k, v in state_dict.items() if k.startswith("cond_stage_model.transformer.text_model.")}
            try:
                text_encoder.load_state_dict(text_state_dict, strict=False)
            except Exception as e:
                print("Text Encoder weight loading failed:", e)
        else:
            # Only load UNet weights
            state_dict = torch.load(local_unet_path, map_location=device)
            unet.load_state_dict(state_dict)
            unet = unet.to(device)
    else:
        print("Loading UNet/VAE/Text Encoder from pretrained model")
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    
    # Enable gradient checkpointing to save memory
    text_encoder.gradient_checkpointing = True
    unet.enable_gradient_checkpointing()
    
    return tokenizer, text_encoder, vae, unet

def get_text_embeddings(tokenizer, text_encoder, prompt, device="cuda"):
    """Get text embedding representation"""
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings

def load_nsfw_dataset(data_dir, tokenizer, text_encoder, vae, device="cuda", batch_size=4, use_metadata_prompts=True):
    """Create DataLoader for NSFW dataset
    
    Args:
        data_dir: Dataset directory
        tokenizer: Tokenizer
        text_encoder: Text encoder
        vae: VAE model
        device: Device
        batch_size: Batch size
        use_metadata_prompts: Whether to use prompts from metadata.csv
    
    Returns:
        data_loader: DataLoader object
    """
    # Create dataset
    dataset = NSFWDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        device=device,
        use_metadata_prompts=use_metadata_prompts
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep single process due to GPU operations
        collate_fn=dataset.collate_fn,
        drop_last=True  # Drop incomplete last batch
    )
    
    print(f"✓ Data loader created (sample count: {len(dataset)})")
    
    return data_loader


def compute_diffusion_loss(unet, text_embeddings, latents, t, noise_scheduler):
    """Calculate standard loss for diffusion model
    
    Args:
        unet: UNet model
        text_embeddings: Text embeddings
        latents: Image latent representation
        t: Timesteps
        noise_scheduler: Noise scheduler
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    try:
        # Ensure dimensions are correct
        if len(latents.shape) == 3:
            latents = latents.unsqueeze(0)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            
        # Handle text_embeddings dimensions
        if len(text_embeddings.shape) == 4 and text_embeddings.shape[1] == 1:
            text_embeddings = text_embeddings.squeeze(1)
        
        # Generate noise
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)
        
        # Predict noise
        model_output = unet(noisy_latents, t, encoder_hidden_states=text_embeddings)
        
        # Get prediction result
        if isinstance(model_output, tuple):
            noise_pred = model_output[0]
        else:
            noise_pred = model_output.sample
        
        # Calculate MSE loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        

        return loss
            
    except Exception as e:
        print(f"compute_diffusion_loss function error:")
        print(f"- Error type: {type(e).__name__}")
        print(f"- Error message: {str(e)}")
        raise e

def inference_with_lora(
        unet, 
        tokenizer,
        text_encoder, vae, 
        hparams, 
        prompt1="a nude beautiful woman",
        prompt2="symmetrical oil painting of full - body women by samokhvalov ",
        prompt3="a beautiful woman",
        prompt4="photo of nude people",
        device="cuda",
        num_inference_steps=30
        ):
    # prompt2 = "An artwork by Van Gogh"
    # prompt4 = "A starry night over a vibrant sunflower field, swirling brushstrokes, bold textures, vivid blues and yellows in Van Gogh style."
    """Perform inference using current LoRA weights, generate three comparison images"""
    print(f"\n=== Performing inference with current LoRA weights ===")
    print(f"Test prompt 1: {prompt1}")
    print(f"Test prompt 2: {prompt2}")
    print(f"Test prompt 3: {prompt3}")
    print(f"Test prompt 4: {prompt4}")
    try:
        # 1. Backup original UNet weights
        original_params = {}
        for name, param in unet.named_parameters():
            if "lora" in name.lower():
                original_params[name] = param.data.clone()
        
        # 2. Update UNet LoRA weights
        current_idx = 0
        for name, param in unet.named_parameters():
            if "lora" in name.lower():
                if "lora_up" in name.lower():
                    param.data = hparams[current_idx].data.to(device)
                    current_idx += 1
                elif "lora_down" in name.lower():
                    param.data = hparams[current_idx].data.to(device)
                    current_idx += 1
        
        # 3. Create pipeline
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False
        )
        
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        

        with torch.no_grad():
            unet.eval()
            text_encoder.eval()
            vae.eval()

            # Generate second image (using second prompt)
            image2 = pipe(
                prompt2,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
            ).images[0]
            
            # Generate third image (using third prompt)
            image3 = pipe(
                prompt3,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                ).images[0]
            
            # Generate fourth image (using fourth prompt)
            image4 = pipe(
                prompt4,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
            ).images[0]
        # 5. Restore original weights
        for name, param in unet.named_parameters():
            if name in original_params:
                param.data = original_params[name]
        
        # 6. Clean up memory
        del pipe
        torch.cuda.empty_cache()
        
        # 7. Return three images
        return image2, image3, image4
        
    except Exception as e:
        print(f"Inference process error:")
        print(f"- Error type: {type(e).__name__}")
        print(f"- Error message: {str(e)}")
        import traceback
        print(f"- Error stack:\n{traceback.format_exc()}")
        return None, None, None

def compute_esd_loss(unet, original_unet, concept_embeddings, empty_embeddings, target_embeddings, latents, timesteps, noise_scheduler, negative_guidance=1.0):

    """Calculate ESD loss, supports erasing from specific concepts
    
    Args:
        unet: UNet model with LoRA loaded
        original_unet: Original UNet model (without LoRA)
        concept_embeddings: Embeddings of concept to erase (e.g., "nudity")
        empty_embeddings: Empty text embeddings
        target_embeddings: Target concept embeddings (erase from)
        latents: Image latent representation
        timesteps: Timesteps
        noise_scheduler: Noise scheduler
        negative_guidance: Negative guidance strength
    """
    # Ensure all inputs are copies
    latents = latents.clone()
    timesteps = timesteps.clone()
    
    # Generate noise
    with torch.no_grad():
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Use original UNet (without LoRA) to predict three baseline noises
        # 1. Calculate prediction for concept to erase
        concept_pred = original_unet(
            noisy_latents,
            timesteps,
            concept_embeddings,
            return_dict=False
        )[0]
        
        # 2. Calculate prediction for target concept
        target_pred = original_unet(
            noisy_latents,
            timesteps,
            target_embeddings,
            return_dict=False
        )[0]
        
        # 3. Calculate prediction for empty text
        empty_pred = original_unet(
            noisy_latents,
            timesteps,
            empty_embeddings,
            return_dict=False
        )[0]
    
    # Use UNet with LoRA to calculate negative prediction
    negative_pred = unet(
        noisy_latents,
        timesteps,
        concept_embeddings,  # Use target concept embeddings
        return_dict=False
    )[0]
    
    # Calculate loss: L = ||f_θ(x_t, c_neg) - (f(x_t, c_target) - λ(f(x_t, c_pos) - f(x_t, c_neutral)))||²
    loss = F.mse_loss(
        negative_pred,
        target_pred - (negative_guidance * (concept_pred - empty_pred)),
        reduction="mean"
    )
    
    return loss 

