import argparse
import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(pretrained_model="CompVis/stable-diffusion-v1-4", unet_weights=None, text_encoder_weights=None):
    """
    Load Stable Diffusion pipeline with optional custom weights for UNet and Text Encoder
    
    Args:
        pretrained_model: Base model path
        unet_weights: Path to UNet weights file (.pt or .safetensors)
        text_encoder_weights: Path to text encoder weights file (.pt or .safetensors)
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # Load custom UNet weights if provided
    if unet_weights:

        if 'receler' in unet_weights:
            import sys
            sys.path.append('/data/home/Boheng/sanowip/harmful_ft')
            from receler.erasers.diffusers_erasers import inject_eraser
            inject_eraser(pipe.unet, torch.load(unet_weights.replace('.pt', '_eraser.pth'), map_location='cuda'))
            pipe.unet = pipe.unet.to(pipe.device)

        print(f"Loading UNet weights from: {unet_weights}")
        if unet_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(unet_weights)
        else:
            state_dict = torch.load(unet_weights, map_location='cuda')
        pipe.unet.load_state_dict(state_dict, strict=False)
        print("UNet weights loaded successfully")

    # Load custom Text Encoder weights if provided
    if text_encoder_weights:
        print(f"Loading Text Encoder weights from: {text_encoder_weights}")
        if text_encoder_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(text_encoder_weights)
        else:
            state_dict = torch.load(text_encoder_weights, map_location='cuda')
        pipe.text_encoder.load_state_dict(state_dict, strict=False)
        print("Text Encoder weights loaded successfully")

    return pipe


def generate_images(args):
    """Generate images using the loaded pipeline"""
    pipe = load_model(
        pretrained_model=args.pretrained_model,
        unet_weights=args.unet_weight, 
        text_encoder_weights=args.text_encoder_weight
    )

    # Load prompts from CSV file
    prompts = pd.read_csv(args.prompt_dir)['prompt'].tolist()
    os.makedirs(args.image_dir, exist_ok=True)

    print(f"Generating {len(prompts) * args.num} images...")
    print(f"Output directory: {args.image_dir}")

    # Process in batches for better memory management
    batch_size = 64
    total_prompts = len(prompts)
    
    for batch_start in range(0, total_prompts, batch_size):
        # Get current batch of prompts
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_index = batch_start // batch_size + 1
        print(f"Processing batch {batch_index}, prompts {batch_start} to {batch_start + len(batch_prompts) - 1}")

        for j in range(args.num):  
            torch.cuda.empty_cache()
            
            # Generate images for the current batch
            images = pipe(
                batch_prompts,    
                height=512,
                width=512, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            ).images
            
            # Save images with proper naming
            for i in range(len(images)):
                prompt_idx = batch_start + i + 1  # 1-indexed
                file_name = f'{prompt_idx}-{j}.png'
                file_path = os.path.join(args.image_dir, file_name)
                images[i].save(file_path)
                
        print(f"Batch {batch_index} completed")
            
    torch.cuda.empty_cache()
    print(f"Generated {len(prompts)*args.num} images to {args.image_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned Stable Diffusion")
    parser.add_argument("--prompt_dir", required=True, help="Path to CSV file containing prompts")
    parser.add_argument("--image_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--pretrained_model", default="CompVis/stable-diffusion-v1-4", help="Base model path")
    parser.add_argument("--unet_weight", default=None, help="Path to fine-tuned UNet weights")
    parser.add_argument("--text_encoder_weight", default=None, help="Path to fine-tuned text encoder weights")
    parser.add_argument("--num", type=int, default=3, help="Number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("Generation Configuration:")
    print(f"  Prompt file: {args.prompt_dir}")
    print(f"  Output directory: {args.image_dir}")
    print(f"  Base model: {args.pretrained_model}")
    print(f"  UNet weights: {args.unet_weight or 'Using pretrained'}")
    print(f"  Text encoder weights: {args.text_encoder_weight or 'Using pretrained'}")
    print(f"  Images per prompt: {args.num}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print("=" * 50)

    set_seed(args.seed)
    generate_images(args)


if __name__ == "__main__":
    main()