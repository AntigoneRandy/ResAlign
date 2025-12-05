import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearning", type=str, default=None, help="Unlearning method (optional if --model_dir is provided)")
    parser.add_argument("--model_dir", type=str, default=None, help="Model weight directory (directory under outputs, e.g., outputs/160epochs-20251105-1700)")
    parser.add_argument("--dataset_dir", type=str, default="data/new_mscoco10k", help="Evaluation dataset directory")
    parser.add_argument("--result_root", type=str, default=None, help="Result root directory (defaults to model_dir)")
    parser.add_argument("--num_per_prompt", type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion sampling steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Parallel batch size")
    args = parser.parse_args()
    
    # If model_dir is provided, automatically set result directory
    if args.model_dir:
        if args.result_root is None:
            args.result_root = args.model_dir
        if args.unlearning is None:
            # Use directory name as unlearning identifier
            args.unlearning = os.path.basename(args.model_dir)

    # Must be set before importing torch, etc.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = "cuda:0"

    import pandas as pd
    from tqdm import tqdm
    import torch
    from diffusers import DiffusionPipeline, UNet2DConditionModel
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from T2IBenchmark import calculate_fid
    from safetensors.torch import load_file
    import torch.nn as nn
    from urllib.request import urlretrieve
    from os.path import expanduser
    import clip
    import shutil

    def get_aesthetic_model(clip_model="vit_l_14"):
        """load the aesthetic model"""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + f"/sa_0_4_{clip_model}_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model, map_location="cpu")
        m.load_state_dict(s)
        m.eval()
        return m

    def compute_aesthetic_score(gen_dir, num_per_prompt=1):
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
        aesthetic_model = get_aesthetic_model("vit_l_14").to(device)
        scores = []
        img_list = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png')])
        for img_name in tqdm(img_list, desc="Aesthetic Evaluation"):
            img_path = os.path.join(gen_dir, img_name)
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                score = aesthetic_model(image_features.float()).item()
                scores.append(score)
        mean_score = sum(scores) / len(scores)
        return mean_score

    def load_pipe_with_unlearning(args):
        # If model_dir is provided, prioritize weights from that directory
        if args.model_dir:
            model_path = os.path.join(args.model_dir, "final_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weight file not found: {model_path}")
            
            print(f"Loading weights from local directory: {model_path}")
            pipe = DiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            unet = pipe.unet
            unet.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Weights loaded successfully: {model_path}")
            return pipe
        
        if args.unlearning == "safegen":
            model_id = "LetterJohn/SafeGen-Pretrained-Weights"
            print(f"Loading safegen weights: {model_id}")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            return pipe
        if args.unlearning == "sd14":
            model_id = "CompVis/stable-diffusion-v1-4"
            print(f"Loading original SD1.4 weights: {model_id}")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            return pipe
        if args.unlearning == "advunlearn":
            print("Loading advunlearn weights: CompVis/stable-diffusion-v1-4 + AdvUnlearn_Nudity_UNet.pt + AdvUnlearn_Nudity_text_encoder_full.pt")
            pipe = DiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            # Load text_encoder parameters
            def extract_text_encoder_ckpt(ckpt_path):
                full_ckpt = torch.load(ckpt_path, map_location=device)
                new_ckpt = {}
                for key in full_ckpt.keys():
                    if 'text_encoder.text_model' in key:
                        new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
                return new_ckpt
            text_encoder_path = "/ephemeral/unlearning/advunlearn/AdvUnlearn_Nudity_text_encoder_full.pt"
            text_encoder_state = extract_text_encoder_ckpt(text_encoder_path)
            pipe.text_encoder.load_state_dict(text_encoder_state, strict=False)
            print("advunlearn weights loaded successfully")
            return pipe
        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        unet = pipe.unet
        try:
            if args.unlearning == "esd":
                model_path = "/ephemeral/unlearning/esd/diffusers-nudity-ESDu1-UNET.pt"
                unet.load_state_dict(torch.load(model_path, map_location=device))
            else:
                if args.unlearning is None:
                    raise ValueError("Must provide --unlearning parameter or --model_dir parameter")
                raise ValueError(f"Unknown unlearning type: {args.unlearning}")
            if args.unlearning:
                print(f"Weights {args.unlearning} loaded successfully: {model_path}")
        except Exception as e:
            print(f"Weight loading failed: {e}")
            raise
        return pipe

    def generate_images(pipe, prompts_df, save_dir, num_per_prompt=1, seed_base=42, num_inference_steps=30, batch_size=8):
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for idx, row in prompts_df.iterrows():
            case_number = row['case_number']
            prompt = row['prompt']
            for j in range(num_per_prompt):
                img_name = f"{case_number}_{j}.png"
                img_path = os.path.join(save_dir, img_name)
                if os.path.exists(img_path):
                    continue
                seed = seed_base + int(case_number) * num_per_prompt + j
                tasks.append({
                    "prompt": prompt,
                    "seed": seed,
                    "save_path": img_path
                })

        if len(tasks) == 0:
            return

        for start in tqdm(range(0, len(tasks), batch_size), desc="Parallel Generation Batches"):
            batch = tasks[start:start + batch_size]
            prompts = [t["prompt"] for t in batch]
            generators = [torch.Generator(device=device).manual_seed(t["seed"]) for t in batch]

            images = pipe(
                prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generators,
                height=512,
                width=512
            ).images

            for img, t in zip(images, batch):
                img.save(t["save_path"])

    def _has_images(img_dir):
        if not os.path.isdir(img_dir):
            return False
        for n in os.listdir(img_dir):
            l = n.lower()
            if l.endswith('.png') or l.endswith('.jpg') or l.endswith('.jpeg'):
                return True
        return False

    def compute_fid(gen_dir, ref_dir):
        if not _has_images(gen_dir):
            print(f"Skipping FID: No images in generation directory: {gen_dir}")
            return None
        if not _has_images(ref_dir):
            print(f"Skipping FID: No images in reference directory: {ref_dir}")
            return None
        fid, _ = calculate_fid(gen_dir, ref_dir)
        return fid

    def compute_clip_score(gen_dir, prompts_df, num_per_prompt=1):
        # Use OpenAI CLIP to directly calculate image-text similarity, avoiding transformers requirement for torch>=2.6
        model, preprocess = clip.load("ViT-B/32", device=device)
        clip_scores = []
        for idx, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="CLIP Evaluation"):
            case_number = row['case_number']
            prompt = row['prompt']
            for j in range(num_per_prompt):
                img_path = os.path.join(gen_dir, f"{case_number}_{j}.png")
                if not os.path.exists(img_path):
                    continue
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                text = clip.tokenize([prompt]).to(device)
                with torch.no_grad():
                    logits_per_image, _ = model(image, text)
                clip_scores.append(logits_per_image.item())
        mean_clip = sum(clip_scores) / len(clip_scores) if len(clip_scores) > 0 else 0.0
        return mean_clip

    # Determine result directory
    if args.result_root:
        if args.model_dir and args.result_root == args.model_dir:
            # If using model_dir, save results in eval subdirectory
            result_dir = os.path.join(args.result_root, "eval")
        else:
            result_dir = os.path.join(args.result_root, f"{args.unlearning}_coco")
    else:
        result_dir = os.path.join("results", f"{args.unlearning}_coco")
    
    gen_img_dir = os.path.join(result_dir, "gen_images")
    os.makedirs(gen_img_dir, exist_ok=True)

    prompts_csv = os.path.join(args.dataset_dir, "prompts.csv")
    prompts_df = pd.read_csv(prompts_csv)
    prompts_df['case_number'] = prompts_df['case_number'].astype(str)

    print("Loading model and generating images...")
    pipe = load_pipe_with_unlearning(args)
    generate_images(
        pipe,
        prompts_df,
        gen_img_dir,
        num_per_prompt=args.num_per_prompt,
        num_inference_steps=args.num_inference_steps,
        batch_size=args.batch_size,
    )

    print("Computing FID score...")
    ref_img_dir = os.path.join(args.dataset_dir, "images")
    fid_score = compute_fid(gen_img_dir, ref_img_dir)

    print("Computing CLIP score...")
    clip_score = compute_clip_score(gen_img_dir, prompts_df, num_per_prompt=args.num_per_prompt)

    print("Computing Aesthetic score...")
    aesthetic_score = compute_aesthetic_score(gen_img_dir, num_per_prompt=args.num_per_prompt)

    with open(os.path.join(result_dir, "result.txt"), "a") as f:
        if fid_score is None:
            f.write("FID: skipped (no images)\n")
        else:
            f.write(f"FID: {fid_score}\n")
        f.write(f"CLIP: {clip_score}\n")
        f.write(f"Aesthetic: {aesthetic_score}\n")
    print(f"Evaluation complete, results saved to {result_dir}/result.txt")

if __name__ == "__main__":
    main()

# python first_evl.py --unlearning 1028run1 --cuda_device 0
# python first_evl.py --unlearning 1028run2 --cuda_device 1
# python first_evl.py --unlearning 1028run3 --cuda_device 2
# python first_evl.py --unlearning safegen --cuda_device 1
# python first_evl.py --unlearning advunlearn --cuda_device 3
# python first_evl.py --unlearning esd --cuda_device 3
# python first_evl.py --unlearning sd14 --cuda_device 0

# python first_evl.py --unlearning lt --cuda_device 2
# python first_evl.py --unlearning ng --cuda_device 0

# python first_evl.py --unlearning retain --cuda_device 0
# python first_evl.py --unlearning inner_5 --cuda_device 1
# python first_evl.py --unlearning inner_10 --cuda_device 2
# python first_evl.py --unlearning inner_20 --cuda_device 3
# python first_evl.py --unlearning inner_40 --cuda_device 2
# python first_evl.py --unlearning inner_50 --cuda_device 3
# python first_evl.py --unlearning ex_ours --cuda_device 3

# python first_evl.py --unlearning inner_1_new --cuda_device 0
# python first_evl.py --unlearning rece --cuda_device 0