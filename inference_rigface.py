
import argparse
import os

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL
from diffusers import (
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from rigface.models.pipelineRigFace import RigFacePipeline as RigFacePipelineInference
from rigface.models.unet_ID_2d_condition import UNetID2DConditionModel
from rigface.models.unet_denoising_2d_condition import UNetDenoise2DConditionModel



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stable-diffusion-v1-5/stable-diffusion-v1-5',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument("--seed", type=int, default=424, help="A seed for reproducible training.")


    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
    )


    parser.add_argument(
        "--vit_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )

    parser.add_argument(
        "--vton_unet_path",
        type=str,
        default='./pre_trained/unet_denoise/checkpoint-70000',
    )
    
    parser.add_argument(
        "--garm_unet_path",
        type=str,
        default='./pre_trained/unet_id/checkpoint-70000',
    )
    parser.add_argument(
        "--id_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--bg_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--render_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='',
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args



def make_data(args):

    transform = transforms.ToTensor()

    img_name = args.id_path
    bg_name = args.bg_path
    render_name = args.render_path

    source = Image.open(img_name)
    source = transform(source)

    bg = Image.open(bg_name)
    bg = transform(bg)

    render = Image.open(render_name)
    # render = render.resize((512, 512))
    render = transform(render)

    return source, bg, render


def tokenize_captions(tokenizer, captions, max_length):

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def main(args):

    device = 'cuda'
    vton_unet_path = args.vton_unet_path
    garm_unet_path = args.garm_unet_path

    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae"
        ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
        ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

    unet_id = UNetID2DConditionModel.from_pretrained(
            garm_unet_path,
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )

    unet_denoising = UNetDenoise2DConditionModel.from_pretrained(
            vton_unet_path,
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )

    unet_denoising.requires_grad_(False)
    unet_id.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32


    pipeline = RigFacePipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet_id=unet_id,
        unet_denoising=unet_denoising,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(device)


    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    source, bg, rend = make_data(args)
    prompt = 'A close up of a person.'
    source = source.unsqueeze(0)
    bg = bg.unsqueeze(0)
    rend = rend.unsqueeze(0)

    prompt_embeds = text_encoder(tokenize_captions(tokenizer, [prompt], 2).to(device))[0]


    exp = np.load(args.exp_path)

    os.makedirs(args.save_path, exist_ok=True)
    tor_exp = torch.from_numpy(exp).unsqueeze(0)

    samples = pipeline(
        prompt_embeds=prompt_embeds,
        source=source,
        bg=bg,
        render=rend,
        exp=tor_exp,
        num_inference_steps=args.inference_steps,
        generator=generator,
    ).images[0]
    samples.save(os.path.join(args.save_path, f'out.png'))

if __name__ == "__main__":
    args = parse_args()

    main(args)
