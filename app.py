import gradio as gr
from utils.utils_net import prep_unet, AttentionStore, register_attention_control, get_token_cross_attention
import time
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from lavis.models import load_model_and_preprocess, load_model
from main import MfMOEPipeline

from utils.nti import NullInversion
from utils.ptp_utils import *
from utils.utils_mask import show_all_attention_maps, postprocess_mask
from utils.utils_app import global_context

logging.set_verbosity_error()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float16)
    
def generate_prompt(source_image):
    torch.set_default_dtype(torch.float32)
    device = global_context["device"]

    # Initialize BLIP captioner
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # img = Image.open(opt.image_path).resize((512,512), Image.Resampling.LANCZOS)
    # generate the caption
    _image = vis_processors["eval"](Image.fromarray(source_image)).unsqueeze(0).to(device)
    prompt_str = model_blip.generate({"image": _image})[0]
    print("Generated source prompt: ", prompt_str)
    del vis_processors
    del model_blip
    return prompt_str

def mfmoe_edit(source_image, source_prompt, model_version, steps, bootstrapping, token_position,
               ca_coef, seg_coef, appended_prompt, negative_prompt, num_masks, seed):
    # Seed
    seed_everything(seed)
    sample_count = len(os.listdir(global_context['out_dir']))
    out_dir = os.path.join(global_context['out_dir'], f'exp_{sample_count}')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Source prompt: ", source_prompt)
    if source_prompt is not None:
        prompt_str =  source_prompt

    
    sd = global_context["pipeline"]
    nti = global_context["inversion"]
    device = global_context["device"]
    sd.scheduler.num_inference_steps = steps
    
    # Null-text inversion
    (image_gt, image_enc), x_t, uncond_embeddings = nti.invert(source_image, prompt_str, offsets=(0, 0, 0, 0), verbose=True)
    print("Latent shape: ", x_t.shape)    
    
    del nti
    masks = None

    prompts = [prompt_str]
    neg_prompts = [prompt_str]
    
    rec_img, noise_loss_list, fg_masks = sd.reconstruct(masks, prompts, neg_prompts, 
                                                        512, 512, steps, 
                                                        bootstrapping=bootstrapping, 
                                                        latent=x_t, 
                                                        latent_path=None, 
                                                        latent_list_path=None, 
                                                        num_fgmasks=num_masks+1,
                                                        token_positions=[token_position],
                                                        out_dir='./results')
    # rec_img.save(os.path.join(out_dir, rec_path))
    
    # Process background mask
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    prompts = [prompt_str] + [appended_prompt]
    neg_prompts = [negative_prompt] + [negative_prompt]
    
    print("Prompts: ", prompts)
    print("Negative prompts: ", neg_prompts)
    start_gen = time.time()

    img = sd.generate(masks, prompts, neg_prompts, 
                      512, 512, steps, 
                      bootstrapping=bootstrapping, 
                      ca_coef=ca_coef, 
                      seg_coef=seg_coef, 
                      noise_loss_list=noise_loss_list, 
                      latent=x_t, 
                      latent_path=None, 
                      latent_list_path=None)

    end = time.time()
        
    # print(f"Total inference time: {end - start} seconds")
    print(f"Editing time: {end - start_gen} seconds")
    return [
        source_image,
        rec_img,
        img
    ]
    

def mfmoe_launcher():
    with gr.Blocks() as app:
        gr.Markdown("Mask-Free Multi-Object Editing")
        with gr.Row():
            with gr.Column():
                model_version = gr.Dropdown(
                    ['1.4', '1.5', '2.0', 'ip'],
                    value="2.0",
                    label="Stable Diffusion checkpoint", info="Select the SD checkpoint to use!"
                )
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Source Image", value=None, interactive=True)
                source_prompt = gr.Textbox(label="Source Prompt",
                                        value=None,
                                        interactive=True)
                caption_btn = gr.Button(value="Generate prompt")
                with gr.Row():
                    ddim_steps = gr.Slider(label="Null-text Inversion Steps",
                                        minimum=1,
                                        maximum=999,
                                        value=50,
                                        step=1)
                    steps = gr.Slider(label="Number of Inference Steps",
                                            minimum=0,
                                            maximum=999,
                                            value=50,
                                            step=1)
                    bootstrapping = gr.Slider(label="Number of bootstrapping iterations",
                                            minimum=0,
                                            maximum=50,
                                            value=25,
                                            step=1)
                with gr.Row():
                    token_position = gr.Number(label="Token Positions",
                                                value=None,
                                                interactive=True)
                    ca_coef = gr.Slider(label="CA Loss Coefficient",
                                        minimum=0,
                                        maximum=5,
                                        value=1.0,
                                        step=0.25)
                    seg_coef = gr.Slider(label="Background Loss Coefficient",
                                         minimum=0,
                                         maximum=5,
                                         value=1.75,
                                         step=0.25)
                
                run_btn = gr.Button(value="Run")
            with gr.Column():
                appended_prompt = gr.Textbox(label="Edited Prompts", value='')
                negative_prompt = gr.Textbox(label="Negative Prompts", value='')
                with gr.Row():
                    num_masks = gr.Slider(label="Number of masks",
                                    minimum=1,
                                    maximum=5,
                                    value=2,
                                    step=1)
                    seed = gr.Slider(label="Seed",
                                    minimum=-1,
                                    maximum=2147483647,
                                    value=1999,
                                    step=1)

        gr.Markdown("## **Output**")
        with gr.Row():
            original_image = gr.Image(label="Original Image")
            reconstructed_image = gr.Image(label="Reconstructed Image")
            edited_image = gr.Image(label="Edited Image")

        inputs = [
            source_image, source_prompt, model_version, steps, bootstrapping, token_position,
            ca_coef, seg_coef, appended_prompt, negative_prompt, num_masks, seed
        ]
        inputs_prompt = [source_image]
        
        caption_btn.click(generate_prompt, inputs_prompt, source_prompt)
        run_btn.click(mfmoe_edit, inputs,
                    [original_image, reconstructed_image, edited_image])

    return app


if __name__ == '__main__':
    mfmoe_app = mfmoe_launcher()
    mfmoe_app.launch()