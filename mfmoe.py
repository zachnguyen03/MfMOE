# Mask-free Multi-Object Editing 

from utils.utils_net import prep_unet, AttentionStore
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from ddim_inv import ddim_inversion
from utils.utils_mask import preprocess_mask

logging.set_verbosity_error()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float16)

def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MfMOEPipeline(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        else:
            model_key = self.sd_version

        self.vae = AutoencoderKL.from_pretrained(
            model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder").toCrossAttention
        self.unet = prep_unet(self.unet)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler")

        self.d_ref_t2attn = {}  
        self.image_latent_ref = {} 

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_random_background(self, n_samples):
        backgrounds = torch.rand(n_samples, 3, device=self.device)[
            :, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def reconstruct(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, latent_path=None, latent_list_path=None, num_fgmasks=2):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)

        latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        noise_loss_list=[]
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            count.zero_()
            value.zero_()

            for h_start, h_end, w_start, w_end in views:
                latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(len(prompts), 1, 1, 1)

                latent_model_input = torch.cat([latent_view] * 2).type(torch.cuda.HalfTensor)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                self.d_ref_t2attn[t.item()] = {}
                self.image_latent_ref[t.item()] = {}
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "CrossAttention" and 'attn2' in name:
                        attn_mask = module.attn_probs
                        attn_mask = torch.cat(tuple([attn_mask] * num_fgmasks), dim=0)
                        self.d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                if latent_list is not None:
                    noise_loss_list.append(latent_list[-2-i] - latents_view_denoised)
                    latents_view_denoised = latents_view_denoised + noise_loss_list[-1]

            latent = latents_view_denoised
            self.image_latent_ref[t.item()] = latent.detach().cpu()

        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor))
        img = T.ToPILImage()(imgs[0].cpu())
        return img, noise_loss_list

    def generate(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, ca_coef=1, seg_coef=0.25, noise_loss_list=None, latent_path=None, latent_list_path=None):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)

        latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        
        print("Noise shape: ", noise.shape)
        print("Views: ", views)
        print("Latent shape: ", latent.shape)

        mask_tensor = masks[0]
        mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = torch.Tensor(np.array([np.array(mask_tensor.cpu())] * 4)).to(self.device)
        
        print("Masks size: ", mask_tensor.shape)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            count.zero_()
            value.zero_()

            masks_view = masks
            latent_view = latent.repeat(len(prompts), 1, 1, 1) # latents for all prompts
            if i < bootstrapping:
                bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                bg = self.scheduler.add_noise(bg, noise, t)
                latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (1 - masks_view[1:])

            latent_model_input = torch.cat([latent_view] * 2).type(torch.cuda.HalfTensor)

            x_in = latent_model_input.detach().clone()
            x_in.requires_grad = True
            
            print("Input to UNet shape: ", x_in.shape)

            opt = torch.optim.SGD([x_in], lr=0.1)

            noise_pred = self.unet(x_in, t, encoder_hidden_states=text_embeds.detach())['sample']

            loss = 0.0
            loss_ca = 0.0
            loss_seg = 0.0
            for name, module in self.unet.named_modules():
                module_name = type(module).__name__
                if module_name == "CrossAttention" and 'attn2' in name:
                    curr = module.attn_probs
                    ref = self.d_ref_t2attn[t.item()][name].detach().to(self.device)
                    loss_ca += ((curr-ref)**2).sum((1, 2)).mean(0)

            latents = x_in.chunk(2)[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_view_denoised = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            if noise_loss_list is not None:
                latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True) # 00:57
            latent = torch.where(count > 0, latent / count, latent) # 00:57

            latent_cur = latent.squeeze(0)
            latent_ref = self.image_latent_ref[t.item()].detach().to(self.device).squeeze(0)

            loss_seg += (torch.multiply(mask_tensor, latent_cur - latent_ref)**2).sum((1, 2)).mean(0)

            torch.cuda.empty_cache()
            loss = ca_coef * loss_ca + seg_coef * loss_seg
            loss.backward(retain_graph=False)
            opt.step()

            with torch.no_grad():
                noise_pred = self.unet(x_in.detach(), t, encoder_hidden_states=text_embeds)['sample']

            latents = x_in.detach().chunk(2)[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_view_denoised = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            if noise_loss_list is not None:
                latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True)
            latent = torch.where(count > 0, latent / count, latent)

        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor)) # half tensor for computational efficiency
        img = T.ToPILImage()(imgs[0].cpu())
        return img
