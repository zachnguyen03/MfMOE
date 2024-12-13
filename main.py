from utils.utils_net import prep_unet, AttentionStore, register_attention_control, get_token_cross_attention
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# from ddim_inv import ddim_inversion
from lavis.models import load_model_and_preprocess, load_model

from utils.nti import NullInversion
from utils.ptp_utils import *
from utils.utils_mask import attention_to_binary_mask, postprocess_mask, gaussian_map

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
            model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet").to(self.device)
        self.unet = prep_unet(self.unet)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler")

        self.d_ref_t2attn = {}  
        self.image_latent_ref = {} 
        self.attention_store = {'up_cross': [], 'mid_cross': [], 'down_cross': []}

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
    def reconstruct(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, latent=None, latent_path=None, latent_list_path=None, num_fgmasks=2):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)

        # latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        # latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

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
                        if attn_mask.shape[1] == 16 * 16: #save only 16x16 maps
                            if 'up' in name:
                                self.attention_store['up_cross'].append(attn_mask.detach().cpu())
                            elif 'mid' in name:
                                self.attention_store['mid_cross'].append(attn_mask.detach().cpu())
                            else: # add down block cross attention
                                self.attention_store['down_cross'].append(attn_mask.detach().cpu())
                            attn_mask = torch.cat(tuple([attn_mask] * num_fgmasks), dim=0)
                            self.d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                # if latent_list is not None:
                #     noise_loss_list.append(latent_list[-2-i] - latents_view_denoised)
                #     latents_view_denoised = latents_view_denoised + noise_loss_list[-1]

            latent = latents_view_denoised
            self.image_latent_ref[t.item()] = latent.detach().cpu()

        # for key in self.d_ref_t2attn.keys():
        #     print(self.d_ref_t2attn[key].keys())
        #     print("Number of attention maps per timestep: ", len(self.d_ref_t2attn[key]))
        binary_mask = get_token_cross_attention(self.d_ref_t2attn, prompts, self.tokenizer, timestep=21, block='up_blocks.1.attentions.2.transformer_blocks.0.attn2', token_idx=2)
        cv2.imwrite('./results/mask.png', binary_mask)
        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor))
        img = T.ToPILImage()(imgs[0].cpu())
        return img, noise_loss_list
    
    def invert(
        self,
        start_latents,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        do_classifier_free_guidance=True,
        negative_prompt="",
        ):

        # Encode prompt
        text_embeddings = self.get_text_embeds(prompt, negative_prompt)

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)

    def generate(self, masks, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50, guidance_scale=7.5, bootstrapping=20, ca_coef=1, seg_coef=0.25, noise_loss_list=None, latent=None, latent_path=None, latent_list_path=None):

        bootstrapping_backgrounds = self.get_random_background(bootstrapping)
        print("Bootstrap background shape: ", bootstrapping_backgrounds.shape)

        text_embeds = self.get_text_embeds(prompts, negative_prompts).type(torch.cuda.HalfTensor)
        print("Text embeds shape: ", text_embeds.shape)
        # latent = torch.load(latent_path).unsqueeze(0).to(self.device)
        # latent_list = [x.to(self.device) for x in torch.load(latent_list_path)]

        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        
        print("Noise shape: ", noise.shape)
        print("Views: ", views)
        print("Latent shape: ", latent.shape)

        mask_tensor = masks[0]
        mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = torch.Tensor(np.array([np.array(mask_tensor.cpu())] * 4)).to(device)
        

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            count.zero_()
            value.zero_()

            masks_view = masks
            latent_view = latent.repeat(len(prompts), 1, 1, 1) # latents for all prompts
            # print("Prompts: ", prompts) # [original prompt, edited prompt1,...,n]
            if i < bootstrapping:
                bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                bg = self.scheduler.add_noise(bg, noise, t)
                latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (1 - masks_view[1:])
                # print("Latent view: ", latent_view[1:].shape)

            latent_model_input = torch.cat([latent_view] * 2).type(torch.cuda.HalfTensor)

            x_in = latent_model_input.detach().clone()
            x_in.requires_grad = True

            opt = torch.optim.SGD([x_in], lr=0.1)

            noise_pred = self.unet(x_in, t, encoder_hidden_states=text_embeds.detach())['sample']

            loss = 0.0
            loss_ca = 0.0
            loss_seg = 0.0
            for name, module in self.unet.named_modules():
                module_name = type(module).__name__
                if module_name == "CrossAttention" and 'attn2' in name:
                    curr = module.attn_probs
                    if curr.shape[1] == 16 * 16:
                        ref = self.d_ref_t2attn[t.item()][name].detach().to(device)
                        loss_ca += ((curr-ref)**2).sum((1, 2)).mean(0)

            latents = x_in.chunk(2)[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_view_denoised = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            # if noise_loss_list is not None:
            #     latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True) # 00:57
            latent = torch.where(count > 0, latent / count, latent) # 00:57

            latent_cur = latent.squeeze(0)
            latent_ref = self.image_latent_ref[t.item()].detach().to(device).squeeze(0)

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
            # if noise_loss_list is not None:
            #     latents_view_denoised=latents_view_denoised+noise_loss_list[i]

            latent = (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count = masks_view.sum(dim=0, keepdims=True)
            latent = torch.where(count > 0, latent / count, latent)

        imgs = self.decode_latents(latent.type(torch.cuda.HalfTensor)) # half tensor for computational efficiency
        img = T.ToPILImage()(imgs[0].cpu())
        return img


def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float16) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--source_prompt', type=str)
    parser.add_argument('--mask_paths', nargs='+')
    parser.add_argument('--rec_path', type=str)
    parser.add_argument('--edit_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--fg_prompts', nargs='+')
    parser.add_argument('--fg_negative', nargs='+')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.4', '1.5', '2.0', 'ip'], help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bootstrapping', type=int, default=20)
    parser.add_argument('--num_fgmasks', type=int, default=1)
    parser.add_argument('--ca_coef', type=float, default=1.0)
    parser.add_argument('--seg_coef', type=float, default=1.75)

    opt = parser.parse_args()

    device = torch.device('cuda:0')
    
    torch.set_default_dtype(torch.float32)
    start = time.time()
    
    if opt.source_prompt is not None:
        prompt_str =  opt.source_prompt
    else:
    # Initialize BLIP captioner
        model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
        img = Image.open(opt.image_path).resize((512,512), Image.Resampling.LANCZOS)
        # generate the caption
        _image = vis_processors["eval"](img).unsqueeze(0).to(device)
        prompt_str = model_blip.generate({"image": _image})[0]
        print("Generated source prompt: ", prompt_str)
        
        del model_blip
        del vis_processors

    ca_coef = opt.ca_coef
    seg_coef = opt.seg_coef

    print(ca_coef, seg_coef)

    seed = opt.seed
    seed_everything(seed)

    sd = MfMOEPipeline(device, opt.sd_version)
    nti = NullInversion(sd)
    
    # Set number of inversion steps
    sd.scheduler.num_inference_steps = 50
    
    # Inversion
    (image_gt, image_enc), x_t, uncond_embeddings = nti.invert(opt.image_path, prompt_str, offsets=(0, 0, 0, 0), verbose=True)
    print("Latent shape: ", x_t.shape)    
    
    del nti
    masks = None
    
    # controller = AttentionStore()
    # register_attention_control(sd, controller)

    prompts = [prompt_str]
    neg_prompts = [prompt_str]

    rec_img, noise_loss_list = sd.reconstruct(masks, prompts, neg_prompts, opt.H, opt.W, opt.steps, bootstrapping=opt.bootstrapping, latent=x_t, latent_path=None, latent_list_path=None, num_fgmasks=opt.num_fgmasks+1)
    rec_img.save(opt.rec_path)

    att_map = sd.attention_store['up_cross'][-1]
    att_map = postprocess_mask(att_map, 3)
    att_map = att_map.save('./results/mask_up.png')

    fg_masks = torch.cat([preprocess_mask(mask_path, opt.H // 8, opt.W // 8, device) for mask_path in opt.mask_paths])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    prompts = [prompt_str] + opt.fg_prompts
    neg_prompts = [prompt_str] + opt.fg_negative
    
    print("Prompts: ", prompts)
    print("Negative prompts: ", neg_prompts)
    start_gen = time.time()

    img = sd.generate(masks, prompts, neg_prompts, opt.H, opt.W, opt.steps, bootstrapping=opt.bootstrapping, ca_coef=ca_coef, seg_coef=seg_coef, noise_loss_list=noise_loss_list, latent=x_t, latent_path=None, latent_list_path=None)

    end = time.time()

    img.save(opt.edit_path) 

    images = [rec_img, img]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    if opt.save_path:
        new_im.save(opt.save_path)
        
    print(f"Total inference time: {end - start} seconds")
    print(f"Editing time: {end - start_gen} seconds")