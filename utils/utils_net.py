import torch
from diffusers.models.attention import CrossAttention

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from math import ceil

from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat

from ptp_utils import view_images

class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.cross_attns = []
        self.cross_attns_step = []

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0
        # self.cross_attns = []
        # self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.cross_attns) == 0:
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.cross_attns)):
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.cross_attns_step = []

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 16 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    
    def get_average_attention(self):
        average_attention = {'up_cross': [item / self.cur_step for item in self.cross_attns]}
        return average_attention

def aggregate_attention(attention_store: AttentionStore, prompts, res_h: int, res_w:int, from_where: List[str], is_cross: bool, select: int):
    out = []
    # print("# of attention maps: ", attention_store.cross_attns)
    attention_maps = attention_store.get_average_attention()
    num_pixels = res_h * res_w
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res_h, res_w, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_all_cross_attention(attention_store: AttentionStore, 
                             prompts: List[str], 
                             tokenizer = None,
                             from_where: List[str] = ['up','down'], 
                             original_resolution=(512, 512),
                             save_path='ca_vis'):
    tokens = tokenizer.encode(prompts[0])
    res_h, res_w = ceil(original_resolution[0]/32), ceil(original_resolution[1]/32)
    attention_maps = aggregate_attention(attention_store, [prompts[0]], res_h, res_w, from_where, True, 0)
    images = []
    print("Tokens: ", tokens)
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((original_resolution[1], original_resolution[0])))
        images.append(image)
    view_images(np.stack(images, axis=0), img_path=save_path+"/crossattn.png")

def get_attention_scores(self, query, key, attention_mask=None):
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # print("Query: ", query.shape)
    # print("Key: ", key.shape)
    attention_scores = torch.baddbmm(
        torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=self.scale,
    )

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if self.upcast_softmax:
        attention_scores = attention_scores.float()
    
    attention_scores = attention_scores/1.25
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(dtype)

    return attention_probs

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=1)
        # print("To q: ", attn.to_q)
        # print("To k: ", attn.to_k)

        query = attn.to_q(hidden_states)
        # if encoder_hidden_states is not None:
        #     print("Hidden state: ", hidden_states.shape)
        #     print("Encoder hidden state: ", encoder_hidden_states.shape)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn.get_attention_scores = get_attention_scores.__get__(attn, type(attn))
        # print("Query: ", query.shape)
        # print("Key: ", key.shape)
        # print("Attention mask: ", attention_mask.shape)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if hidden_states.shape[0] == 4: 
            attn.hs = hidden_states[2:3]
        # get attention map of trg
        else:
            attn.hs = hidden_states[1:2]

        return hidden_states


"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_processor(MyCrossAttnProcessor())
    return unet
