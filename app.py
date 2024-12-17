import gradio as gr
from utils.utils_net import prep_unet, AttentionStore, register_attention_control, get_token_cross_attention
import time
import os
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
from utils.utils_mask import show_all_attention_maps, postprocess_mask

logging.set_verbosity_error()


def mfmoe_launcher():
    with gr.Blocks() as app:
        pass


if __name__ == '__main__':
    mfmoe_launcher()