import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
from main import seed_everything, MfMOEPipeline
from utils.nti import NullInversion

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float16)


out_dir = './results/'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
checkpoint = "2.0"
sd = MfMOEPipeline(device, checkpoint)
nti = NullInversion(sd)

global_context = {
    "out_dir": out_dir,
    "checkpoint": checkpoint,
    "pipeline": sd,
    "device": device,
    "inversion": nti
}