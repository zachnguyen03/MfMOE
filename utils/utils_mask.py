import torch
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
from scipy.ndimage import label, find_objects, measurements

from utils.ptp_utils import view_images


def postprocess_mask(att_map, idx, gaussian=0, binarize_threshold=64, h=512, w=512, save_path=None, device=None):
    att_map = att_map.sum(0) / att_map.shape[0]
    att_map = att_map.reshape(16,16,77)
    att_map = att_map[:,:,idx]
    att_map = 255 * att_map / att_map.max()
    att_map = att_map.numpy().astype(np.uint8)
    att_map = Image.fromarray(att_map).resize((h, w))
    for i in range(gaussian):
        att_map = ndimage.gaussian_filter(att_map, sigma=(5,5), order=0)
    att_map[att_map < binarize_threshold] = 0
    att_map[att_map >= binarize_threshold] = 255
    # att_map = filter_small_regions(att_map)
    if save_path is not None:
        refined_mask = Image.fromarray(att_map)
        refined_mask = refined_mask.save(save_path)
    att_map = att_map[None, None]
    att_map = att_map.astype(np.float16) / 255.0
    att_map = torch.from_numpy(att_map).to(device)
    mask = torch.nn.functional.interpolate(att_map, size=(h//8, w//8), mode='nearest')
    return mask

def show_all_attention_maps(att_maps, prompt_length, save_path):
    images = []
    att_maps = att_maps.sum(0) / att_maps.shape[0]
    att_maps = att_maps.reshape(16,16,77)
    for token in range(prompt_length):
        att_map = att_maps[:,:,token+1] # first token is <START> -> omit
        att_map = 255 * att_map / att_map.max()
        att_map = Image.fromarray(att_map.numpy().astype(np.uint8)).resize((256, 256))
        att_map.save(f'{save_path}/attn_{token+1}.png')
        images.append(att_map)
    # view_images(np.stack(images, axis=0), img_path=save_path+"/crossattn.png")

def gaussian_map(att_map):
    return ndimage.gaussian_filter(att_map, sigma=(5,5), order=0)

def binarize_map(att_map, threshold=128):
    att_map[att_map < threshold] = 0
    att_map[att_map >= threshold] = 1
    return att_map

def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float16) / 255.0
    mask = mask[None, None]
    
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

def attention_to_binary_mask(attention_map, threshold=0.5, min_region_size=100):
    """
    Convert an attention map to a binary mask using thresholding and optional post-processing.
    
    Args:
        attention_map (np.ndarray): Input attention map of shape (H, W) with values in [0, 1]
        threshold (float): Threshold value for binarization (default: 0.5)
        min_region_size (int): Minimum size of connected regions to keep (default: 100)
    
    Returns:
        np.ndarray: Binary mask of the same shape as input with values in {0, 1}
    """
    # Ensure input is numpy array
    attention_map = np.array(attention_map)
    
    # Normalize attention map to [0, 1] if needed
    if attention_map.max() > 1.0 or attention_map.min() < 0.0:
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Apply threshold to create initial binary mask
    binary_mask = (attention_map > threshold).astype(np.uint8)
    
    # Optional: Remove small connected components
    if min_region_size > 0:
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        # Calculate size of each component
        component_sizes = np.bincount(labeled_array.ravel())
        
        # Create mask of components to keep
        keep_components = component_sizes > min_region_size
        keep_components[0] = True  # Keep background
        
        # Filter small components
        binary_mask = keep_components[labeled_array]
    
    # Optional: Fill holes in the mask
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    return binary_mask.astype(np.uint8)


def filter_small_regions(mask, region_thres=64*255):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    labeled_image, num_features = label(mask)
    regions = find_objects(labeled_image)
    for i, region in enumerate(regions):
        if region is not None:
            area = measurements.sum(labeled_image[region], labeled_image[region], index=i+1)
            if area < region_thres:
                mask[region] = 0
    return mask