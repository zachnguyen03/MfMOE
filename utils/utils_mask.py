import torch
import numpy as np
from scipy import ndimage
from PIL import Image

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