import numpy as np
from PIL import Image, ImageOps
from albumentations import (Compose, VerticalFlip, HorizontalFlip, RandomRotate90, Transpose,
                            GaussNoise)

def resize_image(org_img, img_size = 448):
    # Zero padding along smaller dimension
    x, y = org_img.size
    if x < y:
        add_px = (y - x) // 2
        padded_img = ImageOps.expand(org_img, border=(add_px, 0, add_px, 0), fill=0)
    else:
        add_px = (x - y) // 2
        padded_img = ImageOps.expand(org_img, border=(0, add_px, 0, add_px), fill=0)
    #print(org_img.size, padded_img.size)

    return np.asarray(padded_img.resize((img_size, img_size), resample=Image.Resampling.LANCZOS))

def apply_whitebalancing(img, percentile = 99):
    # White patch white balance
    img_float = img.astype(np.float32)
    
    # Find the top intensity threshold for each channel based on the percentile
    max_b = np.percentile(img_float[:, :, 0], percentile)
    max_g = np.percentile(img_float[:, :, 1], percentile)
    max_r = np.percentile(img_float[:, :, 2], percentile)
    
    # Target value (pure white)
    target = 255.0
    
    # Scale channels, protecting against division by zero
    img_float[:, :, 0] *= (target / max(max_b, 1.0))
    img_float[:, :, 1] *= (target / max(max_g, 1.0))
    img_float[:, :, 2] *= (target / max(max_r, 1.0))
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def add_augmentations(img):
    augmentations = Compose([VerticalFlip(), HorizontalFlip(), RandomRotate90(),
                             Transpose(), GaussNoise()], p = 1)
    aug_img = augmentations(image = img)
    return aug_img["image"]