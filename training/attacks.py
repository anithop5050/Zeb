import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- 1. Differentiable JPEG Utils ---

# Torch-native random helpers for reproducibility
def _torch_choice(options, probs=None):
    if probs is None:
        idx = torch.randint(0, len(options), (1,)).item()
        return options[idx]
    probs_t = torch.tensor(probs, dtype=torch.float32)
    idx = torch.multinomial(probs_t, 1).item()
    return options[idx]

class DiffJPEG(nn.Module):
    def __init__(self, quality: int = 80):
        super().__init__()
        self.quality = quality
        # Conversion matrices as buffers
        self.register_buffer(
            'ycbcr_weights',
            torch.tensor([
                [0.299, 0.587, 0.114],
                [-0.1687, -0.3313, 0.5],
                [0.5, -0.4187, -0.0813]
            ], dtype=torch.float32).view(3, 3, 1, 1)
        )
        self.register_buffer('ycbcr_bias', torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32).view(3))
        self.register_buffer(
            'rgb_weights',
            torch.tensor([
                [1.0, 0.0, 1.402],
                [1.0, -0.344136, -0.714136],
                [1.0, 1.772, 0.0]
            ], dtype=torch.float32).view(3, 3, 1, 1)
        )
        # Base JPEG quantization tables
        self.register_buffer('lum_q_base', torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32))
        self.register_buffer('chrom_q_base', torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float32))

    def _scaled_qtables(self, quality: int):
        q = int(quality)
        if q <= 0:
            q = 1
        if q > 100:
            q = 100
        scale = 5000 / q if q < 50 else 200 - 2 * q
        tbs = torch.stack([self.lum_q_base, self.chrom_q_base])
        tbs = (tbs * scale + 50) / 100
        tbs = tbs.clamp_(min=1, max=255)
        return tbs

    def forward(self, img: torch.Tensor, quality: int = None) -> torch.Tensor:
        quality = self.quality if quality is None else quality
        # Keep original size
        orig_h, orig_w = img.shape[2], img.shape[3]
        # Scale to [0,255]
        x = img * 255.0
        # Pad to multiple of 8
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        bs, _, h, w = x.shape
        # RGB -> YCbCr
        yuv = F.conv2d(x, self.ycbcr_weights, bias=self.ycbcr_bias)
        # Block splitting
        yuv = yuv.view(bs * 3, 1, h, w)
        patches = F.unfold(yuv, kernel_size=8, stride=8)
        patches = patches.transpose(1, 2).view(bs * 3, -1, 8, 8)
        # DCT
        patches_dct = dct_8x8(patches - 128)
        # Quantization
        q_table = self._scaled_qtables(quality)
        q_y = q_table[0].unsqueeze(0).unsqueeze(0)
        q_c = q_table[1].unsqueeze(0).unsqueeze(0)
        patches_dct_y = patches_dct[:bs]
        patches_dct_u = patches_dct[bs:2 * bs]
        patches_dct_v = patches_dct[2 * bs:]
        def round_diff(t):
            return t + (torch.round(t) - t).detach()
        q_dct_y = round_diff(patches_dct_y / q_y) * q_y
        q_dct_u = round_diff(patches_dct_u / q_c) * q_c
        q_dct_v = round_diff(patches_dct_v / q_c) * q_c
        patches_dct_quant = torch.cat([q_dct_y, q_dct_u, q_dct_v], dim=0)
        # IDCT
        patches_idct = idct_8x8(patches_dct_quant) + 128
        # Reassemble
        patches_flat = patches_idct.view(bs * 3, -1, 64).transpose(1, 2)
        rec = F.fold(patches_flat, output_size=(h, w), kernel_size=8, stride=8)
        rec = rec.view(bs, 3, h, w)
        # YCbCr -> RGB
        rec = rec - self.ycbcr_bias.view(1, 3, 1, 1)
        y = rec[:, 0:1, :, :]
        cb = rec[:, 1:2, :, :]
        cr = rec[:, 2:3, :, :]
        r = y + 1.402 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.772 * cb
        rgb_img = torch.cat([r, g, b], dim=1)
        # Crop back and rescale to [0,1]
        rgb_img = rgb_img[:, :, :orig_h, :orig_w]
        return torch.clamp(rgb_img / 255.0, 0, 1)

# Backward-compatible functional API with a cached module instance
_DIFF_JPEG = None
def diff_jpeg(img, quality=80):
    global _DIFF_JPEG
    if _DIFF_JPEG is None:
        _DIFF_JPEG = DiffJPEG(quality=quality)
    # Move module to the image device if needed and update default quality
    if _DIFF_JPEG.ycbcr_weights.device != img.device:
        _DIFF_JPEG = _DIFF_JPEG.to(img.device)
    _DIFF_JPEG.quality = quality
    return _DIFF_JPEG(img, quality=quality)

def dct_8x8(x, _cache={}):
    """
    Standard 8x8 DCT type II.
    x: [..., 8, 8]
    OPTIMIZED: Caches DCT matrix per device.
    """
    key = (x.device, x.dtype)
    if key not in _cache:
        pi = math.pi
        mat = torch.zeros((8, 8), device=x.device, dtype=x.dtype)
        for k in range(8):
            for n in range(8):
                norm = math.sqrt(1/8) if k == 0 else math.sqrt(2/8)
                mat[k, n] = norm * math.cos(pi/8 * (n + 0.5) * k)
        _cache[key] = mat
    
    mat = _cache[key]
    original_shape = x.shape
    x_flat = x.reshape(-1, 8, 8)
    x_dct = torch.matmul(mat, torch.matmul(x_flat, mat.t()))
    return x_dct.view(original_shape)

def idct_8x8(x, _cache={}):
    """
    Standard 8x8 IDCT type III.
    OPTIMIZED: Caches DCT matrix per device.
    """
    key = (x.device, x.dtype)
    if key not in _cache:
        pi = math.pi
        mat = torch.zeros((8, 8), device=x.device, dtype=x.dtype)
        for k in range(8):
            for n in range(8):
                norm = math.sqrt(1/8) if k == 0 else math.sqrt(2/8)
                mat[k, n] = norm * math.cos(pi/8 * (n + 0.5) * k)
        _cache[key] = mat
    
    mat = _cache[key]
    original_shape = x.shape
    x_flat = x.reshape(-1, 8, 8)
    x_idct = torch.matmul(mat.t(), torch.matmul(x_flat, mat))
    return x_idct.view(original_shape)

def get_jpeg_quantization_table(quality, device):
    """
    Returns standard JPEG quantization tables (Luma, Chroma) scaled by quality.
    """
    if quality <= 0: quality = 1
    if quality > 100: quality = 100
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
        
    # Standard Luminance Quantization Table
    lum_q = torch.tensor([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=torch.float, device=device)

    # Standard Chrominance Quantization Table
    chrom_q = torch.tensor([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=torch.float, device=device)
    
    tbs = torch.stack([lum_q, chrom_q])
    tbs = (tbs * scale + 50) / 100
    tbs[tbs < 1] = 1
    tbs[tbs > 255] = 255
    return tbs

# --- 2. Existing Layers ---

def random_noise(img, std_range=(0.01, 0.05)):
    if hasattr(std_range, '__iter__'):
        std = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(*std_range)
    else:
        std = std_range
    return torch.clamp(img + torch.randn_like(img) * std, 0, 1)

def random_blur(img, kernel_size=5, sigma_range=(0.5, 1.5)):
    k = kernel_size
    sigma = torch.empty(1).uniform_(*sigma_range).item()
    x_coord = torch.arange(k)
    x_grid = x_coord.repeat(k).view(k, k)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (k - 1) / 2.
    variance = sigma**2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    kernel = gaussian_kernel.view(1, 1, k, k).repeat(3, 1, 1, 1).to(img.device)
    pad = k // 2
    return F.conv2d(img, kernel, padding=pad, groups=3)


def resize_jpeg_resize(img, scale_range=(0.5, 0.9), quality_range=(60, 85)):
    """
    Compound attack: Resize down -> JPEG compress -> Resize back up.
    Common in social media pipelines (Twitter, Instagram, etc.).
    """
    B, C, H, W = img.shape
    scale = torch.empty(1).uniform_(*scale_range).item()
    new_h, new_w = int(H * scale), int(W * scale)
    
    # Resize down
    img_small = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # JPEG compress
    q = int(torch.randint(quality_range[0], quality_range[1], (1,)).item())
    try:
        img_jpeg = diff_jpeg(img_small, quality=q)
    except:
        img_jpeg = img_small
    
    # Resize back up
    img_restored = F.interpolate(img_jpeg, size=(H, W), mode='bilinear', align_corners=False)
    return img_restored

def random_geometry(img, max_trans=0.1, max_scale=0.1, max_rotate=10):
    B = img.shape[0]
    theta = torch.eye(2, 3, device=img.device).unsqueeze(0).repeat(B, 1, 1)
    tx = torch.empty(B).uniform_(-max_trans, max_trans).to(img.device)
    ty = torch.empty(B).uniform_(-max_trans, max_trans).to(img.device)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    s = torch.empty(B).uniform_(1 - max_scale, 1 + max_scale).to(img.device)
    angle_deg = torch.empty(B).uniform_(-max_rotate, max_rotate).to(img.device)
    angle_rad = angle_deg * np.pi / 180.0
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    theta[:, 0, 0] = cos_a * s
    theta[:, 0, 1] = -sin_a * s
    theta[:, 1, 0] = sin_a * s
    theta[:, 1, 1] = cos_a * s
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    x_aug = F.grid_sample(img, grid, align_corners=False)
    return x_aug


def simulated_generative_attack(img, strength='medium'):
    """
    Simulate artifacts from generative AI models (deepfakes, enhancers, etc.)
    without needing a pretrained model.
    
    Generative models typically introduce:
    1. Slight blur (encoder-decoder bottleneck effect)
    2. Color quantization artifacts
    3. High-frequency noise in flat regions
    4. Subtle color shifts
    
    Args:
        img: [B, 3, H, W] in [0, 1]
        strength: 'weak', 'medium', or 'strong'
    """
    B = img.shape[0]
    device = img.device
    
    # Strength presets
    if strength == 'weak':
        blur_sigma = (0.3, 0.6)
        noise_std = (0.005, 0.015)
        color_shift = 0.02
        quant_levels = 64  # Higher = less quantization
    elif strength == 'medium':
        blur_sigma = (0.5, 1.0)
        noise_std = (0.01, 0.025)
        color_shift = 0.04
        quant_levels = 48
    else:  # strong
        blur_sigma = (0.8, 1.4)
        noise_std = (0.02, 0.04)
        color_shift = 0.06
        quant_levels = 32
    
    # 1. Slight Gaussian blur (mimics encoder-decoder reconstruction)
    k = 5
    sigma = torch.empty(1).uniform_(*blur_sigma).item()
    x_coord = torch.arange(k)
    x_grid = x_coord.repeat(k).view(k, k)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (k - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    kernel = gaussian_kernel.view(1, 1, k, k).repeat(3, 1, 1, 1).to(device)
    img_blur = F.conv2d(img, kernel, padding=k // 2, groups=3)
    
    # 2. Color quantization WITH STE (Straight-Through Estimator)
    # Forward: returns quantized value
    # Backward: gradients pass through as identity (enables encoder learning)
    scaled = img_blur * quant_levels
    img_quant = img_blur + (torch.round(scaled) / quant_levels - img_blur).detach()
    
    # 3. Add structured noise (more in flat regions, less in edges)
    # Compute local variance to detect flat regions
    img_gray = img_quant.mean(dim=1, keepdim=True)
    local_mean = F.avg_pool2d(img_gray, 5, stride=1, padding=2)
    local_sq_mean = F.avg_pool2d(img_gray ** 2, 5, stride=1, padding=2)
    local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
    
    # More noise in flat regions (low variance)
    noise_mask = 1.0 / (1.0 + 50 * local_var)  # High in flat, low in textured
    noise_std_val = torch.empty(1).uniform_(*noise_std).item()
    noise = torch.randn_like(img) * noise_std_val * noise_mask
    img_noisy = img_quant + noise
    
    # 4. Subtle color shift (generative models often have slight color drift)
    color_delta = torch.empty(B, 3, 1, 1, device=device).uniform_(-color_shift, color_shift)
    img_shifted = img_noisy + color_delta
    
    # 5. Clamp to valid range
    return torch.clamp(img_shifted, 0, 1)


class AttackSimulationLayer(nn.Module):
    """
    Applies stochastic attacks with curriculum learning.
    Training curriculum MUST be synchronized with loss weight scheduling in colab.py!
    
    PHASES (synced with colab.py):
    - Phase 0 (0-2000): Message-only training - NO attacks
    - Phase 1 (2000-8000): Basic encoding - NO attacks  
    - Phase 2 (8000-15000): Weak attacks
    - Phase 3 (15000-25000): Moderate attacks
    - Phase 4 (25000+): Full attacks
    """
    def __init__(self):
        super().__init__()
        self.attack_types = ['noise', 'blur', 'jpeg', 'geometry', 'dropout', 'color', 'generative']

    def forward(self, img, global_step=0):
        """
        Apply attacks based on curriculum phase.
        """
        if global_step is None: 
            global_step = 999999  # Default to full if unknown
        
        # Phase 0-1: Pure identity (steps 0-8000)
        # Let encoder/decoder learn basic watermarking without interference
        if global_step < 8000:
            return img
        
        # Phase 2: Very weak attacks (steps 8000-15000)
        elif global_step < 15000:
            attack = _torch_choice(['identity', 'noise', 'jpeg'], probs=[0.5, 0.25, 0.25])
            if attack == 'identity':
                return img
            elif attack == 'noise':
                return random_noise(img, std_range=(0.005, 0.015))
            elif attack == 'jpeg':
                q = int(torch.randint(88, 98, (1,)).item())
                try:
                    return diff_jpeg(img, quality=q)
                except:
                    return img
        
        # Phase 3: Moderate attacks (steps 15000-25000)
        # Added JPEG composition (25% chance after non-JPEG attacks)
        elif global_step < 25000:
            attack = _torch_choice(
                ['identity', 'noise', 'jpeg', 'blur', 'color'],
                probs=[0.20, 0.20, 0.30, 0.15, 0.15]  # Increased JPEG probability
            )
            if attack == 'identity':
                return img
            elif attack == 'noise':
                out = random_noise(img, std_range=(0.01, 0.03))
            elif attack == 'jpeg':
                q = int(torch.randint(75, 92, (1,)).item())
                try:
                    return diff_jpeg(img, quality=q)
                except:
                    return img
            elif attack == 'blur':
                out = random_blur(img, sigma_range=(0.4, 0.9))
            elif attack == 'color':
                B = img.shape[0]
                alpha = torch.empty(B, 1, 1, 1, device=img.device).uniform_(0.9, 1.1)
                beta = torch.empty(B, 1, 1, 1, device=img.device).uniform_(-0.04, 0.04)
                out = torch.clamp(alpha * img + beta, 0, 1)
            else:
                out = img
            
            # JPEG composition: 25% chance for non-JPEG attacks
            if attack not in ['identity', 'jpeg'] and torch.rand(1).item() < 0.25:
                q = int(torch.randint(80, 95, (1,)).item())
                try:
                    out = diff_jpeg(out, quality=q)
                except:
                    pass
            return out
        
        # Phase 4: Full spectrum (steps 25000+)
        # Now includes proper generative attack simulation
        # CRITICAL: Force JPEG composition for robustness
        else:
            attack = _torch_choice(
                ['noise', 'blur', 'jpeg', 'geometry', 'dropout', 'color', 'generative', 'compound'],
                probs=[0.12, 0.12, 0.20, 0.12, 0.08, 0.08, 0.13, 0.15]
            )
            
            if attack == 'noise':
                out = random_noise(img, std_range=(0.015, 0.05))
            elif attack == 'blur':
                out = random_blur(img, sigma_range=(0.6, 1.4))
            elif attack == 'jpeg':
                q = int(torch.randint(60, 85, (1,)).item())
                try:
                    out = diff_jpeg(img, quality=q)
                except:
                    out = img
                return out  # Pure JPEG, no composition needed
            elif attack == 'geometry':
                out = random_geometry(img, max_trans=0.1, max_scale=0.1, max_rotate=10)
            elif attack == 'dropout':
                mask = torch.rand_like(img) > 0.12
                out = img * mask.float()
            elif attack == 'color':
                B = img.shape[0]
                alpha = torch.empty(B, 1, 1, 1, device=img.device).uniform_(0.82, 1.18)
                beta = torch.empty(B, 1, 1, 1, device=img.device).uniform_(-0.06, 0.06)
                out = torch.clamp(alpha * img + beta, 0, 1)
            elif attack == 'generative':
                strength = _torch_choice(['weak', 'medium', 'strong'], probs=[0.3, 0.5, 0.2])
                out = simulated_generative_attack(img, strength=strength)
            elif attack == 'compound':
                # Resize -> JPEG -> Resize (social media pipeline)
                return resize_jpeg_resize(img, scale_range=(0.5, 0.85), quality_range=(55, 80))
            else:
                out = img
            
            # JPEG COMPOSITION: 40% chance to apply JPEG after non-JPEG attacks
            # This ensures the model sees JPEG artifacts frequently
            if torch.rand(1).item() < 0.40:
                q = int(torch.randint(70, 90, (1,)).item())
                try:
                    out = diff_jpeg(out, quality=q)
                except:
                    pass
            
            return out
        
        return img
    
    def apply_hard_attacks(self, img):
        """
        Apply a random HARD attack for validation.
        Always applies a meaningful attack (no identity).
        """
        attack = _torch_choice(['noise', 'jpeg', 'blur', 'geometry', 'color', 'compound'])
        
        if attack == 'noise':
            return random_noise(img, std_range=(0.02, 0.05))
        elif attack == 'jpeg':
            q = int(torch.randint(60, 80, (1,)).item())
            try:
                return diff_jpeg(img, quality=q)
            except:
                return random_noise(img, std_range=(0.03, 0.05))
        elif attack == 'blur':
            return random_blur(img, sigma_range=(1.0, 1.5))
        elif attack == 'geometry':
            return random_geometry(img, max_trans=0.1, max_scale=0.1, max_rotate=10)
        elif attack == 'color':
            B = img.shape[0]
            alpha = torch.empty(B, 1, 1, 1, device=img.device).uniform_(0.8, 1.2)
            beta = torch.empty(B, 1, 1, 1, device=img.device).uniform_(-0.08, 0.08)
            return torch.clamp(alpha * img + beta, 0, 1)
        elif attack == 'compound':
            # Social media pipeline simulation
            return resize_jpeg_resize(img, scale_range=(0.5, 0.8), quality_range=(55, 75))
        return img
