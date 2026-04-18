import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Metrics ---

def calculate_psnr(img1, img2):
    # Use clamp to avoid division by zero and keep tensor ops on device
    mse = F.mse_loss(img1, img2)
    mse = torch.clamp(mse, min=1e-12)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ber(pred_bits, target_bits):
    """
    pred_bits: [B, len] (probs or thresholded)
    target_bits: [B, len] (0 or 1)
    """
    # Guard against numerical saturation before rounding
    pred_binary = (pred_bits.clamp(1e-4, 1 - 1e-4) > 0.5).float()
    errors = (pred_binary != target_bits).float().sum()
    total = target_bits.numel()
    return errors / total

# Simplified SSIM for training/metrics
# OPTIMIZED: Cache gaussian window per device
_ssim_cache = {}

def gaussian_window(size, sigma, device, dtype):
    key = (size, sigma, device, dtype)
    if key not in _ssim_cache:
        coords = torch.arange(size, dtype=dtype, device=device)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        _ssim_cache[key] = g.view(1, -1) * g.view(-1, 1)
    return _ssim_cache[key]

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5, img1.device, img1.dtype)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# --- Losses ---

class WatermarkLoss(nn.Module):
    """
    Combined loss for watermarking: Image Quality (MSE + SSIM) + Watermark Accuracy (BCE).
    
    FIXED: Now uses BCEWithLogitsLoss for numerical stability.
    Expects pred_bits to be LOGITS (before sigmoid), not probabilities.
    
    Weight guidelines (sync with attack curriculum):
    - Phase 1-2 (weak/no attacks): w_bce high, w_mse/w_ssim low (focus on learning)
    - Phase 3-4 (strong attacks): balanced weights (need quality for robustness)
    """
    def __init__(self, w_mse=0.1, w_ssim=0.1, w_bce=1.0, use_logits=True):
        super().__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.w_bce = w_bce
        self.use_logits = use_logits
        
        # BCEWithLogitsLoss is more numerically stable than BCELoss + clamp
        # It uses the log-sum-exp trick internally
        if use_logits:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            # Fallback for backward compatibility
            self.bce = nn.BCELoss(reduction='mean')

    def forward(self, orig_img, wm_img, pred_bits, target_bits):
        # 1. Reconstruction Loss (MSE) - for PSNR
        loss_mse = F.mse_loss(wm_img, orig_img)
        
        # 2. Perceptual Loss (SSIM) - maximize SSIM => minimize 1 - SSIM
        # Only compute if weight > 0 (saves computation in early phases)
        if self.w_ssim > 0:
            loss_ssim = 1 - ssim(wm_img, orig_img)
        else:
            loss_ssim = torch.tensor(0.0, device=orig_img.device)
        
        # 3. Watermark Extraction Loss (BCE)
        # FIXED: Using BCEWithLogitsLoss for numerical stability
        if self.use_logits:
            # pred_bits are raw logits, BCEWithLogitsLoss applies sigmoid internally
            loss_bce = self.bce(pred_bits, target_bits)
        else:
            # Backward compatibility: pred_bits are probabilities
            pred_clamped = pred_bits.clamp(1e-4, 1 - 1e-4)
            loss_bce = self.bce(pred_clamped, target_bits)
        
        # Weighted sum
        total_loss = self.w_mse * loss_mse + self.w_ssim * loss_ssim + self.w_bce * loss_bce
        
        return total_loss, {
            "loss_mse": loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse,
            "loss_ssim": loss_ssim.item() if torch.is_tensor(loss_ssim) else loss_ssim,
            "loss_bce": loss_bce.item() if torch.is_tensor(loss_bce) else loss_bce
        }


def calculate_ber_from_logits(pred_logits, target_bits):
    """Calculate BER from logits (before sigmoid)."""
    pred_binary = (pred_logits > 0).float()  # logits > 0 means prob > 0.5
    errors = (pred_binary != target_bits).float().sum()
    total = target_bits.numel()
    return errors / total

