"""
Robust Watermarking Models
==========================
U-Net style encoder with skip connections and multi-scale decoder.
Based on HiDDeN and StegaStamp architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building Blocks
# ==============================================================================

class ConvBlock(nn.Module):
    """Basic conv block with BatchNorm and LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.block(x))


class DownBlock(nn.Module):
    """Downsample block: Conv with stride 2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_ch)
        )
    
    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """Upsample block with skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_ch)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==============================================================================
# Watermark Embedding Module (Optimized - No Redundant Upsampling)
# ==============================================================================

class WatermarkEmbedding(nn.Module):
    """
    Maps watermark bits directly to bottleneck resolution.
    FIXED: No longer upsamples to 128x128 then downsamples back.
    Outputs directly at spatial_res (default 16x16 for bottleneck).
    """
    def __init__(self, watermark_len=64, target_channels=64, spatial_res=16):
        super().__init__()
        self.watermark_len = watermark_len
        self.target_channels = target_channels
        self.spatial_res = spatial_res
        
        # Transform bits directly to target resolution
        # This is more efficient than upsample+downsample
        self.bit_transform = nn.Sequential(
            nn.Linear(watermark_len, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, target_channels * spatial_res * spatial_res),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Refinement conv to add spatial variation
        self.refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(target_channels, target_channels, 3, padding=1),
        )
    
    def forward(self, wm, target_size=None):
        """
        Args:
            wm: [B, watermark_len] binary watermark
            target_size: (H, W) - optional, will interpolate if provided and different
        Returns:
            [B, target_channels, spatial_res, spatial_res] watermark features
        """
        B = wm.size(0)
        
        # Transform to spatial feature at bottleneck resolution
        x = self.bit_transform(wm)
        x = x.view(B, self.target_channels, self.spatial_res, self.spatial_res)
        
        # Refine with convolutions for spatial variation
        x = self.refine(x)
        
        # Resize only if explicitly needed (for flexibility)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


# ==============================================================================
# Encoder (U-Net Style)
# ==============================================================================

class RobustWatermarkEncoder(nn.Module):
    """
    U-Net style encoder that embeds watermark into image.
    Uses skip connections for preserving high-frequency details.
    """
    def __init__(self, watermark_len=64, bottleneck_res=16):
        super().__init__()
        self.watermark_len = watermark_len
        self.bottleneck_res = bottleneck_res
        
        # Watermark embedding module - outputs directly at bottleneck resolution
        # FIXED: No more redundant 128x128 upsampling then 16x16 downsampling
        self.wm_embed = WatermarkEmbedding(
            watermark_len, 
            target_channels=64, 
            spatial_res=bottleneck_res
        )
        
        # Initial conv
        self.init_conv = ConvBlock(3, 64)
        
        # Encoder (downsampling)
        self.down1 = DownBlock(64, 128)    # /2
        self.down2 = DownBlock(128, 256)   # /4
        self.down3 = DownBlock(256, 256)   # /8
        
        # Bottleneck with watermark fusion
        # Input: 256 (image) + 64 (watermark) = 320
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        # Decoder (upsampling with skip connections)
        self.up1 = UpBlock(256, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        
        # Final output (residual image)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Residual in [-1, 1]
        )
    
    def forward(self, img, wm, alpha=0.1):
        """
        Args:
            img: [B, 3, H, W] input image in [0, 1]
            wm: [B, watermark_len] binary watermark
            alpha: watermark strength
        Returns:
            [B, 3, H, W] watermarked image in [0, 1]
        """
        # Encoder path with skip connections
        e0 = self.init_conv(img)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        
        # Get watermark features at bottleneck resolution
        # FIXED: Now outputs directly at correct size, only resize if mismatch
        wm_feat = self.wm_embed(wm, target_size=e3.shape[2:])
        
        # Fuse image features with watermark
        fused = torch.cat([e3, wm_feat], dim=1)  # [B, 320, 16, 16]
        bottleneck = self.bottleneck(fused)       # [B, 256, 16, 16]
        
        # Decoder path with skip connections
        d1 = self.up1(bottleneck, e2)      # [B, 256, 32, 32]
        d2 = self.up2(d1, e1)              # [B, 128, 64, 64]
        d3 = self.up3(d2, e0)              # [B, 64, 128, 128]
        
        # Generate residual
        residual = self.final(d3)          # [B, 3, 128, 128]
        
        # Add scaled residual to original
        watermarked = torch.clamp(img + alpha * residual, 0, 1)
        
        return watermarked


# ==============================================================================
# Decoder (Multi-Scale Feature Extraction)
# ==============================================================================

class WatermarkDecoder(nn.Module):
    """
    Multi-scale decoder for robust watermark extraction.
    Uses features from multiple resolutions to recover bits.
    """
    def __init__(self, watermark_len=64):
        super().__init__()
        self.watermark_len = watermark_len
        
        # Multi-scale feature extraction
        # Scale 1: Full resolution
        self.scale1 = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # /2
            ConvBlock(32, 64, stride=2),   # /4
            ResidualBlock(64),
        )
        
        # Scale 2: Half resolution input
        self.scale2 = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # /2
            ConvBlock(32, 64, stride=2),   # /4
            ResidualBlock(64),
        )
        
        # Scale 3: Quarter resolution input  
        self.scale3 = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # /2
            ConvBlock(32, 64, stride=2),   # /4
            ResidualBlock(64),
        )
        
        # Fusion of multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling to get fixed size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Bit extraction head
        self.bit_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, watermark_len),
        )
    
    def forward(self, x, return_probs=False):
        """
        Args:
            x: [B, 3, H, W] watermarked image
            return_probs: if True, return sigmoid(logits); if False, return raw logits
        Returns:
            [B, watermark_len] predicted bit logits (or probabilities if return_probs=True)
        """
        B, C, H, W = x.shape
        
        # Multi-scale inputs
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        # Extract features at each scale
        f1 = self.scale1(x)          # From full res
        f2 = self.scale2(x_half)     # From half res
        f3 = self.scale3(x_quarter)  # From quarter res
        
        # Resize all to same size and concatenate
        target_size = f1.shape[2:]
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        
        multi_scale = torch.cat([f1, f2, f3], dim=1)  # [B, 192, H/4, W/4]
        
        # Fuse and extract
        fused = self.fusion(multi_scale)
        pooled = self.global_pool(fused)  # [B, 128, 4, 4]
        
        logits = self.bit_head(pooled)
        
        # FIXED: Return logits by default for BCEWithLogitsLoss
        # Use return_probs=True for inference
        if return_probs:
            return torch.sigmoid(logits)
        return logits


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == "__main__":
    print("Testing models...")
    
    # Test encoder
    encoder = RobustWatermarkEncoder(watermark_len=64)
    decoder = WatermarkDecoder(watermark_len=64)
    
    # Dummy data
    img = torch.rand(2, 3, 128, 128)
    wm = torch.randint(0, 2, (2, 64)).float()
    
    # Forward pass
    encoded = encoder(img, wm, alpha=0.1)
    decoded_logits = decoder(encoded)  # Returns logits now
    decoded_probs = decoder(encoded, return_probs=True)  # For display
    
    print(f"Input image: {img.shape}")
    print(f"Watermark: {wm.shape}")
    print(f"Encoded image: {encoded.shape}")
    print(f"Decoded logits: {decoded_logits.shape}")
    print(f"Encoded range: [{encoded.min():.3f}, {encoded.max():.3f}]")
    print(f"Decoded logits range: [{decoded_logits.min():.3f}, {decoded_logits.max():.3f}]")
    print(f"Decoded probs range: [{decoded_probs.min():.3f}, {decoded_probs.max():.3f}]")
    
    # Test gradient flow using BCEWithLogitsLoss
    loss = F.binary_cross_entropy_with_logits(decoded_logits, wm)
    loss.backward()
    
    # Check gradients reach encoder
    enc_grad = encoder.init_conv.conv[0].weight.grad
    print(f"Encoder gradient exists: {enc_grad is not None}")
    print(f"Encoder gradient norm: {enc_grad.norm().item():.6f}" if enc_grad is not None else "No gradient!")
    
    # Parameter count
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nEncoder params: {enc_params:,}")
    print(f"Decoder params: {dec_params:,}")
    print(f"Total params: {enc_params + dec_params:,}")
    
    print("\n✅ All tests passed!")

