"""
Semantic / feature-level watermarking (lightweight placeholder).
Uses edge/landmark proxies (Sobel-based heatmaps) to embed bits via small
perturbations, keeping geometric consistency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(img):
    # img: [B,3,H,W] in [0,1]
    gray = img.mean(dim=1, keepdim=True)
    kx = img.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
    ky = img.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    # More stable normalization: use mean instead of max
    mag_mean = mag.mean(dim=(2, 3), keepdim=True)
    mag = mag / (mag_mean + 1e-3)  # Normalize by mean, not max
    return mag.clamp(0, 10)  # Cap at 10x mean to avoid extreme amplification


class SemanticWatermarkEncoder(nn.Module):
    def __init__(self, watermark_len=64, max_shift_px=2):
        super().__init__()
        self.watermark_len = watermark_len
        self.max_shift_px = max_shift_px
        # Simple MLP to map bits to spatial offset map
        self.mapper = nn.Sequential(
            nn.Linear(watermark_len, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.to_offset = nn.Linear(256, 2)  # dx, dy per image (global tiny shift)
        self.to_mask_scale = nn.Linear(256, 1)

    def forward(self, images, bits):
        B, _, H, W = images.shape
        emb = self.mapper(bits)
        shift = torch.tanh(self.to_offset(emb)) * (self.max_shift_px / max(H, W))
        scale = torch.sigmoid(self.to_mask_scale(emb)).view(B, 1, 1, 1)

        edge_map = sobel_edges(images)  # [B, 1, H, W]

        # Create a gentle flow field using global shift scaled by edges
        # dx, dy are per-image scalars that we broadcast to [B, H, W]
        dx = shift[:, 0].view(B, 1, 1).expand(B, H, W)  # [B, H, W]
        dy = shift[:, 1].view(B, 1, 1).expand(B, H, W)  # [B, H, W]
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=images.device),
            torch.linspace(-1, 1, W, device=images.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Flow field: [B, H, W, 2] where last dim is (dx, dy)
        flow = torch.stack([dx, dy], dim=-1)  # [B, H, W, 2]
        # Scale by edge strength and learned scale
        edge_scale = edge_map.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
        scale_expanded = scale.squeeze(1).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        flow = flow * edge_scale * scale_expanded  # [B, H, W, 2]

        # Warp image slightly (semantic embedding)
        warped = F.grid_sample(images, (base_grid + flow), align_corners=True)
        perturbation = warped - images
        protected = (images + perturbation).clamp(0, 1)

        return {
            "protected_images": protected,
            "edge_map": edge_map,
            "flow": flow,
            "perturbation": perturbation,
        }


class SemanticWatermarkDecoder(nn.Module):
    def __init__(self, watermark_len=64):
        super().__init__()
        self.watermark_len = watermark_len
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, watermark_len),
        )

    def forward(self, images):
        edges = sobel_edges(images)
        feat = self.extractor(edges)
        logits = self.head(feat)
        return logits
