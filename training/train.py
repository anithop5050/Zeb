"""
Watermark Training Script
=========================
Improved training with:
- Alpha annealing (strong -> subtle)
- Learning rate warmup
- Synchronized attack/loss curriculum
- Better hyperparameters
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import glob
import argparse
import math
import random
import numpy as np

# Import our modules
from training.models import RobustWatermarkEncoder, WatermarkDecoder
from training.attacks import AttackSimulationLayer
from training.utils_loss_metrics import WatermarkLoss, calculate_ber, calculate_psnr, calculate_ber_from_logits
from training.adversarial_poison import AdversarialPoisoner
from training.semantic_watermark import SemanticWatermarkEncoder, SemanticWatermarkDecoder


# ==============================================================================
# Configuration
# ==============================================================================

BATCH_SIZE = 4           # Safe default for 14-16 GB GPUs
GRADIENT_ACCUMULATION_STEPS = 4
LR = 5e-4                # Lower base LR (warmup will handle ramp-up)
EPOCHS = 50              # More epochs for proper training
IMAGE_SIZE = 128         # Resize training images to a bounded resolution
DUMMY_IMAGE_SIZE = 128  # used only when no dataset images exist
WATERMARK_LEN = 64
USE_AMP = True

# Multi-defense weights
PIXEL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 0.7
POISON_WEIGHT = 0.3
PERTURB_EPS = 0.02  # overall L_inf budget

# Alpha annealing: Start visible, end invisible
ALPHA_START = 0.3        # Strong initially (easier to learn)
ALPHA_END = 0.1          # Subtle at end (invisible)
ALPHA_ANNEAL_STEPS = 40000  # Extended: alpha should stay higher during hard attacks (Phase 4 at 25k)

# Warmup
WARMUP_STEPS = 1000

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="dataset", help='Path to image dataset')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--save_dir', type=str, default="/content/drive/MyDrive/model", help='Directory to save checkpoints (Google Drive)')
args, _ = parser.parse_known_args()
DATASET_PATH = args.data_dir
SAVE_DIR = args.save_dir

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================================
# Dataset
# ==============================================================================

class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
                     glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True) + \
                     glob.glob(os.path.join(root_dir, "**", "*.jpeg"), recursive=True)
        
        if not self.files:
            print(f"⚠️ No images found in {root_dir}. Using random noise for testing.")
            self.files = ["dummy"] * 500
            self.dummy = True
        else:
            self.dummy = False
            print(f"✅ Found {len(self.files)} images.")
            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.dummy:
            return torch.rand(3, DUMMY_IMAGE_SIZE, DUMMY_IMAGE_SIZE)
        
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.rand(3, DUMMY_IMAGE_SIZE, DUMMY_IMAGE_SIZE)


# ==============================================================================
# Training Utilities
# ==============================================================================

def get_alpha(global_step):
    """Anneal alpha from ALPHA_START to ALPHA_END."""
    if global_step >= ALPHA_ANNEAL_STEPS:
        return ALPHA_END
    
    # Cosine annealing
    progress = global_step / ALPHA_ANNEAL_STEPS
    alpha = ALPHA_END + (ALPHA_START - ALPHA_END) * (1 + math.cos(math.pi * progress)) / 2
    return alpha


def get_lr_multiplier(global_step):
    """Warmup learning rate."""
    if global_step < WARMUP_STEPS:
        return global_step / WARMUP_STEPS
    return 1.0


def pad_to_multiple(imgs, multiple=8):
    """Pad images so H,W are divisible by `multiple`; return padded and pad tuple.
    
    Uses 'replicate' (edge) padding which is safer than 'reflect' for large pads.
    """
    _, _, h, w = imgs.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom) in F.pad order
    if pad_h == 0 and pad_w == 0:
        return imgs, pad
    return F.pad(imgs, pad, mode='replicate'), pad


def crop_to_original(imgs, pad):
    """Remove padding added by pad_to_multiple.
    
    pad is (left, right, top, bottom) from F.pad.
    To crop: remove `right` cols from right, `bottom` rows from bottom.
    """
    left, right, top, bottom = pad
    if (left, right, top, bottom) == (0, 0, 0, 0):
        return imgs
    # F.pad adds: left to left-edge, right to right-edge, top to top-edge, bottom to bottom-edge
    # To undo: slice away the padded pixels
    h, w = imgs.shape[2], imgs.shape[3]
    h_end = h - bottom if bottom > 0 else h
    w_end = w - right if right > 0 else w
    return imgs[:, :, top:h_end, left:w_end]


def crop_batch_to_original(imgs, base_pads, extra_pad):
    """Crop a batch back to per-sample original sizes using stored paddings.
    
    Args:
        imgs: [B, C, H, W] padded batch
        base_pads: list of (left, right, top, bottom) per sample from collate
        extra_pad: (left, right, top, bottom) from pad_to_multiple
    Returns:
        list of tensors with original resolutions (use list since sizes may differ)
    """
    l_extra, r_extra, t_extra, b_extra = extra_pad
    crops = []
    h_total, w_total = imgs.shape[2], imgs.shape[3]
    for idx, pad in enumerate(base_pads):
        l = pad[0] + l_extra
        r = pad[1] + r_extra
        t = pad[2] + t_extra
        b = pad[3] + b_extra
        h_end = h_total - b if b > 0 else h_total
        w_end = w_total - r if r > 0 else w_total
        crops.append(imgs[idx:idx+1, :, t:h_end, l:w_end])
    return crops  # Return list since sizes may differ


def collate_pad_to_largest(batch, multiple=8):
    """Pad all images in a batch to the largest H,W (rounded up to `multiple`).
    
    Uses 'replicate' mode (edge padding) which is safer than 'reflect' when
    padding might exceed input dimensions.
    """
    # batch: list of tensors [3, H, W]
    h_max = max(img.shape[1] for img in batch)
    w_max = max(img.shape[2] for img in batch)
    # round up to multiple for safer down/upsampling
    h_target = math.ceil(h_max / multiple) * multiple
    w_target = math.ceil(w_max / multiple) * multiple
    padded = []
    pads = []
    orig_sizes = []
    for img in batch:
        _, h, w = img.shape
        pad_h = h_target - h
        pad_w = w_target - w
        pad = (0, pad_w, 0, pad_h)
        pads.append(pad)
        orig_sizes.append((h, w))
        # Use 'replicate' (edge padding) which works for any pad size
        # 'reflect' fails when pad > dimension
        padded.append(F.pad(img.unsqueeze(0), pad, mode='replicate').squeeze(0))
    return torch.stack(padded, dim=0), pads, orig_sizes


def update_loss_weights(criterion, global_step):
    """
    Synchronized loss weight curriculum.
    Phases match attack curriculum in attacks.py.
    """
    if global_step < 5000:
        # Phase 1: No attacks - focus purely on watermark learning
        criterion.w_mse = 0.05
        criterion.w_ssim = 0.0
        criterion.w_bce = 5.0  # Very high BCE focus
    elif global_step < 12000:
        # Phase 2: Weak attacks - still prioritize watermark
        criterion.w_mse = 0.1
        criterion.w_ssim = 0.05
        criterion.w_bce = 3.0
    elif global_step < 25000:
        # Phase 3: Moderate attacks - balance quality
        criterion.w_mse = 0.2
        criterion.w_ssim = 0.1
        criterion.w_bce = 2.0
    else:
        # Phase 4: Full attacks - balanced
        criterion.w_mse = 0.3
        criterion.w_ssim = 0.2
        criterion.w_bce = 1.5


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    use_amp = USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # Reproducibility
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _worker_init_fn(worker_id):
        # Make dataloader workers deterministic if num_workers>0
        seed = 42 + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(42)
    
    # Setup Models
    encoder = RobustWatermarkEncoder(watermark_len=WATERMARK_LEN).to(device)
    decoder = WatermarkDecoder(watermark_len=WATERMARK_LEN).to(device)
    attacker = AttackSimulationLayer().to(device)
    poisoner = AdversarialPoisoner(eps=PERTURB_EPS, steps=1, step_size=None).to(device)
    semantic_encoder = SemanticWatermarkEncoder(watermark_len=WATERMARK_LEN).to(device)
    semantic_decoder = SemanticWatermarkDecoder(watermark_len=WATERMARK_LEN).to(device)
    
    # Print model info
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    sem_enc_params = sum(p.numel() for p in semantic_encoder.parameters())
    sem_dec_params = sum(p.numel() for p in semantic_decoder.parameters())
    print(f"📊 Encoder params: {enc_params:,}")
    print(f"📊 Decoder params: {dec_params:,}")
    print(f"📊 Semantic Encoder params: {sem_enc_params:,}")
    print(f"📊 Semantic Decoder params: {sem_dec_params:,}")
    print(f"📊 Total params: {enc_params + dec_params + sem_enc_params + sem_dec_params:,}")
    
    # Optimizer (poisoner has no trainable params)
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(semantic_encoder.parameters()) + list(semantic_decoder.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    bce_logits = nn.BCEWithLogitsLoss()
    
    # Loss
    criterion = WatermarkLoss(w_mse=0.05, w_ssim=0.0, w_bce=5.0).to(device)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor()
    ])
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    # Create full dataset and split into train/val
    full_dataset = WatermarkDataset(DATASET_PATH, transform=transform)
    total_size = len(full_dataset)
    val_size = max(1, int(total_size * 0.1)) if total_size > 1 else 1
    train_size = max(0, total_size - val_size)
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn if 0 > 0 else None,
        collate_fn=collate_pad_to_largest,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn if 0 > 0 else None,
        collate_fn=collate_pad_to_largest,
    )
    
    # Resume or start fresh
    start_epoch = 0
    global_step = 0
    best_ber = 1.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if 'poisoner' in checkpoint:
            poisoner.load_state_dict(checkpoint['poisoner'])
        if 'semantic_encoder' in checkpoint:
            semantic_encoder.load_state_dict(checkpoint['semantic_encoder'])
        if 'semantic_decoder' in checkpoint:
            semantic_decoder.load_state_dict(checkpoint['semantic_decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_ber = checkpoint.get('best_ber', 1.0)
        print(f"✅ Resumed from epoch {start_epoch-1}, step {global_step}")
    else:
        # Check for auto-resume
        if os.path.exists(os.path.join(SAVE_DIR, "best_model.pth")):
            print("📂 Found best_model.pth, loading...")
            checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            if 'poisoner' in checkpoint:
                poisoner.load_state_dict(checkpoint['poisoner'])
            if 'semantic_encoder' in checkpoint:
                semantic_encoder.load_state_dict(checkpoint['semantic_encoder'])
            if 'semantic_decoder' in checkpoint:
                semantic_decoder.load_state_dict(checkpoint['semantic_decoder'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            best_ber = checkpoint.get('best_ber', 1.0)
        else:
            print("🆕 Starting fresh training...")
    
    # Training info
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"  Alpha: {ALPHA_START} → {ALPHA_END} over {ALPHA_ANNEAL_STEPS} steps")
    print(f"  LR: {LR} with {WARMUP_STEPS} warmup steps")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  AMP enabled: {use_amp}")
    print(f"  Watermark length: {WATERMARK_LEN} bits")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        epoch_ber = 0
        epoch_sem_ber = 0
        epoch_psnr = 0
        
        encoder.train()
        decoder.train()
        semantic_encoder.train()
        semantic_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(train_loader):
            imgs, base_pads, orig_sizes = batch
            imgs = imgs.to(device, non_blocking=torch.cuda.is_available())
            imgs_padded, pad = pad_to_multiple(imgs, multiple=8)
            bs = imgs.size(0)
            next_step = global_step + 1
            should_step = ((i + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_loader))
            
            # Generate random watermarks
            wm = torch.randint(0, 2, (bs, WATERMARK_LEN), device=device).float()
            
            # Get current alpha
            alpha = get_alpha(next_step)
            
            # Update loss weights
            update_loss_weights(criterion, next_step)

            with autocast(enabled=use_amp):
                # Pixel-level embed (on padded images)
                encoded_imgs = encoder(imgs_padded, wm, alpha=alpha)

                # Semantic layer
                semantic_out = semantic_encoder(encoded_imgs, wm)
                sem_images = semantic_out["protected_images"]

                # Adversarial poison
                poisoned_imgs, poison_delta = poisoner(sem_images)

                # Attacks
                attacked_imgs = attacker(poisoned_imgs, global_step=next_step)

                # Decode pixel bits
                pred_bits = decoder(attacked_imgs)

                # Decode semantic bits (use pre-attack to keep gradient stable)
                semantic_logits = semantic_decoder(poisoned_imgs)

                # Losses
                pixel_loss, loss_dict = criterion(imgs_padded, encoded_imgs, pred_bits, wm)
                semantic_loss = bce_logits(semantic_logits, wm)

                # Landmark/flow consistency: keep semantic flow tiny
                flow_mag = torch.norm(semantic_out["flow"], dim=-1)
                consistency_loss = flow_mag.mean() * 10.0

                # Perturbation budget (overall)
                total_delta = poisoned_imgs - imgs_padded
                budget_violation = torch.relu(total_delta.abs() - PERTURB_EPS)
                budget_loss = budget_violation.mean() * 50.0

                # Poison regularizer (small but present)
                poison_reg = poison_delta.abs().mean()

                total_loss = (
                    PIXEL_WEIGHT * pixel_loss +
                    SEMANTIC_WEIGHT * semantic_loss +
                    0.1 * consistency_loss +
                    POISON_WEIGHT * poison_reg +
                    budget_loss
                )

            scaled_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()

            if should_step:
                lr_mult = get_lr_multiplier(next_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR * lr_mult

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()) +
                    list(semantic_encoder.parameters()) + list(semantic_decoder.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step = next_step

                if device.type == "cuda" and global_step % 10 == 0:
                    torch.cuda.empty_cache()

            # Metrics
            ber = calculate_ber_from_logits(pred_bits, wm)
            sem_ber = calculate_ber_from_logits(semantic_logits, wm)
            psnr = calculate_psnr(encoded_imgs, imgs_padded)

            epoch_loss += total_loss.item()
            epoch_ber += ber.item()
            epoch_sem_ber += sem_ber.item()
            epoch_psnr += psnr.item()
            
            # Logging
            if i % 20 == 0:
                phase_step = global_step if should_step else next_step
                phase = "P1" if phase_step < 5000 else "P2" if phase_step < 12000 else "P3" if phase_step < 25000 else "P4"
                print(f"[E{epoch}][{i}/{len(train_loader)}][{phase}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(pix:{loss_dict['loss_bce']:.3f}|sem:{semantic_loss.item():.3f}|bud:{budget_loss.item():.3f}) "
                      f"BER pix:{ber.item():.4f} sem:{sem_ber.item():.4f} PSNR:{psnr.item():.1f} "
                      f"α:{alpha:.3f}")
                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"  GPU Memory: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Epoch summary
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_ber = epoch_ber / n_batches
        avg_sem_ber = epoch_sem_ber / n_batches
        avg_psnr = epoch_psnr / n_batches
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Complete")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg BER (pixel): {avg_ber:.4f} {'✅' if avg_ber < 0.1 else '⚠️' if avg_ber < 0.3 else '❌'}")
        print(f"  Avg BER (semantic): {avg_sem_ber:.4f}")
        print(f"  Avg PSNR: {avg_psnr:.2f} dB")
        print(f"  Alpha: {get_alpha(global_step):.3f}")
        print(f"  Steps: {global_step}")
        
        # Validation with hard attacks
        encoder.eval()
        decoder.eval()
        val_ber = 0.0
        val_batches = min(5, len(val_loader)) if len(val_loader) > 0 else 0
        if val_batches > 0:
            with torch.no_grad():
                for b_idx, val_batch in enumerate(val_loader):
                    if b_idx >= val_batches:
                        break
                    val_imgs, val_base_pads, val_orig_sizes = val_batch
                    val_imgs = val_imgs.to(device, non_blocking=torch.cuda.is_available())
                    val_imgs_pad, val_pad = pad_to_multiple(val_imgs, multiple=8)
                    val_wm = torch.randint(0, 2, (val_imgs.size(0), WATERMARK_LEN), device=device).float()
                    with autocast(enabled=use_amp):
                        val_encoded = encoder(val_imgs_pad, val_wm, alpha=ALPHA_END)  # Use final alpha
                        val_attacked = attacker(val_encoded, global_step=999999)
                        val_pred = decoder(val_attacked)  # Returns logits
                    val_ber += calculate_ber_from_logits(val_pred, val_wm).item()
            val_ber /= val_batches
        else:
            val_ber = float('nan')
        print(f"  Validation BER (Hard): {val_ber:.4f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if avg_ber < best_ber:
            best_ber = avg_ber
            print(f"🏆 New best model! BER: {best_ber:.4f}")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'poisoner': poisoner.state_dict(),
                'semantic_encoder': semantic_encoder.state_dict(),
                'semantic_decoder': semantic_decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_ber': best_ber,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
        
        # Save checkpoint every epoch to Drive
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'poisoner': poisoner.state_dict(),
            'semantic_encoder': semantic_encoder.state_dict(),
            'semantic_decoder': semantic_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_ber': best_ber,
        }, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("✅ Training complete!")
    print(f"   Best BER achieved: {best_ber:.4f}")


if __name__ == "__main__":
    train()
