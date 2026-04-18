"""
Enhanced Training Script - Production Alpha Range with Auto-Save
=================================================================

Key Improvements:
1. ✅ Alpha range [0.020, 0.3] - matches production deployment
2. ✅ Auto-save to Google Drive every epoch
3. ✅ Resume from checkpoint with full state restoration
4. ✅ Low-alpha validation for production range
5. ✅ Curriculum learning for [0.020, 0.055] focus
6. ✅ Better logging and progress tracking

Usage:
  # Train from scratch
  python train_production.py --data_dir dataset --save_dir /content/drive/MyDrive/checkpoints
  
  # Resume from checkpoint
  python train_production.py --resume checkpoint.pth --save_dir /content/drive/MyDrive/checkpoints
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import glob
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path (supports Colab root-file layout)
sys.path.insert(0, os.getcwd())

# Import training modules. Prefer repo package imports, but keep the
# Colab root-file fallback because the notebook workflow copies files to /content.
try:
    from training.models import RobustWatermarkEncoder, WatermarkDecoder
    from training.semantic_watermark import SemanticWatermarkEncoder, SemanticWatermarkDecoder
    from training.adversarial_poison import AdversarialPoisoner
    from training.attacks import DiffJPEG, AttackSimulationLayer
except ImportError:
    from models import RobustWatermarkEncoder, WatermarkDecoder
    from semantic_watermark import SemanticWatermarkEncoder, SemanticWatermarkDecoder
    from adversarial_poison import AdversarialPoisoner
    from attacks import DiffJPEG, AttackSimulationLayer

# ==============================================================================
# TRAINING HYPERPARAMETERS - PRODUCTION CONFIGURATION
# ==============================================================================

BATCH_SIZE = 16
LR = 1e-4               # Very gentle for fine-tuning already-good model
EPOCHS = 89             # Resume from epoch 59 + 30 more epochs = total 89
IMAGE_SIZE = 128
WATERMARK_LEN = 64

# Multi-defense weights - FOCUSED on BER improvement
PIXEL_WEIGHT = 0.0005   # Even lower - barely touch image quality
BER_WEIGHT = 30.0       # Maximum priority - fix low-alpha BER only
SEMANTIC_WEIGHT = 0.05  # Minimal - don't disturb what already works
POISON_WEIGHT = 0.02    # Minimal
PERTURB_EPS = 0.02

# Attack curriculum - model already robust, keep attacks active
ATTACK_WARMUP_STEPS = 0  # No warmup needed - model knows attacks

# ✅ ULTRA-LOW ALPHA FINE-TUNING - Fix α=0.020-0.025 ONLY
# Model already excellent at α≥0.035, focus on the gap
ALPHA_START = 0.030      # Start just above production floor
ALPHA_END = 0.020        # PRODUCTION FLOOR
ALPHA_ANNEAL_STEPS = 500 # Reach α=0.020 quickly (4 epochs)

# Warmup
WARMUP_STEPS = 1000

# Auto-save configuration
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every epoch
KEEP_LAST_N_CHECKPOINTS = 5  # Keep only recent N checkpoints + best
VALIDATE_EVERY_N_EPOCHS = 5  # Run low-alpha validation every N epochs

# ==============================================================================
# DATASET
# ==============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        self.image_size = image_size
        
        # Find all images
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        self.image_paths = list(set(self.image_paths))  # Remove duplicates
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor


# ==============================================================================
# LOW-ALPHA VALIDATION (Production Range Testing)
# ==============================================================================

def validate_low_alpha(encoder, decoder, device, test_images, epoch):
    """Validate specifically on production alpha range [0.020, 0.055]."""
    encoder.eval()
    decoder.eval()
    
    test_alphas = [0.020, 0.025, 0.035, 0.045, 0.055]
    results = []
    
    with torch.no_grad():
        for alpha_test in test_alphas:
            alpha_ber = 0
            alpha_psnr = 0
            n_samples = min(5, len(test_images))
            
            for i in range(n_samples):
                img = test_images[i].unsqueeze(0).to(device)
                watermark = (torch.rand(1, WATERMARK_LEN, device=device) > 0.5).float()
                
                # Embed
                watermarked = encoder(img, watermark, alpha=alpha_test)
                
                # Decode
                extracted = decoder(watermarked)
                
                # Metrics
                ber = ((extracted > 0).float() != watermark).float().mean().item()
                psnr = -10 * torch.log10(((watermarked - img) ** 2).mean() + 1e-8).item()
                
                alpha_ber += ber
                alpha_psnr += psnr
            
            avg_ber = alpha_ber / n_samples
            avg_psnr = alpha_psnr / n_samples
            results.append((alpha_test, avg_ber, avg_psnr))
    
    encoder.train()
    decoder.train()
    
    # Log results
    logger.info("=" * 60)
    logger.info(f"LOW-ALPHA VALIDATION - Epoch {epoch}")
    logger.info("=" * 60)
    for alpha, ber, psnr in results:
        status = "✅" if ber < 0.05 else "⚠️"
        logger.info(f"  α={alpha:.3f}: BER={ber:.4f} {status}, PSNR={psnr:.2f} dB")
    logger.info("=" * 60)
    
    # Return average metrics across all alphas
    avg_ber = np.mean([r[1] for r in results])
    avg_psnr = np.mean([r[2] for r in results])
    
    return avg_ber, avg_psnr


# ==============================================================================
# CHECKPOINT MANAGEMENT
# ==============================================================================

def save_checkpoint(epoch, step, models, optimizer, scheduler, best_ber, save_dir, 
                   is_best=False, keep_last_n=5):
    """Save checkpoint with automatic cleanup of old checkpoints."""
    
    encoder, decoder, sem_enc, sem_dec, poisoner = models
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'semantic_encoder': sem_enc.state_dict(),
        'semantic_decoder': sem_dec.state_dict(),
        'poisoner': poisoner.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_ber': best_ber,
        'alpha_range': {'min': ALPHA_END, 'max': ALPHA_START},
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save regular checkpoint
    if not is_best:
        checkpoint_name = f'checkpoint_epoch_{epoch:03d}_step_{step}.pth'
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_name}")
        
        # Cleanup old checkpoints (keep only recent N)
        cleanup_old_checkpoints(save_dir, keep_last_n)
    
    # Always save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_production_range_BEST.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"🏆 Saved BEST checkpoint: BER={best_ber:.4f}")
    
    return checkpoint_path if not is_best else best_path


def cleanup_old_checkpoints(save_dir, keep_last_n):
    """Remove old checkpoints, keeping only the most recent N."""
    checkpoints = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime)
    
    # Remove oldest
    for old_ckpt in checkpoints[:-keep_last_n]:
        try:
            os.remove(old_ckpt)
            logger.debug(f"🗑️ Removed old checkpoint: {os.path.basename(old_ckpt)}")
        except Exception as e:
            logger.warning(f"Failed to remove {old_ckpt}: {e}")


def load_checkpoint(checkpoint_path, models, optimizer=None, scheduler=None, device='cuda'):
    """Load checkpoint and restore training state."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder, decoder, sem_enc, sem_dec, poisoner = models
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    sem_enc.load_state_dict(checkpoint['semantic_encoder'])
    sem_dec.load_state_dict(checkpoint['semantic_decoder'])
    poisoner.load_state_dict(checkpoint['poisoner'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    start_step = checkpoint.get('step', 0)
    best_ber = checkpoint.get('best_ber', 1.0)
    
    logger.info(f"✅ Resumed from epoch {start_epoch}, step {start_step}, best BER={best_ber:.4f}")
    
    return start_epoch, start_step, best_ber


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Checkpoints will save to: {args.save_dir}")
    
    # Dataset
    dataset = ImageDataset(args.data_dir, IMAGE_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Models
    encoder = RobustWatermarkEncoder(watermark_len=WATERMARK_LEN).to(device)
    decoder = WatermarkDecoder(watermark_len=WATERMARK_LEN).to(device)
    sem_encoder = SemanticWatermarkEncoder(watermark_len=WATERMARK_LEN).to(device)
    sem_decoder = SemanticWatermarkDecoder(watermark_len=WATERMARK_LEN).to(device)
    poisoner = AdversarialPoisoner(eps=PERTURB_EPS).to(device)  # Fixed: parameter is 'eps', not 'epsilon'
    
    models = (encoder, decoder, sem_encoder, sem_decoder, poisoner)
    
    # Optimizer and scheduler
    params = list(encoder.parameters()) + list(decoder.parameters()) + \
             list(sem_encoder.parameters()) + list(sem_decoder.parameters()) + \
             list(poisoner.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))
    
    # Attack layer
    attack_layer = AttackSimulationLayer().to(device)
    
    # Resume if checkpoint provided
    start_epoch = 0
    start_step = 0
    best_ber = 1.0
    
    if args.resume:
        start_epoch, start_step, best_ber = load_checkpoint(
            args.resume, models, optimizer, scheduler, device
        )
    
    # Initialize step counter (continue from checkpoint or start at 0)
    step = start_step
    
    # Training loop
    logger.info("\\n" + "=" * 60)
    if args.resume:
        logger.info("FINE-TUNING RESUMED")
    else:
        logger.info("TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Alpha range: [{ALPHA_END:.3f}, {ALPHA_START:.3f}]")
    logger.info(f"Epochs: {EPOCHS} (starting from epoch {start_epoch + 1})")
    logger.info(f"Learning rate: {LR}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Dataset: {len(dataset)} images")
    logger.info(f"Loss weights: PIXEL={PIXEL_WEIGHT}, BER={BER_WEIGHT}, SEM={SEMANTIC_WEIGHT}, POISON={POISON_WEIGHT}")
    logger.info(f"Attack warmup: {ATTACK_WARMUP_STEPS} steps (no attacks until then)")
    logger.info(f"Current best BER: {best_ber:.4f}")
    logger.info("=" * 60 + "\\n")
    
    # Get test images for validation
    test_images = [dataset[i] for i in range(min(10, len(dataset)))]
    
    for epoch in range(start_epoch, EPOCHS):
        encoder.train()
        decoder.train()
        sem_encoder.train()
        sem_decoder.train()
        poisoner.train()
        
        epoch_loss = 0
        epoch_ber = 0
        epoch_psnr = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, imgs in enumerate(pbar):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # Random watermarks
            watermarks = (torch.rand(batch_size, WATERMARK_LEN, device=device) > 0.5).float()
            
            # Alpha annealing with curriculum learning
            if step < ALPHA_ANNEAL_STEPS:
                # Linear annealing
                progress = step / ALPHA_ANNEAL_STEPS
                alpha = ALPHA_START - progress * (ALPHA_START - ALPHA_END)
            else:
                # ✅ EXTREME FOCUS: 95% at ultra-low alpha [0.020, 0.025]
                # This is the ONLY range that needs improvement
                if np.random.rand() < 0.95:
                    # 95% in problem zone
                    alpha = 0.020 + np.random.uniform(0, 0.005)  # [0.020, 0.025]
                else:
                    # 5% slightly higher to maintain robustness
                    alpha = 0.025 + np.random.uniform(0, 0.010)  # [0.025, 0.035]
            
            # Warmup - skip if resuming (step > 0 at start)
            if step < WARMUP_STEPS and start_step == 0:
                lr_mult = step / WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR * lr_mult
            
            # Forward pass
            optimizer.zero_grad()
            
            # Encode
            encoded = encoder(imgs, watermarks, alpha)
            
            # Apply attacks - FIXED: Disable attacks early to let model learn basics
            if step < ATTACK_WARMUP_STEPS:
                attacked = encoded  # No attacks during warmup
            else:
                attacked = attack_layer(encoded, global_step=step)
            
            # Decode
            decoded = decoder(attacked)
            
            # Semantic watermark (auxiliary training path)
            sem_encoded = sem_encoder(imgs, watermarks)["protected_images"]
            sem_decoded = sem_decoder(sem_encoded)
            
            # Adversarial poisoning - FIXED: Train decoder to reject poisoned images
            # Goal: Decoder should output LOW confidence (near 0.5) for poisoned clean images
            # NOT to output the watermark bits (which aren't there!)
            poisoned, _ = poisoner(imgs)
            poison_decoded = decoder(poisoned)
            # Target: random bits (0.5 probability) = model should be uncertain
            poison_target = torch.full_like(watermarks, 0.5)
            
            # Losses
            pixel_loss = F.mse_loss(encoded, imgs)
            ber_loss = F.binary_cross_entropy_with_logits(decoded, watermarks)
            sem_loss = F.binary_cross_entropy_with_logits(sem_decoded, watermarks)
            # FIXED: Poison loss should push toward uncertainty, not toward matching watermarks
            poison_loss = F.mse_loss(torch.sigmoid(poison_decoded), poison_target)
            
            # FIXED: Apply BER_WEIGHT to prioritize watermark learning
            loss = PIXEL_WEIGHT * pixel_loss + \
                   BER_WEIGHT * ber_loss + \
                   SEMANTIC_WEIGHT * sem_loss + \
                   POISON_WEIGHT * poison_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                ber = ((decoded > 0).float() != watermarks).float().mean().item()
                psnr = -10 * torch.log10(pixel_loss + 1e-8).item()
            
            epoch_loss += loss.item()
            epoch_ber += ber
            epoch_psnr += psnr
            
            step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'α': f'{alpha:.3f}',
                'BER': f'{ber:.3f}',
                'PSNR': f'{psnr:.1f}',
                'Loss': f'{loss.item():.3f}'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_ber = epoch_ber / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        
        logger.info(f"\\nEpoch {epoch+1} Summary: Loss={avg_loss:.4f}, BER={avg_ber:.4f}, PSNR={avg_psnr:.2f} dB")
        
        # Low-alpha validation
        if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
            val_ber, val_psnr = validate_low_alpha(encoder, decoder, device, test_images, epoch + 1)
            
            # Update best model
            if val_ber < best_ber:
                best_ber = val_ber
                save_checkpoint(epoch + 1, step, models, optimizer, scheduler, 
                              best_ber, args.save_dir, is_best=True)
        
        # Auto-save every N epochs
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(epoch + 1, step, models, optimizer, scheduler, 
                          best_ber, args.save_dir, is_best=False, 
                          keep_last_n=KEEP_LAST_N_CHECKPOINTS)
    
    logger.info("\\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Best BER: {best_ber:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train watermark model with production alpha range")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(args)
