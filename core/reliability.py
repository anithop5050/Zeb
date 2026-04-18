"""
Reliability Framework for Robust Watermarking (R1.1-R1.8)
==========================================================
This module implements production-ready reliability features to ensure
watermarks survive real-world compression, quantization, and attacks.

Features:
- R1.1: Path validation & save integrity
- R1.2: Alpha clamping with warnings
- R1.3: Quantization-aware embedding (CRITICAL for JPEG/social media)
- R1.4: Post-embed verification
- R1.5: Redundant bit encoding
- R1.6: Majority-vote decoding
- R1.7: BER computation & logging
- R1.8: Context logging for debugging
"""

import os
import logging
import numpy as np
from PIL import Image
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("watermark.reliability")

# Alpha constraints - must match training/inference range alignment
ALPHA_FLOOR = 0.020  # Minimum for reliable extraction with HVS model
ALPHA_CEIL = 0.055   # Maximum for invisibility (PSNR > 40dB)

# Quantization-aware parameters
JPEG_QUALITY_THRESHOLD = 85  # Below this, watermark may degrade
PNG_BIT_DEPTH = 8  # Standard PNG uses 8-bit per channel


# ==============================================================================
# R1.1: Path Validation & Save Integrity
# ==============================================================================

def validate_save_path(output_path, auto_fix=True):
    """Validate output path for lossless/high-quality format.
    
    Args:
        output_path: Requested output file path
        auto_fix: If True, automatically fix extension to lossless format
    
    Returns:
        validated_path: Path with validated/fixed extension
    
    Raises:
        ValueError: If path has lossy extension and auto_fix=False
    """
    if not output_path:
        raise ValueError("Output path cannot be empty")
    
    # Get extension
    _, ext = os.path.splitext(output_path)
    ext_lower = ext.lower()
    
    # Lossy formats that may degrade watermark
    lossy_formats = {'.jpg', '.jpeg'}
    
    # Recommended lossless formats
    lossless_formats = {'.png', '.tiff', '.tif', '.bmp', '.webp'}
    
    if ext_lower in lossy_formats:
        if auto_fix:
            # Replace with PNG
            new_path = output_path.rsplit('.', 1)[0] + '.png'
            logger.warning(f"R1.1: Lossy format {ext} → PNG for integrity: {new_path}")
            return new_path
        else:
            raise ValueError(
                f"Lossy format {ext} may degrade watermark. "
                f"Use lossless format: {lossless_formats}"
            )
    
    if ext_lower not in lossless_formats:
        # Unknown extension - default to PNG
        if auto_fix:
            new_path = output_path + '.png'
            logger.warning(f"R1.1: Unknown format → PNG: {new_path}")
            return new_path
        else:
            logger.warning(f"R1.1: Unknown format {ext}, may cause issues")
    
    return output_path


# ==============================================================================
# R1.2: Alpha Clamping with Warnings
# ==============================================================================

def clamp_alpha(alpha, warn=True):
    """Clamp alpha to safe range for reliable extraction.
    
    Args:
        alpha: Requested watermark strength
        warn: Log warning if clamping occurs
    
    Returns:
        clamped_alpha: Alpha value within [ALPHA_FLOOR, ALPHA_CEIL]
    """
    if alpha < ALPHA_FLOOR:
        if warn:
            logger.warning(
                f"R1.2: Alpha {alpha:.4f} < floor {ALPHA_FLOOR:.4f}. "
                f"Clamping to {ALPHA_FLOOR:.4f} for reliable extraction."
            )
        return ALPHA_FLOOR
    
    if alpha > ALPHA_CEIL:
        if warn:
            logger.warning(
                f"R1.2: Alpha {alpha:.4f} > ceiling {ALPHA_CEIL:.4f}. "
                f"Clamping to {ALPHA_CEIL:.4f} for invisibility."
            )
        return ALPHA_CEIL
    
    return alpha


# ==============================================================================
# R1.3: Quantization-Aware Embedding (CRITICAL)
# ==============================================================================

def apply_quantization_simulation(img_tensor, bit_depth=8):
    """Simulate quantization that occurs during JPEG/PNG saving.
    
    This is CRITICAL for compression robustness. By pre-applying quantization
    during embedding, we ensure the watermark survives real-world compression.
    
    Args:
        img_tensor: [1, 3, H, W] float tensor in [0, 1]
        bit_depth: Bit depth for quantization (default 8 for standard images)
    
    Returns:
        quantized: Tensor after quantization round-trip
    """
    levels = 2 ** bit_depth
    quantized = torch.round(img_tensor * (levels - 1)) / (levels - 1)
    return quantized.clamp(0, 1)


def embed_with_quantization_awareness(watermarked_tensor, device):
    """Apply quantization-aware post-processing to watermarked image.
    
    This simulates the quantization that will occur when saving to PNG/JPEG,
    making the watermark more robust to compression.
    
    Args:
        watermarked_tensor: [1, 3, H, W] watermarked image tensor
        device: torch device
    
    Returns:
        robust_watermarked: Quantization-aware watermarked tensor
    """
    # Simulate 8-bit quantization
    quantized = apply_quantization_simulation(watermarked_tensor, bit_depth=8)
    
    logger.debug(
        f"R1.3: Applied quantization simulation. "
        f"Max delta: {(watermarked_tensor - quantized).abs().max():.6f}"
    )
    
    return quantized


# ==============================================================================
# R1.4: Post-Embed Verification
# ==============================================================================

def post_embed_verify(saved_path, extract_fn, expected_bits, seed, ber_threshold=0.05):
    """Verify watermark integrity after saving to disk.
    
    This catches issues with lossy saving, compression, or file format problems.
    
    Args:
        saved_path: Path to saved watermarked image
        extract_fn: Function(img_float_np) -> extracted_bits_np
        expected_bits: Expected watermark bits (numpy array)
        seed: Random seed used for watermark generation
        ber_threshold: Maximum acceptable BER (default 5%)
    
    Returns:
        ber: Bit error rate
        confidence: Extraction confidence percentage
        passed: True if verification passed
    """
    try:
        # Load saved image
        img_pil = Image.open(saved_path).convert('RGB')
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        
        # Extract watermark
        extracted_bits = extract_fn(img_np)
        
        # Compute BER
        ber = compute_ber(extracted_bits, expected_bits)
        confidence = (1 - ber) * 100
        
        passed = ber < ber_threshold
        
        if passed:
            logger.info(
                f"R1.4: Post-embed verification PASSED. "
                f"BER={ber:.4f}, Confidence={confidence:.1f}%"
            )
        else:
            logger.error(
                f"R1.4: Post-embed verification FAILED! "
                f"BER={ber:.4f} > threshold {ber_threshold:.4f}"
            )
        
        return ber, confidence, passed
        
    except Exception as e:
        logger.error(f"R1.4: Verification failed with error: {e}")
        return 1.0, 0.0, False


# ==============================================================================
# R1.5: Redundant Bit Encoding
# ==============================================================================

def expand_bits_redundant(bits, repetition_factor=3):
    """Expand bits with repetition coding for error correction.
    
    Each bit is repeated `repetition_factor` times. This trades capacity
    for robustness against bit flips.
    
    Args:
        bits: Original bits [watermark_len]
        repetition_factor: How many times to repeat each bit (default 3)
    
    Returns:
        expanded_bits: Redundant bits [watermark_len * repetition_factor]
    """
    if isinstance(bits, torch.Tensor):
        expanded = bits.repeat_interleave(repetition_factor)
    else:
        expanded = np.repeat(bits, repetition_factor)
    
    logger.debug(f"R1.5: Expanded {len(bits)} bits → {len(expanded)} with r={repetition_factor}")
    return expanded


# ==============================================================================
# R1.6: Majority-Vote Decoding
# ==============================================================================

def collapse_bits_majority(expanded_bits, repetition_factor=3):
    """Collapse redundant bits using majority voting.
    
    For each group of `repetition_factor` bits, take the majority value.
    This corrects single-bit errors within each group.
    
    Args:
        expanded_bits: Redundant bits [watermark_len * repetition_factor]
        repetition_factor: Repetition factor used in encoding
    
    Returns:
        collapsed_bits: Original bits [watermark_len] after error correction
    """
    if isinstance(expanded_bits, torch.Tensor):
        expanded_bits = expanded_bits.cpu().numpy()
    
    # Reshape into groups
    n_groups = len(expanded_bits) // repetition_factor
    groups = expanded_bits[:n_groups * repetition_factor].reshape(n_groups, repetition_factor)
    
    # Majority vote: sum > threshold
    collapsed = (groups.sum(axis=1) > (repetition_factor / 2)).astype(np.float32)
    
    logger.debug(f"R1.6: Collapsed {len(expanded_bits)} bits → {len(collapsed)} via majority vote")
    return collapsed


# ==============================================================================
# R1.7: BER Computation & Logging
# ==============================================================================

def compute_ber(extracted_bits, expected_bits):
    """Compute Bit Error Rate between extracted and expected watermarks.
    
    Args:
        extracted_bits: Extracted watermark bits
        expected_bits: Ground-truth watermark bits
    
    Returns:
        ber: Bit error rate in [0, 1]
    """
    if isinstance(extracted_bits, torch.Tensor):
        extracted_bits = extracted_bits.cpu().numpy()
    if isinstance(expected_bits, torch.Tensor):
        expected_bits = expected_bits.cpu().numpy()
    
    # Flatten arrays
    extracted_flat = extracted_bits.flatten()
    expected_flat = expected_bits.flatten()
    
    # Compute errors
    errors = np.sum(extracted_flat != expected_flat)
    total = len(expected_flat)
    
    ber = errors / total if total > 0 else 1.0
    
    return ber


# ==============================================================================
# R1.8: Context Logging for Debugging
# ==============================================================================

def log_embed_context(image_path, alpha, seed, hvs_enabled, semantic_enabled, 
                     poison_enabled, output_path):
    """Log embedding context for debugging and audit trail.
    
    Args:
        image_path: Input image path
        alpha: Watermark strength used
        seed: Random seed
        hvs_enabled: Whether HVS masking was used
        semantic_enabled: Whether semantic watermark was used
        poison_enabled: Whether adversarial poisoning was used
        output_path: Where watermarked image was saved
    """
    logger.info("=" * 60)
    logger.info("R1.8: EMBEDDING CONTEXT")
    logger.info("=" * 60)
    logger.info(f"Input:     {image_path}")
    logger.info(f"Output:    {output_path}")
    logger.info(f"Alpha:     {alpha:.4f}")
    logger.info(f"Seed:      {seed}")
    logger.info(f"HVS:       {'ON' if hvs_enabled else 'OFF'}")
    logger.info(f"Semantic:  {'ON' if semantic_enabled else 'OFF'}")
    logger.info(f"Poison:    {'ON' if poison_enabled else 'OFF'}")
    logger.info("=" * 60)


def log_extract_context(image_path, seed, ber, confidence, expected_bits=None, 
                        extracted_bits=None):
    """Log extraction context for debugging.
    
    Args:
        image_path: Input watermarked image path
        seed: Random seed used
        ber: Bit error rate
        confidence: Extraction confidence percentage
        expected_bits: Ground-truth bits (if known)
        extracted_bits: Extracted bits
    """
    logger.info("=" * 60)
    logger.info("R1.8: EXTRACTION CONTEXT")
    logger.info("=" * 60)
    logger.info(f"Input:      {image_path}")
    logger.info(f"Seed:       {seed}")
    logger.info(f"BER:        {ber:.4f}")
    logger.info(f"Confidence: {confidence:.1f}%")
    
    if expected_bits is not None and extracted_bits is not None:
        if isinstance(expected_bits, torch.Tensor):
            expected_bits = expected_bits.cpu().numpy()
        if isinstance(extracted_bits, torch.Tensor):
            extracted_bits = extracted_bits.cpu().numpy()
        
        # Show first 16 bits for debugging
        exp_str = ''.join(map(str, expected_bits.flatten()[:16].astype(int)))
        ext_str = ''.join(map(str, extracted_bits.flatten()[:16].astype(int)))
        
        logger.info(f"Expected:   {exp_str}...")
        logger.info(f"Extracted:  {ext_str}...")
    
    logger.info("=" * 60)


# ==============================================================================
# Load Image with Exact Dimensions (R1.1 extension)
# ==============================================================================

def load_image_exact(image_path):
    """Load image at exact dimensions without resizing.
    
    This ensures no quality loss from unnecessary resizing operations.
    
    Args:
        image_path: Path to image file
    
    Returns:
        img_float: Numpy array [H, W, 3] in float32 [0, 1]
        original_size: Tuple (H, W)
    """
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    original_size = img_np.shape[:2]
    
    logger.debug(f"R1.1: Loaded image at exact size {original_size}")
    
    return img_np, original_size


# ==============================================================================
# Module Initialization
# ==============================================================================

logger.info("Reliability framework (R1.1-R1.8) loaded successfully")
logger.info(f"Alpha constraints: [{ALPHA_FLOOR:.4f}, {ALPHA_CEIL:.4f}]")
