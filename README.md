# Zeb: Multi-Defense Watermarking System

**Zeb** is a production-ready, deep learning-based invisible watermarking system designed for high-fidelity image authentication and ownership protection. It employs a multi-layered defense strategy—combining pixel-level embedding, semantic-geometric warping, and adversarial poisoning—to ensure watermarks survive aggressive image manipulations while remaining imperceptible to the human eye.

---

## 🚀 Key Features

* **Imperceptible Fidelity:** Maintains a **PSNR > 42dB**, ensuring no visible quality loss during the embedding process.
* **Multi-Layered Defense:** Integrates three distinct protection layers: pixel-level (U-Net), semantic-level (geometric flow), and adversarial-level (protective perturbations).
* **HVS-Aware Embedding:** Utilizes a Human Visual System (HVS) model to mask watermarks in textured or high-contrast regions where they are least visible.
* **Attack Resilience:** Specifically hardened against JPEG compression ($Q \geq 50$), Gaussian noise, blur, and generative AI artifacts.
* **Ownership Registry:** A thread-safe, SQLite-backed database system to track seeds, owners, and license types with full audit logging.
* **Reliability Framework:** Includes a specialized "Round-Trip Hardening" suite (R1.1–R1.8) to prevent watermark degradation during save/reload cycles.

---

## 🏗️ System Architecture

The system is built on a modular deep learning pipeline:

| Component | Functionality |
| :--- | :--- |
| **Encoder (U-Net)** | A 4x downsampling/upsampling architecture that fuses 64-bit watermarks into image bottlenecks. |
| **Semantic Layer** | Uses Sobel edge detection and flow fields to apply tiny, bit-encoded geometric warps. |
| **Adversarial Poisoner** | Applies FGSM/PGD perturbations to the luminance channel to "poison" unauthorized removal attempts. |
| **DiffJPEG** | A differentiable JPEG simulation layer allowing the model to "see" and learn from compression artifacts during training. |

---

## 🛠️ Installation

Ensure you have Python 3.8+ and a CUDA-capable GPU for optimal performance.

```bash
# Clone the repository
git clone https://github.com/anithop5050/Zeb.git
cd Zeb

# Install core dependencies
pip install torch torchvision numpy pillow scikit-image

# Optional: Install LPIPS for advanced perceptual metrics
pip install lpips
```

---

## 📖 Usage Guide

### 0. Launching the GUI (Watermark Studio)
Access the dark-themed interface for interactive embedding, extraction, and attack simulation You can use this for Embedding ,Extracting & verifying .
```bash
python app.py
```
*Note: This interface provides real-time confidence/BER display and X-ray debug visualization panels.*

### 1. Embedding a Watermark
You can embed a 64-bit watermark (defined by a seed) into any image. The system uses tile-based processing to handle images of any resolution.

```bash
python src/inference.py --mode embed --input image.jpg --output protected.png --seed 123456 --alpha 0.05
```

### 2. Extracting & Verifying
Extraction does not require the original image. The multi-scale decoder handles varied resolutions automatically.

```bash
python src/inference.py --mode extract --input protected.png
```

### 3. Registry Management
Manage the ownership database via the CLI tool.

```bash
# Register a new owner
python tools/registry_cli.py register --name "Alice" --email "alice@example.com" --license exclusive

# Look up a seed
python tools/registry_cli.py lookup --seed 123456
```

---

## 🛡️ The Reliability Framework (R1-R8)

To ensure the system is "production-ready," Zeb implements eight core reliability protocols:

* **R1.1:** Forced lossless saving (automatic `.png` conversion).
* **R1.3:** **Quantization-Aware Embedding**, which simulates the `uint8` save/reload cycle and boosts signal strength if the watermark is lost.
* **R1.5:** **Feathered Tile Blending** to eliminate visible seams when processing large images in 256x256 blocks.
* **R1.7:** **Bit Redundancy** (64 $\to$ 192 bits) using majority voting to increase extraction confidence.

---

## 📊 Performance Metrics

Based on Phase 4 curriculum training, the system achieves the following Bit Error Rates (BER):

| Attack Scenario | Avg. BER | Status |
| :--- | :--- | :--- |
| **No Attack** | < 0.01 | ✅ Robust |
| **JPEG ($Q=70$)** | 0.08 | ✅ Robust |
| **Gaussian Noise ($\sigma=0.02$)** | 0.08 | ✅ Robust |
| **Geometry (5° Rotation)** | 0.10 | ✅ Robust |
| **Social Media Resize** | 0.20 | ⚠️ Partial |

---

## 📂 Project Structure

```text
Zeb/
├── src/                # Core logic: models, semantic layers, registry
├── tools/              # Training scripts (Colab), attack simulators, CLI
├── tests/              # Comprehensive unit test suite (25+ tests)
├── checkpoints/        # Model weights (e.g., checkpoint_hvs_best.pth)
├── reliability.py      # Implementation of R1-R8 protocols
└── seed_registry.db    # SQLite database for ownership tracking
```


