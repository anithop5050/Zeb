"""
Adversarial poisoning layer.
Applies constrained perturbations (FGSM/PGD-style) in luminance-dominant regions
with texture masking to stay imperceptible. Designed to be trained jointly so the
decoder learns to see through the poison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialPoisoner(nn.Module):
    def __init__(self, eps=0.02, steps=1, step_size=None):
        """Initialize poisoner.

        Args:
            eps: L-infinity budget per pixel (overall cap).
            steps: Number of PGD iterations (1 = FGSM).
            step_size: Step size for each PGD iteration (default: eps/4).
        """
        super().__init__()
        self.eps = eps
        self.steps = steps
        self.step_size = step_size if step_size is not None else eps / 4

    @staticmethod
    def _rgb_to_ycbcr(img):
        mat = img.new_tensor(
            [[0.299, 0.587, 0.114],
             [-0.1687, -0.3313, 0.5],
             [0.5, -0.4187, -0.0813]]
        ).view(3, 3, 1, 1)
        bias = img.new_tensor([0.0, 0.5, 0.5]).view(1, 3, 1, 1)
        return F.conv2d(img, mat) + bias

    @staticmethod
    def _ycbcr_to_rgb(img):
        mat = img.new_tensor(
            [[1.0, 0.0, 1.402],
             [1.0, -0.344136, -0.714136],
             [1.0, 1.772, 0.0]]
        ).view(3, 3, 1, 1)
        bias = img.new_tensor([0.0, -0.5, -0.5]).view(1, 3, 1, 1)
        return F.conv2d(img + bias, mat)

    @staticmethod
    def _texture_mask(img):
        """Compute texture mask using Sobel gradients.
        
        Returns a mask [B, 1, H, W] with higher values in textured regions.
        """
        C = img.shape[1]  # Number of channels (typically 3 for RGB)
        # Create Sobel kernels and replicate for each input channel
        sobel_x = img.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        sobel_y = img.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
        # Repeat kernel for each channel: [C, 1, 3, 3] for grouped convolution
        sobel_x = sobel_x.repeat(C, 1, 1, 1)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)
        # Apply grouped convolution (each channel independently)
        gx = F.conv2d(img, sobel_x, padding=1, groups=C)
        gy = F.conv2d(img, sobel_y, padding=1, groups=C)
        # Magnitude and average across channels
        # FIX: Add epsilon inside sqrt for numerical stability
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).mean(dim=1, keepdim=True)
        # FIX: Use larger epsilon and clamp denominator to prevent division instability
        denom = mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        mag = mag / denom
        return mag.clamp(0, 1)

    def forward(self, images, target_logits=None, loss_fn=None, poison_type="fgsm"):
        """Apply adversarial perturbation.

        Args:
            images: Tensor in [0,1].
            target_logits: Optional surrogate logits to target (for joint training, pass decoder preds).
            loss_fn: Optional loss used to craft perturbation. If None, use image smoothing loss.
            poison_type: "fgsm" or "pgd".
        Returns:
            poisoned images, perturbation tensor
        """
        # Don't detach - need gradients to flow to encoder during joint training
        x = images.clone()
        x.requires_grad_(True)

        # Default loss encourages changing luminance in textured zones
        if loss_fn is None:
            def loss_fn_def(z):
                ycbcr = self._rgb_to_ycbcr(z)
                y = ycbcr[:, :1]
                tex = self._texture_mask(y)
                # encourage deviation in textured regions only
                return (y * tex).mean()
            loss_fn_use = loss_fn_def
        else:
            loss_fn_use = loss_fn

        # For FGSM (1 step), use full eps; for PGD use step_size
        fgsm_step = self.eps if self.steps == 1 else self.step_size

        # Craft gradient
        # FIX: Use create_graph=False - we only need gradient values, not gradients-of-gradients
        # create_graph=True was causing second-order gradient explosion at Phase 4 activation
        if poison_type == "pgd":
            delta = torch.zeros_like(x)
            for _ in range(self.steps):
                adv = (x + delta).clamp(0, 1)
                loss = loss_fn_use(adv)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                delta = (delta + self.step_size * grad.sign()).clamp(-self.eps, self.eps)
            pert = delta.detach()
            poisoned = (x + pert).clamp(0, 1)
        else:  # fgsm - use full eps for single step
            loss = loss_fn_use(x)
            grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
            pert = (fgsm_step * grad.sign()).clamp(-self.eps, self.eps).detach()
            poisoned = (x + pert).clamp(0, 1)

        # Luminance-only emphasis: project perturbation to Y, then back to RGB
        ycbcr = self._rgb_to_ycbcr(poisoned)
        ycbcr_clean = self._rgb_to_ycbcr(images)
        ycbcr[:, 1:] = ycbcr_clean[:, 1:]
        poisoned = self._ycbcr_to_rgb(ycbcr).clamp(0, 1)

        # Texture mask to attenuate in smooth regions
        # FIX: Detach tex_mask to prevent gradient flow through Sobel computation
        tex_mask = self._texture_mask(images).detach()
        poisoned = (images + (poisoned - images) * tex_mask).clamp(0, 1)
        
        # FIX: Detach perturbation but preserve gradient flow through images
        # This allows encoder gradients via the 'images' input path
        pert_final = (poisoned - images).detach()
        
        # Apply detached perturbation to original images for clean gradient path
        poisoned_final = (images + pert_final).clamp(0, 1)

        return poisoned_final, pert_final
