"""Augmentation for the long-tail strategy (section 1.3).

Photometric/geometric augmentation simulating natural undergrowth lighting is
implemented with pure PIL (no torchvision dependency). Mixup and CutMix operate
on batches of tensors and blend targets, used for rare classes during training.
"""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class TrainTransform:
    """Random augmentation pipeline: colour jitter, rotation, blur, flips.

    Simulates varying natural light in undergrowth and field-of-view geometry.
    """

    def __init__(
        self,
        image_size: int = 224,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue_shift: float = 0.1,
        rotation_deg: float = 20.0,
        blur_prob: float = 0.2,
        flip_prob: float = 0.5,
    ):
        self.image_size = image_size
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_shift = hue_shift
        self.rotation_deg = rotation_deg
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        # Geometric: rotation + upscale-crop + optional flip.
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        if random.random() < self.flip_prob:
            img = ImageOps.mirror(img)
        img = _resize_center_crop(img, self.image_size)
        # Photometric jitter (natural light variation).
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(1 - self.brightness, 1 + self.brightness))
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(1 - self.contrast, 1 + self.contrast))
        img = ImageEnhance.Color(img).enhance(
            random.uniform(1 - self.saturation, 1 + self.saturation))
        img = _shift_hue(img, self.hue_shift)
        if random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        return _pil_to_tensor(img)


class EvalTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return _pil_to_tensor(_resize_center_crop(img.convert("RGB"), self.image_size))


def train_transform(image_size: int = 224, **kwargs) -> TrainTransform:
    return TrainTransform(image_size=image_size, **kwargs)


def eval_transform(image_size: int = 224) -> EvalTransform:
    return EvalTransform(image_size=image_size)


# ---------------------------------------------------------------- helpers

def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _shift_hue(img: Image.Image, shift: float) -> Image.Image:
    if shift <= 0:
        return img
    import numpy as np

    hsv = np.asarray(img.convert("HSV"), dtype="int16").copy()
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-shift, shift) * 255) % 256
    return Image.fromarray(hsv.astype("uint8"), mode="HSV").convert("RGB")


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ------------------------------------------------------- Mixup / CutMix

def mixup_cutmix_batch(
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    alpha: float = 0.4,
    mixup_prob: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    """Blend a batch with a shuffled copy (Mixup) or a pasted rectangle (CutMix).

    Returns (augmented_images, targets, lam). Targets are returned unchanged;
    the caller should compute loss as ``lam*CE(out, y_a) + (1-lam)*CE(out, y_b)``
    for each level (y_a = original, y_b = permuted).
    """
    if alpha <= 0:
        return images, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    y_b = {k: v[idx] for k, v in targets.items()}
    if torch.rand(()) < mixup_prob:
        # Mixup: global blend.
        out = lam * images + (1 - lam) * images[idx]
    else:
        # CutMix: paste a random rectangle from the shuffled batch.
        out = images.clone()
        _, _, H, W = images.shape
        cut_rat = (1 - lam) ** 0.5
        rw, rh = int(W * cut_rat), int(H * cut_rat)
        rx, ry = random.randint(0, W - rw), random.randint(0, H - rh)
        out[:, :, ry:ry + rh, rx:rx + rw] = images[idx, :, ry:ry + rh, rx:rx + rw]
        lam = 1.0 - (rh * rw) / (W * H)  # exact area ratio
    return out, y_b, lam
