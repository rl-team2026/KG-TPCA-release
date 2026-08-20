"""Mammo-CLIP backbone wrapper that provides the same interface as BiomedCLIP."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from cdf_vlm.extractors.mammoclip import MammoCLIPImageExtractor, build_mammo_preprocess


def build_mammoclip_preprocess(image_size: tuple[int, int] = (760, 456)):
    """Build Mammo-CLIP preprocess for mammogram images.

    Default 760x456 balances Mammo-CLIP's native 5:3 aspect ratio with memory constraints.
    Uses the checkpoint's normalization stats (mean=0.309, std=0.251).
    """
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3089, 0.3089, 0.3089], std=[0.2505, 0.2505, 0.2505]),
    ])


class MammoCLIPWrapper(nn.Module):
    """Wraps a Mammo-CLIP image encoder to mimic BiomedCLIP's model interface.

    Provides:
      - .encode_image(image) -> 512-dim normalised feature
      - .visual   (for LoRA compatibility; no-op on CNN backbones)
    """

    def __init__(
        self,
        checkpoint: str | Path = "external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
        source_dir: str | Path = "external/Mammo-CLIP-src/src/codebase",
        device: str | torch.device | None = None,
        input_size: tuple[int, int] = (760, 456),
    ):
        super().__init__()
        self.extractor = MammoCLIPImageExtractor(
            checkpoint=checkpoint, source_dir=source_dir, device=device,
        )
        self.input_size = input_size

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.extractor.encode(image)
