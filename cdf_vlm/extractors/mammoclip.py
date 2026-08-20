from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import transforms


def _to_plain_container(obj: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "omegaconf":
            raise RuntimeError("Install omegaconf to load Mammo-CLIP checkpoints.") from exc
        raise
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_modules(source_dir: Path) -> None:
    package_dir = source_dir / "breastclip"
    modules_dir = package_dir / "model" / "modules"
    if not modules_dir.exists():
        raise FileNotFoundError(f"Mammo-CLIP modules not found: {modules_dir}")

    breastclip_pkg = types.ModuleType("breastclip")
    breastclip_pkg.__path__ = [str(package_dir)]
    sys.modules.setdefault("breastclip", breastclip_pkg)

    model_pkg = types.ModuleType("breastclip.model")
    model_pkg.__path__ = [str(package_dir / "model")]
    sys.modules.setdefault("breastclip.model", model_pkg)

    modules_pkg = types.ModuleType("breastclip.model.modules")
    modules_pkg.__path__ = [str(modules_dir)]
    sys.modules.setdefault("breastclip.model.modules", modules_pkg)

    for mod_name, filename in [
        ("breastclip.model.modules.efficient_net_custom_utils", "efficient_net_custom_utils.py"),
        ("breastclip.model.modules.efficientnet_custom", "efficientnet_custom.py"),
        ("breastclip.model.modules.image_encoder", "image_encoder.py"),
        ("breastclip.model.modules.projection", "projection.py"),
    ]:
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, str(modules_dir / filename))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {mod_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


def build_mammo_preprocess(config: dict[str, Any]):
    base = config.get("base", {})
    height = int(base.get("image_size_h", 1520))
    width = int(base.get("image_size_w", 912))
    mean = float(base.get("mean", 0.3089279))
    std = float(base.get("std", 0.25053555408335154))
    return transforms.Compose(
        [
            transforms.Resize((height, width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )


class MammoCLIPImageExtractor:
    """Mammo-CLIP image encoder plus image projection, without text inference."""

    def __init__(
        self,
        checkpoint: str | Path = "external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
        source_dir: str | Path = "external/Mammo-CLIP-src/src/codebase",
        device: str | torch.device | None = None,
    ):
        self.checkpoint_path = Path(checkpoint)
        self.source_dir = Path(source_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = _torch_load(self.checkpoint_path)
        self.config = _to_plain_container(self.ckpt["config"])
        self.encoder, self.projection = self._load_image_modules()
        self.preprocess = build_mammo_preprocess(self.config)

    def _load_image_modules(self):
        _load_modules(self.source_dir)
        from breastclip.model.modules.efficientnet_custom import EfficientNet
        from breastclip.model.modules.image_encoder import EfficientNet_Mammo, ResNet
        from breastclip.model.modules.projection import LinearProjectionHead, MLPProjectionHead

        model_cfg = self.config["model"]
        img_cfg = model_cfg["image_encoder"]
        encoder_name = img_cfg["name"]

        if encoder_name == "tf_efficientnetv2-detect":
            encoder = EfficientNet.from_name("efficientnet-b2", num_classes=1)
            encoder.out_dim = 1408
        elif encoder_name in {"tf_efficientnet_b5_ns", "tf_efficientnetv2_s"}:
            encoder = EfficientNet_Mammo(name=encoder_name)
        else:
            encoder = ResNet(name=encoder_name, pretrained=False)

        img_state = {
            key.replace("image_encoder.", ""): value
            for key, value in self.ckpt["model"].items()
            if key.startswith("image_encoder.")
        }
        encoder.load_state_dict(img_state, strict=False)

        proj_cfg = model_cfg.get("projection_head", {})
        proj_dim = int(proj_cfg.get("proj_dim", 512))
        proj_name = str(proj_cfg.get("name", "linear")).lower()
        if "mlp" in proj_name:
            projection = MLPProjectionHead(
                encoder.out_dim,
                proj_dim,
                dropout=float(proj_cfg.get("dropout", 0.1)),
            )
        else:
            projection = LinearProjectionHead(encoder.out_dim, proj_dim)

        proj_state = {
            key.replace("image_projection.", ""): value
            for key, value in self.ckpt["model"].items()
            if key.startswith("image_projection.")
        }
        projection.load_state_dict(proj_state, strict=False)

        encoder.to(self.device).eval()
        projection.to(self.device).eval()
        return encoder, projection

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats = self.encoder(images)
        feats = self.projection(feats)
        return F.normalize(feats.float(), dim=-1)
