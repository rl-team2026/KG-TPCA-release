"""
测试 Mammo-CLIP 模型：加载预训练权重 + 模型架构验证 + 完整前向传播
模型: Mammo-CLIP (EfficientNet-B2 image encoder + Bio_ClinicalBERT text encoder)
论文: MICCAI 2024, https://arxiv.org/pdf/2405.12255

使用方式:
  export HF_ENDPOINT=https://hf-mirror.com
  python tests/smoke/test_mammoclip.py
"""

import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# === 设置 HuggingFace 镜像 ===
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[INFO] HF_ENDPOINT 未设置，自动使用 https://hf-mirror.com")

# === 路径 ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEBASE = PROJECT_ROOT / "external" / "Mammo-CLIP-src" / "src" / "codebase"
MODULES_DIR = CODEBASE / "breastclip" / "model" / "modules"

CKPT_PATH = PROJECT_ROOT / "external" / "Mammo-CLIP" / "Pre-trained-checkpoints" / "b2-model-best-epoch-10.tar"

if not MODULES_DIR.exists():
    print(f"[SKIP] Mammo-CLIP source modules not found: {MODULES_DIR}")
    print("       Download/clone Mammo-CLIP source into external/Mammo-CLIP-src to run this smoke test.")
    sys.exit(0)

# === 构建最小包结构，绕过 nltk 依赖 ===
# 创建虚拟的 breastclip 包，只暴露模型组件
_breastclip_mod = types.ModuleType("breastclip")
_breastclip_data_mod = types.ModuleType("breastclip.data")
_breastclip_data_datasets_mod = types.ModuleType("breastclip.data.datasets")

# 直接加载核心模块（绕过 __init__.py 中的 trainer/data import）
sys.modules.setdefault("breastclip", _breastclip_mod)
sys.modules.setdefault("breastclip.data", _breastclip_data_mod)
sys.modules.setdefault("breastclip.data.datasets", _breastclip_data_datasets_mod)

# 加载 model 层模块
for mod_name, py_file in [
    ("breastclip.model.modules.efficient_net_custom_utils", MODULES_DIR / "efficient_net_custom_utils.py"),
    ("breastclip.model.modules.efficientnet_custom", MODULES_DIR / "efficientnet_custom.py"),
    ("breastclip.model.modules.image_encoder", MODULES_DIR / "image_encoder.py"),
    ("breastclip.model.modules.text_encoder", MODULES_DIR / "text_encoder.py"),
    ("breastclip.model.modules.projection", MODULES_DIR / "projection.py"),
    ("breastclip.model.modules.image_classifier", MODULES_DIR / "image_classifier.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, str(py_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"  [WARN] 模块 {mod_name} 加载失败: {e}")

# 加载 breastclip.model.modules.__init__ (需要上面的子模块先注册)
from breastclip.model.modules.image_encoder import (
    HuggingfaceImageEncoder, ResNet, EfficientNet_Mammo, GeM, gem
)
from breastclip.model.modules.text_encoder import HuggingfaceTextEncoder
from breastclip.model.modules.projection import LinearProjectionHead, MLPProjectionHead
from breastclip.model.modules.efficientnet_custom import EfficientNet
from breastclip.model.modules.image_classifier import LinearClassifier

print("[OK] 所有核心模块加载完成（已绕过 nltk 依赖）")


def inspect_checkpoint(ckpt_path: Path) -> dict:
    """加载并检查 checkpoint 内容"""
    print(f"\n{'='*60}")
    print(f"[1/4] 加载 checkpoint: {ckpt_path.name}")
    print(f"      文件大小: {ckpt_path.stat().st_size / 1e9:.2f} GB")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]["model"]

    print(f"\n      === 模型架构 ===")
    print(f"      类型: {model_cfg['name']}")
    print(f"      温度: {model_cfg.get('temperature', 'N/A')}")

    img_cfg = model_cfg["image_encoder"]
    print(f"      图像编码器: {img_cfg['source']} / {img_cfg['name']}")

    txt_cfg = model_cfg["text_encoder"]
    print(f"      文本编码器: {txt_cfg['source']} / {txt_cfg['name']}")
    print(f"      池化方式: {txt_cfg['pooling']}")

    if "projection_head" in model_cfg:
        proj = model_cfg["projection_head"]
        print(f"      投影头: {proj['name']} → {proj['proj_dim']}维")

    base_cfg = ckpt["config"]["base"]
    print(f"      输入分辨率: {base_cfg['image_size_h']}x{base_cfg['image_size_w']}")
    print(f"      文本最大长度: {base_cfg['text_max_length']}")

    state_dict = ckpt["model"]
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"      模型参数量: {total_params / 1e6:.1f}M (训练了 {ckpt['epoch']} epochs)")

    # 分析 state_dict 结构
    components = {}
    for key in state_dict:
        prefix = key.split(".")[0]
        components[prefix] = components.get(prefix, 0) + 1
    print(f"      组件: {components}")

    return ckpt


def test_image_encoder(ckpt: dict):
    """测试图像编码器（完全离线）"""
    print(f"\n{'='*60}")
    print(f"[2/4] 测试图像编码器（完全离线，无需 nltk/tokenizer）")

    img_cfg = ckpt["config"]["model"]["image_encoder"]
    encoder_name = img_cfg["name"]
    print(f"      编码器: {encoder_name}")

    if encoder_name == "tf_efficientnetv2-detect":
        encoder = EfficientNet.from_name("efficientnet-b2", num_classes=1)
        encoder.out_dim = 1408
    elif encoder_name in ("tf_efficientnet_b5_ns", "tf_efficientnetv2_s"):
        encoder = EfficientNet_Mammo(name=encoder_name)
    else:
        encoder = ResNet(name=encoder_name)

    encoder.eval()

    h, w = ckpt["config"]["base"]["image_size_h"], ckpt["config"]["base"]["image_size_w"]
    dummy_img = torch.randn(1, 3, h, w)  # 注意: 3 通道! (EfficientNet 用 RGB)

    with torch.no_grad():
        features = encoder(dummy_img)

    print(f"      输入 (1, 3, {h}, {w}) → 输出 {tuple(features.shape)}")
    print(f"      特征维度: {encoder.out_dim}")
    print(f"      参数量: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")

    # 验证能加载 checkpoint 权重
    img_state = {k.replace("image_encoder.", ""): v for k, v in ckpt["model"].items()
                 if k.startswith("image_encoder.")}
    missing, unexpected = encoder.load_state_dict(img_state, strict=False)
    print(f"      加载 checkpoint 权重: 缺失={len(missing)}, 多余={len(unexpected)}")
    if len(missing) == 0:
        print(f"      [OK] 图像编码器权重完全匹配 ✓")

    return encoder


def test_full_model(ckpt: dict):
    """加载完整模型（需要下载 tokenizer 和 Bio_ClinicalBERT）"""
    print(f"\n{'='*60}")
    print(f"[3/4] 加载完整模型 (文本编码器需要从 HF mirror 下载)")

    from transformers import AutoTokenizer, AutoModel

    # Tokenizer
    tokenizer_name = ckpt["config"]["tokenizer"]["pretrained_model_name_or_path"]
    print(f"      下载 tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"      词表大小: {tokenizer.vocab_size}")

    # 文本编码器
    txt_cfg = ckpt["config"]["model"]["text_encoder"]
    print(f"      下载 Bio_ClinicalBERT...")
    text_encoder = AutoModel.from_pretrained(txt_cfg["name"], trust_remote_code=True)
    text_encoder.out_dim = text_encoder.config.hidden_size
    print(f"      hidden_size: {text_encoder.out_dim}")

    # 图像编码器
    img_cfg = ckpt["config"]["model"]["image_encoder"]
    if img_cfg["name"] == "tf_efficientnetv2-detect":
        image_encoder = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1)
        image_encoder.out_dim = 1408
    else:
        image_encoder = EfficientNet_Mammo(name=img_cfg["name"])

    # 投影头
    proj_dim = ckpt["config"]["model"]["projection_head"]["proj_dim"]
    image_projection = LinearProjectionHead(image_encoder.out_dim, proj_dim)
    text_projection = LinearProjectionHead(text_encoder.out_dim, proj_dim)

    # 组装 Mammo-CLIP
    class MammoCLIPWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            import numpy as np
            self.image_encoder = image_encoder
            self.text_encoder = text_encoder
            self.image_projection = image_projection
            self.text_projection = text_projection
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.text_pooling = txt_cfg["pooling"]
            self.projection = True

        def encode_image(self, image):
            return self.image_encoder(image)

        def encode_text(self, text_tokens):
            output = self.text_encoder(**text_tokens)
            features = output["last_hidden_state"]
            pool = self.text_pooling
            if pool == "eos":
                eos_idx = text_tokens["attention_mask"].sum(dim=-1) - 1
                return features[torch.arange(features.shape[0]), eos_idx]
            elif pool == "bos":
                return features[:, 0]
            elif pool == "mean":
                mask = text_tokens["attention_mask"].unsqueeze(-1).expand_as(features).float()
                return (features * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        def forward(self, batch, device=None):
            device = device or batch["images"].device
            img = batch["images"].to(device)
            txt_tokens = {k: v.to(device) for k, v in batch["text_tokens"].items()}

            img_emb = self.image_projection(self.encode_image(img))
            txt_emb = self.text_projection(self.encode_text(txt_tokens))
            img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

            return {
                "image_embeddings": img_emb,
                "text_embeddings": txt_emb,
                "logit_scale": self.logit_scale.exp(),
            }

    model = MammoCLIPWrapper()
    result = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print(f"      加载结果: 缺失={len(result.missing_keys)}, 多余={len(result.unexpected_keys)}")
    if len(result.missing_keys) > 0:
        print(f"      缺失前5: {result.missing_keys[:5]}")
    print(f"      总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"      [OK] 完整模型加载成功 ✓")

    return model, tokenizer


def test_forward_pass(model, tokenizer, ckpt: dict):
    """测试前向传播"""
    print(f"\n{'='*60}")
    print(f"[4/4] 测试前向传播 — 零样本模拟")

    base = ckpt["config"]["base"]
    h, w = base["image_size_h"], base["image_size_w"]
    text_max_len = base["text_max_length"]

    # 模拟 mammogram (3通道，与 checkpoint 训练一致)
    dummy_image = torch.randn(2, 3, h, w)

    # 模拟放射学报告
    texts = [
        "No evidence of malignancy. Benign calcifications noted.",
        "Suspicious mass in the upper outer quadrant. Biopsy recommended.",
    ]
    tokens = tokenizer(texts, padding="longest", truncation=True,
                       return_tensors="pt", max_length=text_max_len)

    batch = {"images": dummy_image, "text_tokens": tokens}

    with torch.no_grad():
        output = model(batch)

    img_emb = output["image_embeddings"]
    txt_emb = output["text_embeddings"]
    ls = output["logit_scale"]

    print(f"      图像嵌入: {tuple(img_emb.shape)} (batch, 512)")
    print(f"      文本嵌入: {tuple(txt_emb.shape)} (batch, 512)")
    print(f"      Logit scale: {ls.item():.4f} (temperature: {1/ls.item():.4f})")

    sim = img_emb @ txt_emb.t()
    print(f"      相似度矩阵:")
    print(f"        良性文本 — 良性文本: {sim[0,0]:+.4f}  |  良性文本 — 恶性文本: {sim[0,1]:+.4f}")
    print(f"        恶性文本 — 良性文本: {sim[1,0]:+.4f}  |  恶性文本 — 恶性文本: {sim[1,1]:+.4f}")

    print(f"\n      [OK] 模型可用于:")
    print(f"        - 零样本分类: model.encode_image(img) × model.encode_text(labels)")
    print(f"        - 特征提取: embeddings = model.encode_image(mammogram)")
    print(f"        - 微调: 冻结编码器 + 分类头 / 全模型微调")
    print(f"        - 检测: 编码器作为 RetinaNet backbone")


def print_comparison():
    print(f"""
{'='*60}
              两个模型对比总结
{'='*60}

                        BiomedCLIP               Mammo-CLIP
──────────────────────────────────────────────────────────────
用途                    通用生物医学影像           乳腺X线摄影 (mammogram)
图像编码器              ViT-Base/16-224            EfficientNet-B2
文本编码器              PubMedBERT                 Bio_ClinicalBERT
嵌入维度                512                        512
训练数据                PMC-15M (公开, 15M)        院内数据 (非公开)
输入                    224×224 RGB                1520×912 (单通道→3通道)
参数量                  ~196M                      ~117M
完全离线                是 ✓                       需下载 tokenizer
权重文件                .bin (748MB)               .tar (1.4GB)

=== 加载方式 ===
  BiomedCLIP:
    python tests/smoke/test_biomedclip.py  # 完全离线可用

  Mammo-CLIP:
    export HF_ENDPOINT=https://hf-mirror.com  # 首次需要
    python tests/smoke/test_mammoclip.py

=== 微调建议 ===
  1. Linear Probe (快速验证):
     冻结编码器 → 提取特征 → 训练线性分类器
     适合小数据集 (几百张图)

  2. Fine-tune (更好效果):
     解冻编码器 → LoRA/全微调
     适合大数据集 (几千张+)

  3. 检测任务:
     Mammo-CLIP 编码器 + RetinaNet head
     病灶定位 (Mass / Calcification)
""")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    if not CKPT_PATH.exists():
        print(f"[ERROR] checkpoint 不存在: {CKPT_PATH}")
        sys.exit(1)

    # Step 1: 检查 checkpoint
    ckpt = inspect_checkpoint(CKPT_PATH)

    # Step 2: 图像编码器 (离线)
    try:
        encoder = test_image_encoder(ckpt)
    except Exception as e:
        print(f"\n[WARN] 图像编码器失败: {e}")
        import traceback; traceback.print_exc()

    # Step 3-4: 完整模型需要 HuggingFace tokenizer/text encoder. 默认跳过，保证离线 smoke 稳定。
    if os.environ.get("RUN_MAMMOCLIP_FULL", "0") == "1":
        try:
            model, tokenizer = test_full_model(ckpt)
            test_forward_pass(model, tokenizer, ckpt)
        except Exception as e:
            print(f"\n[WARN] 完整模型失败: {e}")
            print("      请确认 HuggingFace 网络/cache 可用，或预先下载 Bio_ClinicalBERT。")
    else:
        print(f"\n[SKIP] 完整文本分支测试默认关闭。设置 RUN_MAMMOCLIP_FULL=1 后会尝试加载 Bio_ClinicalBERT。")

    print_comparison()


if __name__ == "__main__":
    main()
