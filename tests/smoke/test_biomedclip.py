"""
测试 BiomedCLIP 模型：本地加载 + 零样本图像分类
模型: BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (Microsoft)
论文: NEJM AI 2024, https://aka.ms/biomedclip-paper
"""

import json
import sys
from pathlib import Path

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from PIL import Image

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "external" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONFIG_PATH = MODEL_DIR / "open_clip_config.json"
WEIGHTS_PATH = MODEL_DIR / "open_clip_pytorch_model.bin"
EXAMPLE_DIR = MODEL_DIR / "example_data" / "biomed_image_classification_example_data"

# === 零样本分类标签 ===
LABELS = [
    "adenocarcinoma histopathology",
    "brain MRI",
    "covid line chart",
    "squamous cell carcinoma histopathology",
    "immunohistochemistry histopathology",
    "bone X-ray",
    "chest X-ray",
    "pie chart",
    "hematoxylin and eosin histopathology",
]

# === 可用的测试图片 ===
TEST_IMAGES = [
    "squamous_cell_carcinoma_histopathology.jpeg",
    "H_and_E_histopathology.jpg",
    "bone_X-ray.jpg",
    "adenocarcinoma_histopathology.jpg",
    "covid_line_chart.png",
    "IHC_histopathology.jpg",
    "chest_X-ray.jpg",
    "brain_MRI.jpg",
    "pie_chart.png",
]


def load_model(model_dir: Path):
    """从本地文件加载 BiomedCLIP 模型（完全离线，不访问 HuggingFace）"""
    print(f"[1/4] 读取配置: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"       模型架构: ViT-Base/16-224 + PubMedBERT")
    print(f"       嵌入维度: {config['model_cfg']['embed_dim']}")
    print(f"       文本上下文长度: {config['model_cfg']['text_cfg']['context_length']}")

    # 关键：将 hf_model_name 改为本地 PubMedBERT 配置目录
    # 避免 open_clip 去 HuggingFace 下载 config.json
    local_bert_config = model_dir / "pubmedbert_config"
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(local_bert_config)
    config["model_cfg"]["text_cfg"]["hf_model_pretrained"] = False
    # 同时修改 tokenizer 路径使用本地文件
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(model_dir)

    model_name = "biomedclip_local"
    _MODEL_CONFIGS[model_name] = config["model_cfg"]

    print(f"[2/4] 加载权重: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1e9:.2f} GB)")

    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(WEIGHTS_PATH),
        **{f"image_{k}": v for k, v in config["preprocess_cfg"].items()},
    )
    tokenizer = get_tokenizer(model_name)

    print(f"[3/4] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, preprocess, tokenizer


def run_zero_shot(model, preprocess, tokenizer, device, top_k: int = 3):
    """在示例图片上运行零样本分类"""
    print(f"[4/4] 运行零样本分类 (top-{top_k})...\n")

    template = "this is a photo of "
    prompts = [template + label for label in LABELS]

    model = model.to(device).eval()

    for img_name in TEST_IMAGES:
        img_path = EXAMPLE_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] 图片不存在: {img_path}")
            continue

        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        text = tokenizer(prompts, context_length=256).to(device)

        with torch.no_grad():
            image_features, text_features, logit_scale = model(image, text)
            probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)

        scores, indices = probs[0].topk(min(top_k, len(LABELS)))

        print(f"  {img_name}:")
        for score, idx in zip(scores.tolist(), indices.tolist()):
            marker = " <== TOP-1" if idx == indices[0] else ""
            print(f"    {LABELS[idx]}: {score:.4f}{marker}")
        print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}\n")

    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] 模型文件不存在: {WEIGHTS_PATH}")
        print("请确保已通过 git-lfs 下载模型权重文件")
        sys.exit(1)

    model, preprocess, tokenizer = load_model(MODEL_DIR)
    run_zero_shot(model, preprocess, tokenizer, device)


if __name__ == "__main__":
    main()
