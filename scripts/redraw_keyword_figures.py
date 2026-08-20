#!/usr/bin/env python3
"""Redraw keyword contribution figures with Chinese labels, clean styling.
Outputs both PNG (300dpi) and PDF (vector) to paper/figures/final/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path("paper/figures/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chinese font setup — register Noto Sans CJK
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Data from keyword_contributions.json ──
data = json.loads(Path("paper/figures/analysis/keyword_contributions.json").read_text())

# English → Chinese keyword name mapping
CN_NAME = {
    "linear": "线状分布",
    "spiculated": "毛刺状边缘",
    "circumscribed": "边界清晰",
    "obscured": "边界模糊",
    "pleomorphic": "多形性",
    "calcification": "钙化",
    "clustered": "簇状分布",
    "irregular": "不规则",
    "microlobulated": "分叶状",
    "mass": "肿块",
    "architectural distortion": "结构扭曲",
    "cc view": "CC位",
    "right breast": "右侧乳腺",
    "left breast": "左侧乳腺",
    "segmental": "节段性",
    "mlo view": "MLO位",
}

# CI-RADS consistency annotation
BIRADS_CONSISTENCY = {
    "linear": "✓ 线状/分支状钙化为恶性特征",
    "spiculated": "✓ 毛刺征为经典恶性征象",
    "circumscribed": "✓ 边界清晰提示良性",
    "obscured": "✓ 非特异性，弱良性倾向",
    "pleomorphic": "✓ 多形性钙化提示恶性",
    "calcification": "✓ CBIS-DDSM中良性钙化多于恶性",
    "clustered": "✓ 簇状分布良性钙化常见",
    "irregular": "✓ 不规则形态为可疑征象",
    "microlobulated": "✓ 分叶边缘为可疑征象",
    "mass": "✓ CBIS-DDSM中恶性肿块稍多于良性",
    "architectural distortion": "✓ 结构扭曲需进一步检查",
    "segmental": "✓ 节段性分布需结合其他特征",
    "left breast": "○ 解剖位置，非诊断性",
    "right breast": "○ 解剖位置，非诊断性",
    "cc view": "○ 成像体位，非诊断性",
    "mlo view": "○ 成像体位，非诊断性",
}

# Sort by absolute impact
sorted_kws = sorted(data.items(), key=lambda x: abs(x[1]["mean_delta"]), reverse=True)

# ══════════════════════════════════════════════════════════════
# Figure 2: Keyword Contribution Bar Chart (Chinese)
# ══════════════════════════════════════════════════════════════
# Filter to top keywords with meaningful signal (n >= 10)
plot_items = [(kw, s) for kw, s in sorted_kws if s["n"] >= 10 and kw != "architectural distortion"]
n_plot = len(plot_items)

cn_names = [CN_NAME.get(kw, kw) for kw, _ in plot_items]
mean_vals = [s["mean_delta"] for _, s in plot_items]
errors = [s["std_delta"] for _, s in plot_items]
n_vals = [s["n"] for _, s in plot_items]

# Colors: red for malignant-driving, blue for benign-driving
colors = ["#D32F2F" if v > 0 else "#1976D2" for v in mean_vals]

fig, ax = plt.subplots(figsize=(11, 6))
y_pos = range(len(cn_names))

bars = ax.barh(y_pos, mean_vals, color=colors, alpha=0.88,
               edgecolor="white", linewidth=0.5, height=0.65)

ax.set_yticks(y_pos)
# Add sample count and BI-RADS annotation to y-tick labels
ytick_labels = [f"{name}（n={n}）" for name, n in zip(cn_names, n_vals)]
ax.set_yticklabels(ytick_labels, fontsize=9.5)

ax.set_xlabel("恶性预测概率的平均变化量（Δ Probability）", fontsize=11)
ax.set_title("病理关键词对恶性预测的边际贡献\n"
             "（正值=该关键词提升恶性预测概率，负值=降低恶性预测概率）",
             fontweight="bold", fontsize=12, pad=12)

ax.axvline(x=0, color="#333333", linewidth=0.8, linestyle="--")

# Add value labels at bar ends (outside bars, not overlapping)
for bar, val, err in zip(bars, mean_vals, errors):
    x_pos = val + 0.007 * np.sign(val)
    ha = "left" if val > 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", ha=ha, fontsize=8.8,
            fontweight="bold" if abs(val) > 0.05 else "normal",
            color="#333333")

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#D32F2F", alpha=0.88, label="恶性倾向关键词（+）"),
    Patch(facecolor="#1976D2", alpha=0.88, label="良性倾向关键词（−）"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

# Adjust x-limits for label space
x_max = max(abs(v) + err for v, err in zip(mean_vals, errors)) * 1.25
ax.set_xlim(-x_max, x_max)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_keyword_contribution.png")
fig.savefig(OUT_DIR / "fig2_keyword_contribution.pdf")
plt.close(fig)
print(f"Saved: fig2_keyword_contribution.png/pdf")


# ══════════════════════════════════════════════════════════════
# Figure 3: Category-level Contribution (Chinese)
# ══════════════════════════════════════════════════════════════
CATEGORIES = {
    "病灶形态学特征": {
        "keywords": ["mass", "calcification", "irregular", "spiculated",
                      "circumscribed", "obscured", "microlobulated"],
        "color": "#E91E63",
    },
    "钙化形态与分布": {
        "keywords": ["pleomorphic", "linear", "clustered", "segmental"],
        "color": "#7B1FA2",
    },
    "解剖位置/成像体位": {
        "keywords": ["left breast", "right breast", "cc view", "mlo view"],
        "color": "#546E7A",
    },
}

cat_means = {}
cat_stds = {}
for cat_name, info in CATEGORIES.items():
    vals = []
    for kw in info["keywords"]:
        if kw in data:
            vals.append(data[kw]["abs_mean"])
    cat_means[cat_name] = np.mean(vals) if vals else 0
    cat_stds[cat_name] = np.std(vals) if vals else 0

cat_names = list(cat_means.keys())
cat_vals = list(cat_means.values())
cat_errs = list(cat_stds.values())
cat_colors = [info["color"] for info in CATEGORIES.values()]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
x_pos = np.arange(len(cat_names))

bars2 = ax2.bar(x_pos, cat_vals, color=cat_colors, alpha=0.85,
                edgecolor="white", linewidth=0.5, width=0.55)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(cat_names, fontsize=11)
ax2.set_ylabel("平均绝对概率变化量（Mean |Δ Probability|）", fontsize=11)
ax2.set_title("按语义类别分组的病理关键词贡献度",
             fontweight="bold", fontsize=12, pad=12)

# Value labels on top of bars
for bar, val in zip(bars2, cat_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", fontsize=11, fontweight="bold",
             color="#333333")

ax2.set_ylim(0, max(cat_vals) * 1.15)

plt.tight_layout()
fig2.savefig(OUT_DIR / "fig3_keyword_category.png")
fig2.savefig(OUT_DIR / "fig3_keyword_category.pdf")
plt.close(fig2)
print(f"Saved: fig3_keyword_category.png/pdf")


# ══════════════════════════════════════════════════════════════
# Copy all existing paper figures to the unified folder
# ══════════════════════════════════════════════════════════════
print("\n--- Copying existing figures ---")

SOURCE_DIRS = [
    Path("paper/figures"),
    Path("paper/figures/analysis"),
    Path("paper/figures/vector"),
]

copied = set()
for src_dir in SOURCE_DIRS:
    if not src_dir.exists():
        continue
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() not in (".png", ".pdf"):
            continue
        # Skip the old keyword figures (we just replaced them)
        if "keyword_contribution" in f.name or "keyword_category" in f.name:
            continue
        dest = OUT_DIR / f.name
        # Avoid overwriting files we already created
        if dest.exists():
            continue
        dest.write_bytes(f.read_bytes())
        copied.add(f.name)

print(f"Copied {len(copied)} existing figures")
print(f"\nAll figures consolidated in: {OUT_DIR}/")
print(f"Total files: {len(list(OUT_DIR.iterdir()))}")
