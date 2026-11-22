#!/usr/bin/env python3
"""English vector Fig.1 (KG-TPCA flowchart) for the AIIM journal submission.

Outputs: AIIM/figures/fig1_vector.pdf (vector) and fig1_vector.png (300 dpi).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_INPUT = "#E3F2FD"
C_ENCODER = "#C8E6C9"
C_PROMPT = "#FFF9C4"
C_CORE = "#FFE0B2"
C_LOSS = "#F3E5F5"
C_OUTPUT = "#FFCDD2"
C_BORDER = "#546E7A"

BOXES = [
    # inputs
    (0.4, 7.6, 2.0, 0.9, "CC image +\nlesion mask", C_INPUT, True, 6),
    (0.4, 5.9, 2.0, 0.9, "MLO image +\nlesion mask", C_INPUT, True, 6),
    (0.4, 3.1, 2.0, 0.9, "CC/MLO structured\ndiagnostic text", C_INPUT, True, 5.5),
    (0.4, 0.6, 2.0, 0.8, "16 BI-RADS\nlesion keywords", C_INPUT, True, 6),
    # visual encoders
    (3.0, 7.6, 2.0, 0.9, "Visual encoder\nBiomedCLIP (frozen)\nglobal + local feat.", C_ENCODER, True, 5.5),
    (3.0, 5.9, 2.0, 0.9, "Visual encoder\nBiomedCLIP (frozen)\nglobal + local feat.", C_ENCODER, True, 5.5),
    # gated fusion
    (5.6, 7.6, 1.8, 0.9, "Global-local\ngated fusion", C_ENCODER, False, 6),
    (5.6, 5.9, 1.8, 0.9, "Global-local\ngated fusion", C_ENCODER, False, 6),
    # text path
    (3.0, 3.1, 2.0, 0.9, "Keyword extraction\n(regex matching)", C_PROMPT, False, 5.5),
    (5.6, 3.1, 2.0, 0.9, "BI-RADS KG-enhanced\nprompt encoder\n(KG regularization)", C_PROMPT, False, 5.5),
    # prompt gates
    (8.0, 7.6, 1.6, 0.9, "Prompt attention\ngating", C_PROMPT, False, 6),
    (8.0, 5.9, 1.6, 0.9, "Prompt attention\ngating", C_PROMPT, False, 6),
    # single-view classifiers
    (8.0, 4.2, 1.6, 0.8, "Single-view\nclassifier", C_CORE, False, 6),
    (8.0, 1.4, 1.6, 0.8, "Single-view\nclassifier", C_CORE, False, 6),
    # dual-view fusion + classifier
    (10.2, 6.8, 1.9, 0.9, "Dual-view fusion\n(equal-weight avg.)", C_CORE, False, 6),
    (10.2, 4.4, 1.9, 0.9, "Dual-view\nclassifier", C_CORE, False, 6),
    # loss + output
    (12.8, 6.2, 1.8, 2.2, "Joint loss\nL_total = L_single\n+ L_pair_cls\n+ l1 * L_prior\n+ l2 * L_rel", C_LOSS, False, 5.5),
    (12.8, 3.6, 1.7, 0.9, "Benign/malignant\noutput", C_OUTPUT, True, 6.5),
]

ARROWS = [
    (2.4, 8.05, 3.0, 8.05),
    (2.4, 6.35, 3.0, 6.35),
    (5.0, 8.05, 5.6, 8.05),
    (5.0, 6.35, 5.6, 6.35),
    (2.4, 3.55, 3.0, 3.55),
    (5.0, 3.55, 5.6, 3.55),
    (1.4, 1.4, 1.4, 2.2),
    (1.4, 2.2, 5.6, 3.1),
    (7.4, 8.05, 8.0, 8.05),
    (7.4, 6.35, 8.0, 6.35),
    (6.6, 3.55, 6.6, 4.6),
    (6.6, 4.6, 8.0, 7.6),
    (6.6, 4.6, 8.0, 5.9),
    (8.8, 7.6, 8.8, 5.0),
    (8.8, 5.9, 8.8, 4.2),
    (9.6, 8.05, 10.2, 7.5),
    (9.6, 6.35, 10.2, 6.9),
    (11.15, 6.8, 11.15, 5.3),
    (12.1, 4.85, 12.8, 4.3),
    (9.6, 4.6, 12.8, 6.4),
    (9.6, 1.8, 12.8, 6.2),
    (11.15, 4.4, 12.8, 6.2),
]

LABELS = [
    (1.4, 8.9, "Input", C_BORDER, 8),
    (4.0, 8.9, "Visual encoding (frozen backbone)", C_BORDER, 8),
    (3.0, 4.4, "Text prompt encoding", C_BORDER, 8),
    (8.8, 8.9, "Prompt gating & classification", C_BORDER, 8),
    (13.6, 8.9, "Loss & output", C_BORDER, 8),
]


def draw_box(ax, x, y, w, h, text, color, bold, fontsize):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="#455A64", linewidth=1.1,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight="bold" if bold else "normal")


def draw_arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0,
                                linestyle="--" if dashed else "-"))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("white")

    for x, y, w, h, text, color, bold, fs in BOXES:
        draw_box(ax, x, y, w, h, text, color, bold, fs)
    for x1, y1, x2, y2 in ARROWS:
        draw_arrow(ax, x1, y1, x2, y2)
    draw_arrow(ax, 9.6, 4.6, 12.8, 6.4, dashed=True)
    draw_arrow(ax, 9.6, 1.8, 12.8, 6.2, dashed=True)
    for x, y, text, color, fs in LABELS:
        ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center", style="italic")

    ax.text(8, 9.4, "KG-TPCA Overall Architecture", ha="center", fontsize=13, fontweight="bold", color="#1A237E")

    fig.tight_layout(pad=0.4)
    out_dir = "AIIM/figures"
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/fig1_vector.pdf", facecolor="white")
    fig.savefig(f"{out_dir}/fig1_vector.png", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved {out_dir}/fig1_vector.pdf and fig1_vector.png")


if __name__ == "__main__":
    main()
