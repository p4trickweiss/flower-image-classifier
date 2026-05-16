"""
Generates architecture diagrams for all model configs.
Saves PNG files to docs/images/architectures/.

Usage:
    python src/visualize_architectures.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = "docs/images/architectures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors per layer type ─────────────────────────────────────────────────────
C = {
    "aug":   "#e0e0e0",
    "conv":  "#4C72B0",
    "bn":    "#55A868",
    "pool":  "#C44E52",
    "flat":  "#8172B2",
    "dense": "#DD8452",
    "drop":  "#b0b0b0",
    "soft":  "#64B5CD",
    "svm":   "#C44E52",
    "input": "#aec6cf",
}

def draw_block(ax, x, y, w, h, color, label, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.04",
                          facecolor=color, edgecolor="#333333", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white" if color not in (C["aug"], C["drop"], C["input"]) else "#333333",
            wrap=True, multialignment="center")

def arrow(ax, x_start, x_end, y=2.0):
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4))

def save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Helper: draw a generic CNN diagram ───────────────────────────────────────
def draw_cnn(title, conv_blocks, dense_units, dropout, optimizer, lr, filename):
    """
    conv_blocks: list of (n_filters, label) e.g. [(32,"32"), (64,"64"), ...]
    """
    n = len(conv_blocks)
    # fixed blocks: input, aug, [conv+bn+pool]*n, flat, dense, drop, softmax
    total = 2 + n * 3 + 3
    fig_w = max(14, total * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw, bh, gap = 0.75, 2.0, 0.25   # block width, height, gap between groups
    y0 = 1.5
    x = 0.3

    def blk(label, color, width=bw):
        nonlocal x
        draw_block(ax, x, y0, width, bh, color, label, fontsize=7)
        cx = x + width / 2
        x += width + 0.15
        return cx

    # Input
    cx_prev = blk("Input\n128×128×3", C["input"], width=0.9)

    # Augmentation group
    x += 0.1
    aug_x = x
    cx_prev2 = blk("Aug\n(train)", C["aug"])
    x += 0.1

    # Conv blocks
    for i, (f, flabel) in enumerate(conv_blocks):
        blk(f"Conv2D\n{flabel} filters", C["conv"])
        blk(f"BN", C["bn"], width=0.4)
        blk(f"MaxPool\n2×2", C["pool"])
        x += gap

    # Flatten
    blk("Flatten", C["flat"], width=0.6)

    # Dense + Dropout + Softmax
    blk(f"Dense\n{dense_units}", C["dense"])
    blk(f"Dropout\n{dropout}", C["drop"], width=0.6)
    blk("Softmax\n5 classes", C["soft"])

    # Arrows
    layer_centers = []
    _x = 0.3 + 0.45   # center of first block
    # (approximate: just draw a baseline arrow under everything)
    ax.annotate("", xy=(x - 0.15, y0 - 0.3),
                xytext=(0.3, y0 - 0.3),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))

    # Title + meta info
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    meta = f"Optimizer: {optimizer}  |  LR: {lr}  |  Dropout: {dropout}"
    ax.text(fig_w / 2, 0.3, meta, ha="center", fontsize=9, color="#555555")

    # Legend
    legend_patches = [
        mpatches.Patch(color=C["conv"],  label="Conv2D"),
        mpatches.Patch(color=C["bn"],    label="BatchNorm"),
        mpatches.Patch(color=C["pool"],  label="MaxPooling"),
        mpatches.Patch(color=C["flat"],  label="Flatten"),
        mpatches.Patch(color=C["dense"], label="Dense"),
        mpatches.Patch(color=C["drop"],  label="Dropout"),
        mpatches.Patch(color=C["soft"],  label="Softmax"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7, ncol=2, framealpha=0.9)

    save(fig, filename)


# ── 1. baseline ───────────────────────────────────────────────────────────────
draw_cnn(
    title="baseline – 4-Block CNN (Adam)",
    conv_blocks=[(32,"32"), (64,"64"), (128,"128"), (256,"256")],
    dense_units=512, dropout=0.5,
    optimizer="Adam", lr="0.001",
    filename="baseline",
)

# ── 2. shallow_sgd ────────────────────────────────────────────────────────────
draw_cnn(
    title="shallow_sgd – 2-Block CNN (SGD)",
    conv_blocks=[(32,"32"), (64,"64")],
    dense_units=256, dropout=0.3,
    optimizer="SGD", lr="0.01",
    filename="shallow_sgd",
)

# ── 3. wide_rmsprop ───────────────────────────────────────────────────────────
draw_cnn(
    title="wide_rmsprop – 4-Block CNN wide (RMSprop)",
    conv_blocks=[(64,"64"), (128,"128"), (256,"256"), (512,"512")],
    dense_units=512, dropout=0.5,
    optimizer="RMSprop", lr="0.0005",
    filename="wide_rmsprop",
)

# ── 4. AlexNet ────────────────────────────────────────────────────────────────
def draw_alexnet():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y0, bh = 1.5, 2.0
    x = 0.3

    def blk(label, color, width=0.85):
        nonlocal x
        rect = FancyBboxPatch((x, y0), width, bh,
                              boxstyle="round,pad=0.04",
                              facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + width / 2, y0 + bh / 2, label,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if color not in (C["aug"], C["input"]) else "#333333",
                multialignment="center")
        x += width + 0.15

    blk("Input\n224×224×3", C["input"], width=0.95)
    blk("Aug\n(train)", C["aug"], width=0.7)
    x += 0.1
    blk("Conv2D\n96 / 11×11\nstride 4", C["conv"])
    blk("BN+\nMaxPool", C["bn"], width=0.65)
    blk("Conv2D\n256 / 5×5", C["conv"])
    blk("BN+\nMaxPool", C["bn"], width=0.65)
    blk("Conv2D\n384 / 3×3", C["conv"])
    blk("Conv2D\n384 / 3×3", C["conv"])
    blk("Conv2D\n256 / 3×3", C["conv"])
    blk("MaxPool", C["pool"], width=0.7)
    blk("Flatten", C["flat"], width=0.65)
    blk("Dense\n4096", C["dense"])
    blk("Drop\n0.5", C["drop"], width=0.55)
    blk("Dense\n4096", C["dense"])
    blk("Drop\n0.5", C["drop"], width=0.55)
    blk("Softmax\n5 classes", C["soft"])

    ax.set_title("alexnet – AlexNet-inspired (Adam, lr=0.0001)", fontsize=13, fontweight="bold", pad=10)
    ax.text(8, 0.3, "Input: 224×224  |  Optimizer: Adam  |  LR: 0.0001  |  Dropout: 0.5  |  58.3M params",
            ha="center", fontsize=9, color="#555555")

    legend_patches = [
        mpatches.Patch(color=C["conv"],  label="Conv2D"),
        mpatches.Patch(color=C["bn"],    label="BN / MaxPool"),
        mpatches.Patch(color=C["flat"],  label="Flatten"),
        mpatches.Patch(color=C["dense"], label="Dense"),
        mpatches.Patch(color=C["drop"],  label="Dropout"),
        mpatches.Patch(color=C["soft"],  label="Softmax"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    save(fig, "alexnet")

draw_alexnet()

# ── 5. CNN+SVM ────────────────────────────────────────────────────────────────
def draw_cnn_svm():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y0, bh = 1.5, 2.0
    x = 0.3

    blocks = [
        ("Input\n128×128×3",      C["input"],  0.95),
        ("Aug\n(train)",          C["aug"],    0.75),
        ("Conv Block 1\n32 filters", C["conv"], 1.1),
        ("Conv Block 2\n64 filters", C["conv"], 1.1),
        ("Conv Block 3\n128 filters",C["conv"], 1.1),
        ("Conv Block 4\n256 filters",C["conv"], 1.1),
        ("Dense\n512-dim\n(frozen)", C["dense"],1.0),
        ("Standard\nScaler",      C["flat"],   0.85),
        ("SVM\nRBF / C=10",       C["svm"],    0.85),
        ("Class\n(5)",            C["soft"],   0.75),
    ]

    centers = []
    for label, color, w in blocks:
        rect = FancyBboxPatch((x, y0), w, bh,
                              boxstyle="round,pad=0.04",
                              facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + bh / 2, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white" if color not in (C["aug"], C["input"]) else "#333333",
                multialignment="center")
        centers.append(x + w / 2)
        x += w + 0.18

    # Arrows
    for i in range(len(centers) - 1):
        ax.annotate("", xy=(centers[i+1] - 0.4, y0 + bh / 2),
                    xytext=(centers[i] + 0.4, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4))

    # Brackets
    cnn_start = centers[1] - 0.3
    cnn_end   = centers[6] + 0.4
    svm_start = centers[7] - 0.3
    svm_end   = centers[8] + 0.4

    ax.annotate("", xy=(cnn_end, y0 - 0.4), xytext=(cnn_start, y0 - 0.4),
                arrowprops=dict(arrowstyle="<->", color=C["conv"], lw=1.5))
    ax.text((cnn_start + cnn_end) / 2, y0 - 0.75,
            "CNN Backbone (Keras, frozen)", ha="center", fontsize=8, color=C["conv"], fontweight="bold")

    ax.annotate("", xy=(svm_end, y0 - 0.4), xytext=(svm_start, y0 - 0.4),
                arrowprops=dict(arrowstyle="<->", color=C["svm"], lw=1.5))
    ax.text((svm_start + svm_end) / 2, y0 - 0.75,
            "Classifier (sklearn)", ha="center", fontsize=8, color=C["svm"], fontweight="bold")

    ax.set_title("cnn_svm – Hybrid CNN + SVM", fontsize=13, fontweight="bold", pad=10)
    ax.text(7, 4.65, "baseline_best_model.keras  (feature extraction)  +  cnn_svm_classifier.pkl  (SVM)",
            ha="center", fontsize=8.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc"))

    save(fig, "cnn_svm")

draw_cnn_svm()

print("\nDone. Check docs/images/architectures/")
