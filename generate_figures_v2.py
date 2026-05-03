"""
Paper Figures v2 — NeurIPS/ICML quality
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy import stats

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 2.0,
    "patch.linewidth": 0.5,
})

BLUE = "#1B3A6B"
RED = "#C0392B"
GREEN = "#0D7377"
ORANGE = "#E67E22"
PURPLE = "#8E44AD"
GRAY = "#7F8C8D"
LIGHT_GRAY = "#BDC3C7"

FAMILY_COLORS = {
    "GPT-2": BLUE,
    "Phi": ORANGE,
    "Qwen": PURPLE,
    "Pythia": GREEN,
    "GPT-Neo": RED,
}

# ── Figure 1: Layer probability curve ─────────────────────────────

print("Figure 1 — Layer probability curve...")

with open("results/experiment_01_gpt2_xl.json") as f:
    xl_data = json.load(f)

france = next(
    r for r in xl_data["results"] if "France" in r["prompt"]
)
layer_probs = france["layer_probs"]
n_layers = len(layer_probs)
layers = list(range(n_layers))
peak_layer = france["peak_layer"]
peak_prob = france["peak_prob"]
final_prob = france["final_prob"]
ratio = france["suppression_ratio"]

fig, ax = plt.subplots(figsize=(7.5, 4.2))

ax.fill_between(layers, layer_probs, alpha=0.06, color=BLUE)
ax.plot(layers, layer_probs, color=BLUE, linewidth=2.2,
        zorder=4, label='P("Paris" | logit lens at layer k)')

ax.scatter([peak_layer], [peak_prob],
           color=GREEN, s=80, zorder=5)
ax.scatter([n_layers - 1], [final_prob],
           color=RED, s=80, zorder=5)

ax.annotate(
    f'Peak: {peak_prob:.3f}\n(Block {peak_layer})',
    xy=(peak_layer, peak_prob),
    xytext=(peak_layer - 13, peak_prob + 0.022),
    fontsize=10.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN,
                    lw=1.4, connectionstyle="arc3,rad=0.1")
)

ax.annotate(
    f'Final: {final_prob:.4f}\n({ratio:.0f}× suppressed)',
    xy=(n_layers - 1, final_prob),
    xytext=(n_layers - 17, final_prob + 0.048),
    fontsize=10.5, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED,
                    lw=1.4, connectionstyle="arc3,rad=-0.1")
)

ax.axvspan(peak_layer - 0.4, n_layers - 0.6,
           alpha=0.04, color=RED,
           label="Suppression region")

ax.set_xlabel("Transformer Layer (Block Index)")
ax.set_ylabel("Probability of Correct Answer\n"
               'P("Paris" | logit lens at layer k)')
ax.set_title(
    "Last-Layer Suppression: GPT-2 XL knows "
    '"Paris" but suppresses it at the final layer',
    pad=10, fontsize=12
)
ax.legend(loc="upper left", framealpha=0.92,
          edgecolor=LIGHT_GRAY, frameon=True)
ax.set_xlim(-1, n_layers)
ax.set_ylim(bottom=-0.002)

plt.tight_layout()
plt.savefig("figures/figure1_layer_suppression.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure1_layer_suppression.png",
            bbox_inches="tight")
plt.close()
print("  Done.")

# ── Figure 2: Scaling law (single clean panel) ────────────────────

print("Figure 2 — Scaling law...")

scaling_files = [
    ("results/experiment_01_layer_analysis.json",
     "TYPE2A_LAST_LAYER_SUPPRESSION", 0.124, 12, "Small\n124M"),
    ("results/experiment_01_gpt2_medium.json",
     "TYPE2A_LAST_LAYER_SUPPRESSION", 0.345, 24, "Medium\n345M"),
    ("results/experiment_01_gpt2_large.json",
     "TYPE2A_LAST_LAYER_SUPPRESSION", 0.774, 36, "Large\n774M"),
    ("results/experiment_01_gpt2_xl.json",
     "TYPE2A_LAST_LAYER_SUPPRESSION", 1.5, 48, "XL\n1.5B"),
]

params = []
suppressions = []
type2a_pcts = []
rel_depths = []
labels = []

for fpath, key, p, nl, lbl in scaling_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data.get("results", [])
    t2a = [r for r in results if r["hallucination_type"] == key]
    if not t2a:
        continue
    avg_s = np.mean([r.get("suppression_ratio", 0) for r in t2a])
    avg_r = np.mean([
        r.get("peak_layer_relative",
              r.get("relative_depth", 0))
        for r in t2a
    ])
    t2a_pct = len(t2a) / len(results) * 100
    params.append(p)
    suppressions.append(avg_s)
    type2a_pcts.append(t2a_pct)
    rel_depths.append(avg_r)
    labels.append(lbl)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))

panel_data = [
    (suppressions, "Average Suppression Ratio (×)",
     "Suppression Ratio Scales with Model Size"),
    (type2a_pcts, "Type 2a Cases (%)",
     "Type 2a Rate Increases with Scale"),
    (rel_depths, "Relative Peak Layer Depth (0–1)",
     "Peak Layer Converges at ~0.83 Depth"),
]

for ax, (ydata, ylabel, title) in zip(axes, panel_data):
    ax.plot(params, ydata, 'o-',
            color=BLUE, linewidth=2.2,
            markersize=9, markerfacecolor="white",
            markeredgewidth=2.2, zorder=4)

    for p, y, lbl in zip(params, ydata, labels):
        ax.annotate(lbl, (p, y),
                    textcoords="offset points",
                    xytext=(8, 2), fontsize=9,
                    color=BLUE)

    ax.set_xlabel("Parameters (Billions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xscale("log")
    ax.set_xticks([0.124, 0.345, 0.774, 1.5])
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, _: f"{x}B")
    )

    if "Depth" in title:
        ax.axhline(y=0.83, color=GRAY, linestyle=":",
                   linewidth=1.5,
                   label="0.83 universal constant")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9.5, framealpha=0.9)

fig.suptitle(
    "Figure 2: Last-Layer Suppression Scaling Laws "
    "in the GPT-2 Family",
    fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig("figures/figure2_scaling_law.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure2_scaling_law.png",
            bbox_inches="tight")
plt.close()
print("  Done.")

# ── Figure 3: Scatter plot — two families ─────────────────────────

print("Figure 3 — Architecture families scatter...")

models = [
    ("GPT-2 XL", "GPT-2", 1.5, 20.8, 50, 45),
    ("GPT-2 Large", "GPT-2", 0.774, 17.1, 15, 0),
    ("GPT-2 Medium", "GPT-2", 0.345, 15.6, 40, 0),
    ("Phi-2", "Phi", 2.7, 10.8, 20, 5),
    ("Qwen 1.5", "Qwen", 1.8, 2.5, 70, 40),
    ("GPT-2 Small", "GPT-2", 0.124, 2.4, 10, 0),
    ("Pythia 2.8B", "Pythia", 2.8, 1.1, 50, 0),
    ("GPT-Neo 2.7B", "GPT-Neo", 2.7, 1.0, 65, 0),
    ("GPT-Neo 1.3B", "GPT-Neo", 1.3, 1.0, 45, 0),
    ("GPT-Neo 125M", "GPT-Neo", 0.125, 1.0, 25, 0),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

ax = axes[0]
for name, family, params_b, supp, t2a, interv in models:
    color = FAMILY_COLORS[family]
    ax.scatter(params_b, supp, color=color,
               s=120, zorder=4,
               edgecolors="white", linewidth=0.8)
    offset = (6, 4)
    if name == "GPT-Neo 2.7B":
        offset = (6, -12)
    if name == "Pythia 2.8B":
        offset = (-60, 6)
    if name == "Qwen 1.5":
        offset = (6, -12)
    ax.annotate(name, (params_b, supp),
                textcoords="offset points",
                xytext=offset, fontsize=8.5,
                color=color)

ax.axhline(y=1.5, color=GRAY, linestyle="--",
           linewidth=1.2, alpha=0.7,
           label="Suppression threshold (1.5×)")
ax.fill_between([0.1, 3.0], [1.5, 1.5], [25, 25],
                alpha=0.04, color=BLUE,
                label="Strong suppression family")
ax.fill_between([0.1, 3.0], [0, 0], [1.5, 1.5],
                alpha=0.04, color=RED,
                label="Weak suppression family")
ax.set_xlabel("Model Parameters (Billions)")
ax.set_ylabel("Average Suppression Ratio (×)")
ax.set_title("Suppression Ratio vs Model Size\nby Architecture Family",
             fontsize=11)
ax.set_xscale("log")
ax.set_xlim(0.08, 4.0)
ax.set_ylim(-0.5, 24)
ax.legend(fontsize=9, framealpha=0.9,
          edgecolor=LIGHT_GRAY, loc="upper left")

ax = axes[1]
for name, family, params_b, supp, t2a, interv in models:
    color = FAMILY_COLORS[family]
    ax.scatter(supp, interv, color=color,
               s=120, zorder=4,
               edgecolors="white", linewidth=0.8)
    offset = (6, 4)
    if name == "GPT-Neo 2.7B":
        offset = (6, -12)
    if name == "Pythia 2.8B":
        offset = (-65, 6)
    if name == "Qwen 1.5":
        offset = (6, -14)
    ax.annotate(name, (supp, interv),
                textcoords="offset points",
                xytext=offset, fontsize=8.5,
                color=color)

supp_vals = [m[3] for m in models]
interv_vals = [m[5] for m in models]
slope, intercept, r, p_val, _ = stats.linregress(
    supp_vals, interv_vals
)
x_line = np.linspace(0, 22, 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color=GRAY,
        linestyle="--", linewidth=1.2,
        alpha=0.6,
        label=f"Linear fit (r={r:.2f}, p={p_val:.3f})")

ax.set_xlabel("Suppression Ratio (×)")
ax.set_ylabel("Intervention Improvement (%)")
ax.set_title("Suppression Ratio Predicts\nIntervention Effectiveness",
             fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, edgecolor=LIGHT_GRAY)
ax.set_xlim(-0.5, 23)
ax.set_ylim(-3, 52)

legend_patches = [
    mpatches.Patch(color=c, label=f)
    for f, c in FAMILY_COLORS.items()
]
fig.legend(
    handles=legend_patches,
    loc="lower center", ncol=5,
    bbox_to_anchor=(0.5, -0.07),
    title="Model Family",
    framealpha=0.92,
    edgecolor=LIGHT_GRAY,
    fontsize=10
)

fig.suptitle(
    "Figure 3: Two Distinct Hallucination Families "
    "Across 10 Models and 5 Architecture Families",
    fontsize=13
)
plt.tight_layout()
plt.savefig("figures/figure3_architecture_families.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure3_architecture_families.png",
            bbox_inches="tight")
plt.close()
print("  Done.")

# ── Figure 4: HallBench v2 ────────────────────────────────────────

print("Figure 4 — HallBench v2...")

hb = {
    "GPT-2 XL\n(strong)": {
        "t1": (36, 80), "t2": (47, 53), "t3": (0, 0),
        "family": "strong"
    },
    "Qwen 1.5\n(strong)": {
        "t1": (32, 60), "t2": (7, 40), "t3": (30, 30),
        "family": "strong"
    },
    "Phi-2\n(strong)": {
        "t1": (72, 80), "t2": (53, 53), "t3": (30, 30),
        "family": "strong"
    },
    "GPT-Neo 2.7B\n(weak)": {
        "t1": (20, 20), "t2": (20, 20), "t3": (20, 20),
        "family": "weak"
    },
}

tier_keys = ["t1", "t2", "t3"]
tier_titles = [
    "Tier 1: High Suppression Facts\n"
    "(capitals & science — confirmed Type 2a)",
    "Tier 2: Borderline Facts\n"
    "(sometimes survive — model-size dependent)",
    "Tier 3: Knowledge Gap Facts\n"
    "(Type 2b — intervention has no effect)",
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
model_names = list(hb.keys())
x = np.arange(len(model_names))
width = 0.36

for t_idx, (tk, title) in enumerate(zip(tier_keys, tier_titles)):
    ax = axes[t_idx]
    baselines = [hb[m][tk][0] for m in model_names]
    bests = [hb[m][tk][1] for m in model_names]
    bar_colors = [
        BLUE if hb[m]["family"] == "strong" else RED
        for m in model_names
    ]

    b1 = ax.bar(x - width / 2, baselines, width,
                label="Baseline (α=0)", color=LIGHT_GRAY,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, bests, width,
                label="With Intervention",
                color=bar_colors, alpha=0.88,
                edgecolor="white", linewidth=0.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h >= 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2, f"{h:.0f}%",
                ha="center", va="bottom",
                fontsize=8.5, color="#333333"
            )

    for i, (bl, bs) in enumerate(zip(baselines, bests)):
        if bs > bl:
            ax.annotate(
                f"+{bs-bl}%",
                xy=(x[i] + width / 2, bs),
                xytext=(x[i] + width / 2, bs + 8),
                ha="center", fontsize=8,
                color=GREEN, fontweight="bold",
                arrowprops=dict(
                    arrowstyle="-",
                    color=GREEN, lw=0.8
                )
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9.5, framealpha=0.92,
              edgecolor=LIGHT_GRAY, loc="upper right")

    if t_idx == 2:
        ax.text(
            0.5, 0.42,
            "✓ Intervention correctly\nhas no effect on\nunknown facts",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9.5, color=GRAY,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F8F9FA",
                      edgecolor=LIGHT_GRAY,
                      alpha=0.9)
        )

fig.suptitle(
    "Figure 4: HallBench v2 — Logit Blending Recovers Suppressed "
    "Knowledge (Tier 1–2) Without Hallucinating New Facts (Tier 3)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("figures/figure4_hallbench_v2.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure4_hallbench_v2.png",
            bbox_inches="tight")
plt.close()
print("  Done.")

print("\nAll figures saved to figures/")
for f in sorted(os.listdir("figures")):
    if f.endswith(".png"):
        size = os.path.getsize(f"figures/{f}") // 1024
        print(f"  {f} ({size} KB)")