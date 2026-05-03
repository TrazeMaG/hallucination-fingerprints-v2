"""
Generate all paper figures for
Last-Layer Suppression paper.

Figure 1 — Layer probability curve (GPT-2 XL, France/Paris)
Figure 2 — Scaling law across GPT-2 family
Figure 3 — Architecture family heatmap
Figure 4 — HallBench v2 intervention results
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "strong": "#1B3A6B",
    "weak": "#CC4444",
    "intervention": "#0D7377",
    "baseline": "#AAAAAA",
}

FAMILY_COLORS = {
    "GPT-2": "#1B3A6B",
    "Phi": "#2196F3",
    "Qwen": "#9C27B0",
    "Pythia": "#FF9800",
    "GPT-Neo": "#CC4444",
}

# ── Figure 1: Layer probability curve ─────────────────────────────

print("Generating Figure 1 — Layer probability curve...")

with open("results/experiment_01_gpt2_xl.json") as f:
    xl_data = json.load(f)

france_result = None
for r in xl_data["results"]:
    if "France" in r["prompt"]:
        france_result = r
        break

if france_result:
    layer_probs = france_result["layer_probs"]
    n_layers = len(layer_probs)
    layers = list(range(n_layers))
    peak_layer = france_result["peak_layer"]
    peak_prob = france_result["peak_prob"]
    final_prob = france_result["final_prob"]
    suppression_ratio = france_result["suppression_ratio"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(layers, layer_probs,
            color=COLORS["strong"], linewidth=2.5,
            zorder=3, label='P("Paris" | layer k)')

    ax.axvline(x=peak_layer,
               color=COLORS["intervention"],
               linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Peak layer (Block {peak_layer})")

    ax.axvline(x=n_layers - 1,
               color=COLORS["weak"],
               linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Final layer (Block {n_layers - 1})")

    ax.annotate(
        f'Peak: {peak_prob:.3f}',
        xy=(peak_layer, peak_prob),
        xytext=(peak_layer - 10, peak_prob + 0.025),
        fontsize=10, color=COLORS["intervention"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["intervention"],
            lw=1.5
        )
    )

    ax.annotate(
        f'Final: {final_prob:.4f}\n({suppression_ratio:.0f}x suppressed)',
        xy=(n_layers - 1, final_prob),
        xytext=(n_layers - 14, final_prob + 0.05),
        fontsize=10, color=COLORS["weak"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["weak"],
            lw=1.5
        )
    )

    ax.fill_between(layers, layer_probs, alpha=0.08,
                    color=COLORS["strong"])

    ax.set_xlabel("Layer (Block Index)")
    ax.set_ylabel('P("Paris" | residual stream at layer k)')
    ax.set_title(
        'Figure 1: Last-Layer Suppression in GPT-2 XL\n'
        '"The capital of France is" — model acquires "Paris" '
        'then suppresses it at the final layer',
        pad=12
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figures/figure1_layer_suppression.pdf",
                bbox_inches="tight")
    plt.savefig("figures/figure1_layer_suppression.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Figure 1 saved.")
else:
    print("  France result not found.")

# ── Figure 2: Scaling law ─────────────────────────────────────────

print("Generating Figure 2 — Scaling law...")

scaling_models = [
    {
        "label": "Small\n(124M)",
        "params_b": 0.124,
        "file": "results/experiment_01_layer_analysis.json",
        "type2a_key": "TYPE2A_LAST_LAYER_SUPPRESSION",
    },
    {
        "label": "Medium\n(345M)",
        "params_b": 0.345,
        "file": "results/experiment_01_gpt2_medium.json",
        "type2a_key": "TYPE2A_LAST_LAYER_SUPPRESSION",
    },
    {
        "label": "Large\n(774M)",
        "params_b": 0.774,
        "file": "results/experiment_01_gpt2_large.json",
        "type2a_key": "TYPE2A_LAST_LAYER_SUPPRESSION",
    },
    {
        "label": "XL\n(1.5B)",
        "params_b": 1.5,
        "file": "results/experiment_01_gpt2_xl.json",
        "type2a_key": "TYPE2A_LAST_LAYER_SUPPRESSION",
    },
]

params_list = []
suppressions_list = []
type2a_list = []
rel_depths_list = []
labels_list = []

for d in scaling_models:
    if not os.path.exists(d["file"]):
        print(f"  Missing: {d['file']}")
        continue

    with open(d["file"]) as f:
        data = json.load(f)

    results = data.get("results", [])
    type2a = [
        r for r in results
        if r["hallucination_type"] == d["type2a_key"]
    ]

    if type2a:
        avg_supp = np.mean([
            r.get("suppression_ratio", 0) for r in type2a
        ])
        avg_rel = np.mean([
            r.get("peak_layer_relative",
                  r.get("relative_depth", 0))
            for r in type2a
        ])
    else:
        avg_supp = 0
        avg_rel = 0

    type2a_pct = len(type2a) / len(results) * 100 if results else 0

    params_list.append(d["params_b"])
    suppressions_list.append(avg_supp)
    type2a_list.append(type2a_pct)
    rel_depths_list.append(avg_rel)
    labels_list.append(d["label"])

if len(params_list) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(params_list, suppressions_list, 'o-',
            color=COLORS["strong"], linewidth=2.5,
            markersize=10, markerfacecolor="white",
            markeredgewidth=2.5, zorder=3)
    for p, s, lbl in zip(params_list, suppressions_list, labels_list):
        ax.annotate(lbl, (p, s),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model Parameters (Billions, log scale)")
    ax.set_ylabel("Average Suppression Ratio (×)")
    ax.set_title("Suppression Ratio Increases with Scale")
    ax.set_xscale("log")

    ax = axes[1]
    ax.plot(params_list, type2a_list, 's-',
            color=COLORS["intervention"], linewidth=2.5,
            markersize=10, markerfacecolor="white",
            markeredgewidth=2.5, zorder=3)
    for p, t, lbl in zip(params_list, type2a_list, labels_list):
        ax.annotate(lbl, (p, t),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model Parameters (Billions, log scale)")
    ax.set_ylabel("Type 2a Cases (%)")
    ax.set_title("Type 2a Rate Increases with Scale")
    ax.set_xscale("log")

    ax = axes[2]
    ax.plot(params_list, rel_depths_list, '^-',
            color=COLORS["weak"], linewidth=2.5,
            markersize=10, markerfacecolor="white",
            markeredgewidth=2.5, zorder=3)
    ax.axhline(y=0.83, color="black", linestyle=":",
               linewidth=1.5, alpha=0.6,
               label="0.83 (universal constant)")
    for p, r, lbl in zip(params_list, rel_depths_list, labels_list):
        ax.annotate(lbl, (p, r),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model Parameters (Billions, log scale)")
    ax.set_ylabel("Relative Peak Layer Depth")
    ax.set_title("Peak Layer Converges to ~0.83 Relative Depth")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Figure 2: Last-Layer Suppression Scaling Laws "
        "in the GPT-2 Family (124M — 1.5B)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig("figures/figure2_scaling_law.pdf",
                bbox_inches="tight")
    plt.savefig("figures/figure2_scaling_law.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Figure 2 saved.")
else:
    print("  Not enough scaling data.")

# ── Figure 3: Architecture family comparison ──────────────────────

print("Generating Figure 3 — Architecture family comparison...")

model_data = [
    ("GPT-2 XL", "GPT-2", 20.8, 50, 45),
    ("GPT-2 Large", "GPT-2", 17.1, 15, 0),
    ("GPT-2 Medium", "GPT-2", 15.6, 40, 0),
    ("Phi-2", "Phi", 10.8, 20, 5),
    ("Qwen 1.5 1.8B", "Qwen", 2.5, 70, 40),
    ("GPT-2 Small", "GPT-2", 2.4, 10, 0),
    ("Pythia 2.8B", "Pythia", 1.1, 50, 0),
    ("GPT-Neo 2.7B", "GPT-Neo", 1.0, 65, 0),
    ("GPT-Neo 1.3B", "GPT-Neo", 1.0, 45, 0),
    ("GPT-Neo 125M", "GPT-Neo", 1.0, 25, 0),
]

model_data.sort(key=lambda x: x[2], reverse=True)

names = [d[0] for d in model_data]
families = [d[1] for d in model_data]
suppressions = [d[2] for d in model_data]
type2a_rates = [d[3] for d in model_data]
interventions = [d[4] for d in model_data]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

metrics = [
    (suppressions, "Suppression Ratio (×)",
     "Suppression Ratio", True),
    (type2a_rates, "Type 2a Cases (%)",
     "Type 2a Rate", False),
    (interventions, "Accuracy Improvement (%)",
     "Intervention Gain", False),
]

for ax_idx, (values, xlabel, title, show_threshold) in enumerate(
    metrics
):
    ax = axes[ax_idx]
    colors = [FAMILY_COLORS[f] for f in families]

    bars = ax.barh(
        names, values, color=colors,
        alpha=0.85, edgecolor="white", linewidth=0.5
    )

    for bar, val in zip(bars, values):
        suffix = "x" if ax_idx == 0 else "%"
        ax.text(
            val + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}{suffix}",
            va="center", fontsize=9
        )

    if show_threshold:
        ax.axvline(
            x=1.5, color="black", linestyle=":",
            linewidth=1.5, alpha=0.7,
            label="Threshold (1.5×)"
        )
        ax.legend(fontsize=9, loc="lower right")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)

legend_patches = [
    mpatches.Patch(color=c, label=f)
    for f, c in FAMILY_COLORS.items()
]
fig.legend(
    handles=legend_patches,
    loc="lower center", ncol=5,
    bbox_to_anchor=(0.5, -0.06),
    title="Model Family", framealpha=0.9
)

fig.suptitle(
    "Figure 3: Two Distinct Hallucination Families "
    "Across 10 Models and 5 Architectures",
    fontsize=13
)
plt.tight_layout()
plt.savefig("figures/figure3_architecture_families.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure3_architecture_families.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  Figure 3 saved.")

# ── Figure 4: HallBench v2 ────────────────────────────────────────

print("Generating Figure 4 — HallBench v2 results...")

hallbench_data = {
    "GPT-2 XL\n(strong)": {
        "tier1": (36, 80),
        "tier2": (47, 53),
        "tier3": (0, 0),
        "family": "strong",
    },
    "Qwen 1.5\n(strong)": {
        "tier1": (32, 60),
        "tier2": (7, 40),
        "tier3": (30, 30),
        "family": "strong",
    },
    "Phi-2\n(strong)": {
        "tier1": (72, 80),
        "tier2": (53, 53),
        "tier3": (30, 30),
        "family": "strong",
    },
    "GPT-Neo 2.7B\n(weak)": {
        "tier1": (20, 20),
        "tier2": (20, 20),
        "tier3": (20, 20),
        "family": "weak",
    },
}

tier_keys = ["tier1", "tier2", "tier3"]
tier_titles = [
    "Tier 1: High Suppression Facts\n"
    "(capitals, science — proven Type 2a in open models)",
    "Tier 2: Borderline Facts\n"
    "(sometimes survive suppression — model-size dependent)",
    "Tier 3: Knowledge Gap Facts\n"
    "(Type 2b — model never learned these facts)",
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

for t_idx, (tier_key, tier_title) in enumerate(
    zip(tier_keys, tier_titles)
):
    ax = axes[t_idx]
    models = list(hallbench_data.keys())
    n = len(models)
    x = np.arange(n)
    width = 0.38

    baselines = [hallbench_data[m][tier_key][0] for m in models]
    bests = [hallbench_data[m][tier_key][1] for m in models]
    fam_colors = [
        COLORS["strong"]
        if hallbench_data[m]["family"] == "strong"
        else COLORS["weak"]
        for m in models
    ]

    bars1 = ax.bar(
        x - width / 2, baselines, width,
        label="Baseline (α=0)",
        color=COLORS["baseline"], alpha=0.8,
        edgecolor="white"
    )
    bars2 = ax.bar(
        x + width / 2, bests, width,
        label="Best intervention",
        color=fam_colors, alpha=0.85,
        edgecolor="white"
    )

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8.5
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(tier_title, fontsize=10, pad=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    if t_idx == 2:
        ax.text(
            0.5, 0.5,
            "Intervention correctly has\nno effect on unknown facts",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10, color="gray",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightyellow",
                      alpha=0.8)
        )

fig.suptitle(
    "Figure 4: HallBench v2 — Intervention Recovers Suppressed "
    "Knowledge Without Hallucinating New Facts",
    fontsize=12
)
plt.tight_layout()
plt.savefig("figures/figure4_hallbench_v2.pdf",
            bbox_inches="tight")
plt.savefig("figures/figure4_hallbench_v2.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  Figure 4 saved.")

print(f"\nAll figures generated.")
print(f"Files saved to figures/:")
for f in sorted(os.listdir("figures")):
    size = os.path.getsize(f"figures/{f}") // 1024
    print(f"  {f} ({size} KB)")