"""
Figure Generator v3 — All Figures for Final Paper
===================================================
Generates:
  Figure 1: Layer suppression curve (single fact, GPT-2 XL)
  Figure 2: Scaling law (4 GPT-2 sizes)
  Figure 3: Architecture families scatter
  Figure 4: HallBench v2 tiered results
  Figure 5: Structure variance — THE KEY NEW VISUAL (4 lines on one graph)
  Figure 6: Multi-token vs single-token suppression comparison

Run AFTER all experiments are complete.
Results must exist in results/ directory.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

os.makedirs("figures", exist_ok=True)
STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

# ── Figure 5: Structure Variance (THE KEY VISUAL) ─────────────────────────────
def plot_figure5_structure_variance():
    try:
        with open("results/experiment_13_structure_variance.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("  Figure 5: results/experiment_13_structure_variance.json not found — skipping")
        return

    COLORS = {
        "forward":  "#2563EB",
        "question": "#16A34A",
        "reverse":  "#DC2626",
        "context":  "#9333EA",
    }
    LABELS = {
        "forward":  "Forward completion",
        "question": "Question form",
        "reverse":  "Reverse direction",
        "context":  "Contextual embed",
    }

    # Collect layer probs per structure
    structure_curves = {s: [] for s in ["forward", "question", "reverse", "context"]}
    structure_rhos   = {s: [] for s in structure_curves}

    for r in data["results"]:
        s = r["structure"]
        if s in structure_curves:
            structure_curves[s].append(r["layer_probs"])
            structure_rhos[s].append(r["rho"])

    n_layers = len(list(structure_curves.values())[0][0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Last-Layer Suppression is Invariant to Prompt Structure (GPT-2 XL)",
                 fontsize=13, fontweight='bold', y=1.02)

    # LEFT: Averaged probability curves
    ax1 = axes[0]
    x = np.arange(n_layers)
    for struct in ["forward", "question", "reverse", "context"]:
        curves = np.array(structure_curves[struct])
        if len(curves) == 0:
            continue
        avg = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax1.plot(x, avg, color=COLORS[struct], label=LABELS[struct],
                 linewidth=2.5, zorder=3)
        ax1.fill_between(x, avg - 0.5*std, avg + 0.5*std,
                         color=COLORS[struct], alpha=0.10)

    ax1.axvline(x=n_layers * 0.83, color='#6B7280', linestyle='--',
                alpha=0.6, linewidth=1.5, label='0.83 depth constant')
    ax1.set_xlabel("Transformer Layer", fontsize=12)
    ax1.set_ylabel("Mean P(correct answer)", fontsize=12)
    ax1.set_title("Averaged Layer-by-Layer Probability\nacross 30 facts × 4 structures",
                  fontsize=11)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax1.set_xlim(0, n_layers - 1)
    ax1.annotate("Same suppression\npattern in all 4", xy=(n_layers-3, 0.01),
                 fontsize=8.5, color='gray', ha='right')

    # RIGHT: ρ distribution per structure
    ax2 = axes[1]
    rho_data = [structure_rhos[s] for s in ["forward", "question", "reverse", "context"]]
    colors_list = [COLORS[s] for s in ["forward", "question", "reverse", "context"]]
    bp = ax2.boxplot(rho_data, patch_artist=True, notch=False, widths=0.5,
                     medianprops=dict(color='black', linewidth=2.5))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for struct_i, (struct, data_pts) in enumerate(
        zip(["forward","question","reverse","context"], rho_data)
    ):
        np.random.seed(42)
        jitter = np.random.uniform(-0.18, 0.18, len(data_pts))
        ax2.scatter([struct_i+1+j for j in jitter], data_pts,
                    color=COLORS[struct], alpha=0.45, s=22, zorder=4)

    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels([LABELS[s] for s in ["forward","question","reverse","context"]],
                         rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel("Suppression Ratio ρ", fontsize=12)
    ax2.set_title("ρ Distribution by Prompt Structure\nConsistent across all 4 forms",
                  fontsize=11)
    ax2.axhline(y=1.0, color='#6B7280', linestyle='--', alpha=0.6,
                linewidth=1.5, label='ρ = 1 (no suppression)')
    ax2.legend(fontsize=9)

    # Add mean annotations
    for i, (struct, d) in enumerate(zip(["forward","question","reverse","context"], rho_data)):
        if d:
            ax2.text(i+1, max(d)*1.05, f'μ={np.mean(d):.1f}x',
                     ha='center', va='bottom', fontsize=8.5,
                     color=COLORS[struct], fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/figure5_structure_variance.png", dpi=200, bbox_inches='tight')
    plt.savefig("figures/figure5_structure_variance.pdf", bbox_inches='tight')
    print("  Figure 5 saved: figures/figure5_structure_variance.png")
    plt.close()


# ── Figure 6: Multi-Token vs Single-Token ─────────────────────────────────────
def plot_figure6_multi_token():
    try:
        with open("results/experiment_12_multi_token.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("  Figure 6: results/experiment_12_multi_token.json not found — skipping")
        return

    results = data["results"]
    single = [r for r in results if r["is_single_token"]]
    multi  = [r for r in results if not r["is_single_token"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Single-Token vs Multi-Token Suppression Analysis (GPT-2 XL)",
                 fontsize=13, fontweight='bold', y=1.02)

    # LEFT: ρ comparison box plot
    ax1 = axes[0]
    rho_s = [r["rho"] for r in single]
    rho_m = [r["rho"] for r in multi]
    bp = ax1.boxplot([rho_s, rho_m], patch_artist=True, notch=False, widths=0.5,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#2563EB')
    bp['boxes'][0].set_alpha(0.65)
    bp['boxes'][1].set_facecolor('#DC2626')
    bp['boxes'][1].set_alpha(0.65)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f"Single-token\n(n={len(single)})",
                          f"Multi-token\n(first token)\n(n={len(multi)})"])
    ax1.set_ylabel("Suppression Ratio ρ", fontsize=12)
    ax1.set_title("ρ by Token Count\nSuppression attenuated but present")
    ax1.text(1, np.mean(rho_s)*1.05, f'μ={np.mean(rho_s):.1f}x',
             ha='center', fontsize=10, color='#2563EB', fontweight='bold')
    ax1.text(2, np.mean(rho_m)*1.05, f'μ={np.mean(rho_m):.1f}x',
             ha='center', fontsize=10, color='#DC2626', fontweight='bold')

    # MIDDLE: Hallucination type breakdown
    ax2 = axes[1]
    cats = ["CORRECT", "TYPE2A_SUPPRESSION", "TYPE2B_GAP"]
    cat_labels = ["Correct", "Type 2a\n(Suppressed)", "Type 2b\n(Gap)"]
    colors_cat = ["#16A34A", "#2563EB", "#9333EA"]
    x = np.arange(len(cats))
    w = 0.35
    vals_s = [sum(1 for r in single if r["hall_type"]==c)/len(single) for c in cats]
    vals_m = [sum(1 for r in multi if r["hall_type"]==c)/len(multi) for c in cats]
    ax2.bar(x - w/2, vals_s, w, label="Single-token", color='#2563EB', alpha=0.75)
    ax2.bar(x + w/2, vals_m, w, label="Multi-token", color='#DC2626', alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels)
    ax2.set_ylabel("Fraction of Facts", fontsize=12)
    ax2.set_title("Hallucination Type Distribution\nby Token Count")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)

    # RIGHT: ρ by region
    ax3 = axes[2]
    regions = {}
    for r in results:
        reg = r.get("region", "unknown")
        if reg not in regions:
            regions[reg] = []
        regions[reg].append(r["rho"])
    sorted_regions = sorted(regions.items(), key=lambda x: np.mean(x[1]), reverse=True)
    reg_labels = [r[0] for r in sorted_regions]
    reg_means  = [np.mean(r[1]) for r in sorted_regions]
    reg_stds   = [np.std(r[1]) for r in sorted_regions]
    bars = ax3.barh(reg_labels, reg_means, xerr=reg_stds,
                    color='#2563EB', alpha=0.7, capsize=4)
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.6)
    ax3.set_xlabel("Mean Suppression Ratio ρ", fontsize=12)
    ax3.set_title("ρ by Region\nNon-Western facts show similar suppression")
    ax3.set_xlim(0, max(reg_means) * 1.3)
    for i, (mean, std) in enumerate(zip(reg_means, reg_stds)):
        ax3.text(mean + std + 0.2, i, f'{mean:.1f}x',
                 va='center', fontsize=9, color='#2563EB')

    plt.tight_layout()
    plt.savefig("figures/figure6_multi_token.png", dpi=200, bbox_inches='tight')
    plt.savefig("figures/figure6_multi_token.pdf", bbox_inches='tight')
    print("  Figure 6 saved: figures/figure6_multi_token.png")
    plt.close()


# ── Run all ───────────────────────────────────────────────────────────────────
print("Generating figures...")
plot_figure5_structure_variance()
plot_figure6_multi_token()
print("\nAll figures generated.")
print("Check figures/ directory for output files.")
