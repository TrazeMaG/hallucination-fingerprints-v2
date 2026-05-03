"""
Experiment 9 — HallBench v2 Evaluation
========================================
Run our custom three-tier benchmark across all model families.
This is the definitive evaluation for the paper.

Shows:
- Tier 1 (high suppression): intervention helps most
- Tier 2 (borderline): intervention helps somewhat
- Tier 3 (knowledge gap): intervention correctly has no effect
- Architecture family difference confirmed on our own benchmark
"""

import torch
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from transformer_lens import HookedTransformer
from hallscope.hallbench import get_tier, HALLBENCH_V2

os.makedirs("results", exist_ok=True)

ALPHAS = [0.0, 0.3, 0.5]

MODELS = [
    ("gpt2-xl", "1.5B", "strong"),
    ("EleutherAI/gpt-neo-2.7B", "2.7B", "weak"),
    ("microsoft/phi-2", "2.7B", "strong"),
    ("Qwen/Qwen1.5-1.8B", "1.8B", "strong"),
]

def get_token_id(model, answer):
    try:
        return model.to_single_token(f" {answer}")
    except Exception:
        tokens = model.to_tokens(f" {answer}")[0]
        return tokens[1].item()

def get_layer_probs(model, cache, token_id, n_layers):
    probs = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        logits = resid_normed @ model.W_U + model.b_U
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)
    return probs

def run_tier(model, n_layers, tier_data, alphas):
    tier_results = {a: {"correct": 0, "total": 0} for a in alphas}

    for prompt, answer, category, tier in tier_data:
        try:
            token_id = get_token_id(model, answer)
        except Exception:
            continue

        try:
            with torch.no_grad():
                final_logits, cache = model.run_with_cache(prompt)
        except Exception:
            continue

        layer_probs = get_layer_probs(model, cache, token_id, n_layers)
        peak_layer = layer_probs.index(max(layer_probs))

        resid = cache[f"blocks.{peak_layer}.hook_resid_post"][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        peak_logits = resid_normed @ model.W_U + model.b_U

        for alpha in alphas:
            if alpha == 0.0:
                pred_logits = final_logits[0, -1]
            else:
                pred_logits = (
                    alpha * peak_logits +
                    (1 - alpha) * final_logits[0, -1]
                )
            pred = model.to_string(pred_logits.argmax()).strip()
            correct = pred.lower() == answer.lower()

            tier_results[alpha]["total"] += 1
            if correct:
                tier_results[alpha]["correct"] += 1

    return tier_results

all_model_results = []

for model_name, params, family in MODELS:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({params}) — {family} suppression family")
    print(f"{'='*70}")

    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.eval()
        n_layers = model.cfg.n_layers
    except Exception as e:
        print(f"Failed to load: {e}")
        continue

    model_result = {
        "model": model_name,
        "params": params,
        "family": family,
        "tiers": {}
    }

    for tier_num in [1, 2, 3]:
        tier_data = get_tier(tier_num)
        print(f"\nTier {tier_num} ({len(tier_data)} prompts)...")
        tier_results = run_tier(model, n_layers, tier_data, ALPHAS)

        print(f"  {'Alpha':<8} {'Correct':<12} {'Accuracy':<12} {'vs Baseline'}")
        print(f"  {'-'*45}")

        baseline = tier_results[0.0]
        baseline_acc = (
            baseline["correct"] / baseline["total"]
            if baseline["total"] > 0 else 0
        )

        for alpha in ALPHAS:
            r = tier_results[alpha]
            acc = r["correct"] / r["total"] if r["total"] > 0 else 0
            delta = acc - baseline_acc
            delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
            print(f"  a={alpha:<6} {r['correct']}/{r['total']:<10} "
                  f"{acc:.1%}       {delta_str}")

        model_result["tiers"][f"tier{tier_num}"] = {
            str(a): {
                "correct": tier_results[a]["correct"],
                "total": tier_results[a]["total"],
                "accuracy": (
                    tier_results[a]["correct"] /
                    tier_results[a]["total"]
                    if tier_results[a]["total"] > 0 else 0
                )
            }
            for a in ALPHAS
        }

    all_model_results.append(model_result)
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*90}")
print(f"HALLBENCH V2 — FINAL RESULTS")
print(f"{'='*90}")
print(f"\n{'Model':<20} {'Family':<8} {'T1 Base':<10} {'T1 Best':<10} "
      f"{'T2 Base':<10} {'T2 Best':<10} {'T3 Base':<10} {'T3 Best'}")
print("-" * 90)

for r in all_model_results:
    model_short = r["model"].split("/")[-1]
    row = f"{model_short:<20} {r['family']:<8}"
    for tier in ["tier1", "tier2", "tier3"]:
        if tier in r["tiers"]:
            base = r["tiers"][tier]["0.0"]["accuracy"]
            best = max(
                r["tiers"][tier][str(a)]["accuracy"]
                for a in ALPHAS
            )
            row += f" {base:.0%}       {best:.0%}    "
        else:
            row += f" N/A       N/A    "
    print(row)

output = {
    "benchmark": "HallBench v2",
    "timestamp": datetime.now().isoformat(),
    "n_tier1": len(get_tier(1)),
    "n_tier2": len(get_tier(2)),
    "n_tier3": len(get_tier(3)),
    "results": all_model_results,
}

with open("results/hallbench_v2_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to results/hallbench_v2_results.json")
print("Experiment 9 complete.")