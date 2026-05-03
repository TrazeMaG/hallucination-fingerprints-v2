import json
import os
import glob

results_dir = "results"

full_files = glob.glob(f"{results_dir}/full_*.json")

print(f"\n{'='*115}")
print(f"MASTER RESULTS TABLE — All Models")
print(f"{'='*115}")
print(f"{'Model':<25} {'Params':<8} {'Layers':<8} {'Correct':<10} "
      f"{'Type2a':<10} {'Type2b':<10} {'RelDepth':<10} "
      f"{'Suppression':<14} {'Intervention'}")
print("-" * 115)

all_results = []

# ── Process full_ files (new format) ─────────────────────────────

for filepath in sorted(full_files):
    with open(filepath) as f:
        data = json.load(f)

    model = data["model"].split("/")[-1]
    params = data.get("model_params", "?")
    layers = data["n_layers"]

    exp1 = data["experiment_1_layer_analysis"]["summary"]
    exp4 = data["experiment_4_intervention"]["summary"]

    correct_pct = exp1["correct"] / 20 * 100
    type2a_pct = exp1["type2a"] / 20 * 100
    type2b_pct = exp1["type2b"] / 20 * 100
    rel_depth = exp1.get("avg_rel_depth_type2a", 0)
    suppression = exp1.get("avg_suppression_type2a", 0)
    improvement = exp4["improvement"] * 100

    all_results.append({
        "model": model,
        "params": params,
        "layers": layers,
        "correct": correct_pct,
        "type2a": type2a_pct,
        "type2b": type2b_pct,
        "rel_depth": rel_depth,
        "suppression": suppression,
        "intervention": improvement,
    })

# ── Process old format GPT-2 files ───────────────────────────────

gpt2_models = [
    ("experiment_01_layer_analysis.json", "gpt2", "124M", 12, 0.0),
    ("experiment_01_gpt2_medium.json", "gpt2-medium", "345M", 24, 0.0),
    ("experiment_01_gpt2_large.json", "gpt2-large", "774M", 36, 0.0),
    ("experiment_01_gpt2_xl.json", "gpt2-xl", "1.5B", 48, 45.0),
    ("experiment_02_gpt_neo_125m.json", "gpt-neo-125M", "125M", 12, 0.0),
    ("experiment_02_gpt_neo_1b3.json", "gpt-neo-1.3B", "1.3B", 24, 0.0),
]

for filename, model_name, params, layers, intervention in gpt2_models:
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        continue

    with open(filepath) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        continue

    type2a_keys = [
        "TYPE2A_LAST_LAYER_SUPPRESSION",
        "TYPE2A_SUPPRESSION"
    ]
    type2b_keys = [
        "TYPE2B_KNOWLEDGE_GAP",
        "TYPE2B_GAP"
    ]

    n_correct = sum(1 for r in results
                    if r["hallucination_type"] == "CORRECT")
    n_2a = sum(1 for r in results
               if r["hallucination_type"] in type2a_keys)
    n_2b = sum(1 for r in results
               if r["hallucination_type"] in type2b_keys)

    type2a_cases = [r for r in results
                    if r["hallucination_type"] in type2a_keys]

    rel_key = None
    supp_key = None
    if type2a_cases:
        sample = type2a_cases[0]
        rel_key = ("peak_layer_relative" if "peak_layer_relative" in sample
                   else "relative_depth" if "relative_depth" in sample
                   else None)
        supp_key = ("suppression_ratio" if "suppression_ratio" in sample
                    else None)

    rel_depth = 0
    suppression = 0
    if type2a_cases and rel_key:
        rel_depth = (sum(r.get(rel_key, 0) for r in type2a_cases) /
                     len(type2a_cases))
    if type2a_cases and supp_key:
        suppression = (sum(r.get(supp_key, 0) for r in type2a_cases) /
                       len(type2a_cases))

    total = len(results)
    all_results.append({
        "model": model_name,
        "params": params,
        "layers": layers,
        "correct": n_correct / total * 100,
        "type2a": n_2a / total * 100,
        "type2b": n_2b / total * 100,
        "rel_depth": rel_depth,
        "suppression": suppression,
        "intervention": intervention,
    })

# ── Sort and print ────────────────────────────────────────────────

all_results.sort(key=lambda x: x["suppression"], reverse=True)

for r in all_results:
    print(f"{r['model']:<25} {r['params']:<8} {r['layers']:<8} "
          f"{r['correct']:.0f}%{'':<7} "
          f"{r['type2a']:.0f}%{'':<7} "
          f"{r['type2b']:.0f}%{'':<7} "
          f"{r['rel_depth']:.3f}{'':<6} "
          f"{r['suppression']:.1f}x{'':<10} "
          f"+{r['intervention']:.0f}%")

# ── Family Analysis ───────────────────────────────────────────────

print(f"\n{'='*115}")
print(f"FAMILY ANALYSIS")
print(f"{'='*115}")

strong = [r for r in all_results if r["suppression"] > 1.5]
weak = [r for r in all_results if r["suppression"] <= 1.5]

print(f"\nStrong Suppression Family ({len(strong)} models):")
for r in strong:
    print(f"  {r['model']:<25} params={r['params']:<8} "
          f"suppression={r['suppression']:.1f}x  "
          f"intervention=+{r['intervention']:.0f}%")

print(f"\nWeak Suppression Family ({len(weak)} models):")
for r in weak:
    print(f"  {r['model']:<25} params={r['params']:<8} "
          f"suppression={r['suppression']:.1f}x  "
          f"intervention=+{r['intervention']:.0f}%")

avg_strong = (
    sum(r["intervention"] for r in strong) / len(strong)
    if strong else 0
)
avg_weak = (
    sum(r["intervention"] for r in weak) / len(weak)
    if weak else 0
)

print(f"\nKey numbers:")
print(f"  Total models tested:            {len(all_results)}")
print(f"  Strong suppression family:      {len(strong)} models")
print(f"  Weak suppression family:        {len(weak)} models")
print(f"  Strong family avg intervention: +{avg_strong:.1f}%")
print(f"  Weak family avg intervention:   +{avg_weak:.1f}%")
print(f"  Max suppression ratio:          "
      f"{max(r['suppression'] for r in all_results):.1f}x")
print(f"  Max intervention improvement:   "
      f"+{max(r['intervention'] for r in all_results):.0f}%")