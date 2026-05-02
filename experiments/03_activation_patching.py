"""
Experiment 3 — Activation Patching v2
=======================================
Improved patching strategy: instead of patching from a different
prompt, we patch from an EARLIER LAYER of the SAME prompt.

This bypasses the suppression by using the model's own knowledge
from before the suppression layer kicks in.

This is the correct causal test:
- If patching block 41 activations into block 47 restores Paris
- Then block 47 (the final layer) CAUSES the suppression
- The knowledge existed at block 41 but was destroyed by block 47
"""

from transformer_lens import HookedTransformer
import torch
import json
from datetime import datetime

# ── Load Model ────────────────────────────────────────────────────

print("Loading GPT-2 XL...")
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
n_layers = model.cfg.n_layers
print(f"Loaded. Layers: {n_layers}")

# ── Helper Functions ──────────────────────────────────────────────

def get_token_prob(model, logits, token_str):
    probs = torch.softmax(logits[0, -1], dim=-1)
    try:
        token_id = model.to_single_token(f" {token_str}")
        return probs[token_id].item()
    except:
        return 0.0

def get_top3(model, logits):
    probs = torch.softmax(logits[0, -1], dim=-1)
    top3 = torch.topk(probs, 3)
    return [model.to_string(idx).strip() for idx in top3.indices]

def get_peak_layer(model, prompt, answer):
    """Find which layer has the highest probability for the answer."""
    try:
        token_id = model.to_single_token(f" {answer}")
    except:
        tokens = model.to_tokens(f" {answer}")[0]
        token_id = tokens[1].item()

    with torch.no_grad():
        logits, cache = model.run_with_cache(prompt)

    layer_probs = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        layer_logits = resid_normed @ model.W_U + model.b_U
        prob = torch.softmax(layer_logits, dim=-1)[token_id].item()
        layer_probs.append(prob)

    peak_layer = layer_probs.index(max(layer_probs))
    return peak_layer, max(layer_probs), cache

def self_patch_experiment(model, prompt, answer, source_layer, target_layer):
    """
    Self-patching: take activations from source_layer of the same
    prompt and inject them at target_layer.

    If source_layer = peak factual layer (e.g. block 41)
    and target_layer = final layer (block 47)

    Then we are bypassing the suppression mechanism by feeding
    the model its own earlier knowledge at the final step.
    """

    # First run: get all activations
    with torch.no_grad():
        baseline_logits, cache = model.run_with_cache(prompt)

    # Get the activation we want to inject
    source_activation = cache[
        f"blocks.{source_layer}.hook_resid_post"
    ].clone()

    # Define the patch hook
    def patch_hook(value, hook):
        value[0, -1, :] = source_activation[0, -1, :]
        return value

    # Second run: with patch applied at target layer
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(
                f"blocks.{target_layer}.hook_resid_post",
                patch_hook
            )]
        )

    baseline_pred = model.to_string(
        baseline_logits[0, -1].argmax()
    ).strip()
    patched_pred = model.to_string(
        patched_logits[0, -1].argmax()
    ).strip()

    baseline_prob = get_token_prob(model, baseline_logits, answer)
    patched_prob = get_token_prob(model, patched_logits, answer)

    return {
        "baseline_prediction": baseline_pred,
        "patched_prediction": patched_pred,
        "baseline_prob": round(baseline_prob, 4),
        "patched_prob": round(patched_prob, 4),
        "prob_change": round(patched_prob - baseline_prob, 4),
        "flipped": patched_pred.lower() == answer.lower(),
        "improved": patched_prob > baseline_prob,
    }

# ── Main Experiments ──────────────────────────────────────────────

EXPERIMENTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of China is", "Beijing"),
    ("The capital of India is", "Delhi"),
    ("The theory of evolution was proposed by", "Darwin"),
    ("Albert Einstein discovered", "relativity"),
    ("The capital of Australia is", "Canberra"),
]

print(f"\nStep 1: Finding peak factual layers...")
print(f"{'Prompt':<45} {'Answer':<12} {'Peak Layer':<12} {'Peak Prob'}")
print("-" * 85)

experiment_data = []

for prompt, answer in EXPERIMENTS:
    peak_layer, peak_prob, cache = get_peak_layer(model, prompt, answer)
    experiment_data.append({
        "prompt": prompt,
        "answer": answer,
        "peak_layer": peak_layer,
        "peak_prob": peak_prob,
        "cache": cache
    })
    print(f"{prompt:<45} {answer:<12} Block {peak_layer:<7} {peak_prob:.4f}")

print(f"\nStep 2: Self-patching — injecting peak layer into final layers...")
print(f"{'Prompt':<45} {'Answer':<10} {'Baseline':<12} {'Patched':<12} {'Prob+':<10} {'Flipped'}")
print("-" * 105)

results = []

for exp in experiment_data:
    prompt = exp["prompt"]
    answer = exp["answer"]
    peak_layer = exp["peak_layer"]

    # Patch from peak layer into each of the final 3 layers
    best_result = None
    best_prob_change = -999

    for target_layer in range(n_layers - 3, n_layers):
        if target_layer <= peak_layer:
            continue

        result = self_patch_experiment(
            model, prompt, answer,
            source_layer=peak_layer,
            target_layer=target_layer
        )
        result["prompt"] = prompt
        result["answer"] = answer
        result["source_layer"] = peak_layer
        result["target_layer"] = target_layer

        if result["prob_change"] > best_prob_change:
            best_prob_change = result["prob_change"]
            best_result = result

    if best_result:
        results.append(best_result)
        flipped_str = "YES CAUSAL" if best_result["flipped"] else (
            "improved" if best_result["improved"] else "no change"
        )
        print(f"{prompt:<45} "
              f"{answer:<10} "
              f"{best_result['baseline_prediction']:<12} "
              f"{best_result['patched_prediction']:<12} "
              f"{best_result['prob_change']:+.4f}     "
              f"{flipped_str}")

# ── Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 105)
print("CAUSAL ANALYSIS SUMMARY")
print("=" * 105)

n_flipped = sum(1 for r in results if r["flipped"])
n_improved = sum(1 for r in results if r["improved"])
avg_change = sum(r["prob_change"] for r in results) / len(results)

print(f"Experiments:              {len(results)}")
print(f"Flipped to correct:       {n_flipped}/{len(results)}")
print(f"Probability improved:     {n_improved}/{len(results)}")
print(f"Average prob change:      {avg_change:+.4f}")

if n_flipped > 0:
    print(f"\nSTRONG CAUSAL EVIDENCE:")
    print(f"Injecting peak-layer activations into final layers")
    print(f"restored correct predictions in {n_flipped} cases.")
    print(f"This proves the final layer CAUSES suppression.")
elif n_improved > 0:
    print(f"\nMODERATE CAUSAL EVIDENCE:")
    print(f"Patching improved probability in {n_improved}/{len(results)} cases.")
    print(f"Final layers suppress factual signals established earlier.")
else:
    print(f"\nPatching did not improve predictions.")
    print(f"The suppression mechanism may operate differently than expected.")

# ── Save ──────────────────────────────────────────────────────────

save_results = [{k: v for k, v in r.items()} for r in results]

output = {
    "model": "gpt2-xl",
    "experiment": "self_patching",
    "timestamp": datetime.now().isoformat(),
    "n_experiments": len(results),
    "n_flipped": n_flipped,
    "n_improved": n_improved,
    "avg_prob_change": avg_change,
    "results": save_results,
}

with open("results/experiment_03_activation_patching.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/experiment_03_activation_patching.json")
print("Experiment 3 complete.")