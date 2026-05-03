"""
Master Experiment Runner
========================
Runs all experiments on any model with one command.

Usage:
  python run_experiment.py --model EleutherAI/pythia-2.8b --params 2.8B
  python run_experiment.py --model mistralai/Mistral-7B-v0.1 --params 7B
  python run_experiment.py --model EleutherAI/gpt-neo-2.7B --params 2.7B
  python run_experiment.py --model Qwen/Qwen1.5-1.8B --params 1.8B
"""

import argparse
import torch
import json
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
from transformer_lens import HookedTransformer

# ── Argument Parser ───────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--params", type=str, default="unknown")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PARAMS = args.params
MODEL_SAFE = MODEL_NAME.replace("/", "_").replace("-", "_")

os.makedirs("results", exist_ok=True)

# ── Prompts ───────────────────────────────────────────────────────

PROMPTS = [
    ("The capital of France is", "Paris", "capitals"),
    ("The capital of Germany is", "Berlin", "capitals"),
    ("The capital of Japan is", "Tokyo", "capitals"),
    ("The capital of Italy is", "Rome", "capitals"),
    ("The capital of Spain is", "Madrid", "capitals"),
    ("The capital of Australia is", "Canberra", "capitals"),
    ("The capital of Brazil is", "Brasilia", "capitals"),
    ("The capital of China is", "Beijing", "capitals"),
    ("The capital of India is", "Delhi", "capitals"),
    ("The capital of Russia is", "Moscow", "capitals"),
    ("The capital of Canada is", "Ottawa", "capitals"),
    ("The capital of Argentina is", "Buenos", "capitals"),
    ("The Berlin Wall fell in", "1989", "history"),
    ("Water is made of hydrogen and", "oxygen", "science"),
    ("The speed of light is approximately", "299", "science"),
    ("Albert Einstein discovered", "relativity", "science"),
    ("Shakespeare wrote", "Hamlet", "literature"),
    ("The first president of the United States was", "Washington", "history"),
    ("The theory of evolution was proposed by", "Darwin", "science"),
    ("The chemical symbol for gold is", "Au", "science"),
]

PATCH_EXPERIMENTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of China is", "Beijing"),
    ("The theory of evolution was proposed by", "Darwin"),
]

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ── Load Model ────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"MODEL: {MODEL_NAME}")
print(f"{'='*60}")
print(f"Loading...")

try:
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
except Exception as e:
    print(f"Failed to load {MODEL_NAME}: {e}")
    exit(1)

n_layers = model.cfg.n_layers
print(f"Loaded. Layers: {n_layers} | "
      f"Heads: {model.cfg.n_heads} | "
      f"Dim: {model.cfg.d_model}")

# ── Helper Functions ──────────────────────────────────────────────

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


def get_token_prob(model, logits, answer):
    probs = torch.softmax(logits[0, -1], dim=-1)
    try:
        token_id = model.to_single_token(f" {answer}")
        return probs[token_id].item()
    except Exception:
        return 0.0


# ── Experiment 1: Layer Analysis ──────────────────────────────────

print(f"\n--- EXPERIMENT 1: Layer Analysis ---")
print(f"{'Prompt':<45} {'Predicted':<12} {'Type':<28} "
      f"{'Peak':<8} {'Rel':<6} {'Ratio'}")
print("-" * 105)

layer_results = []

for prompt, answer, category in PROMPTS:
    token_id = get_token_id(model, answer)

    with torch.no_grad():
        logits, cache = model.run_with_cache(prompt)

    final_probs = torch.softmax(logits[0, -1], dim=-1)
    predicted = model.to_string(logits[0, -1].argmax()).strip()
    correct_rank = (
        final_probs > final_probs[token_id]
    ).sum().item() + 1

    layer_probs = get_layer_probs(model, cache, token_id, n_layers)
    peak_layer = layer_probs.index(max(layer_probs))
    peak_prob = max(layer_probs)
    final_prob = layer_probs[-1]
    suppression_ratio = peak_prob / (final_prob + 1e-10)
    relative_depth = peak_layer / n_layers

    is_correct = predicted.lower() == answer.lower()

    if is_correct:
        hall_type = "CORRECT"
    elif correct_rank <= 10:
        hall_type = "TYPE2A_SUPPRESSION"
    else:
        hall_type = "TYPE2B_GAP"

    layer_results.append({
        "prompt": prompt,
        "correct_answer": answer,
        "predicted": predicted,
        "is_correct": is_correct,
        "hallucination_type": hall_type,
        "peak_layer": peak_layer,
        "peak_layer_relative": round(relative_depth, 3),
        "peak_prob": round(peak_prob, 4),
        "final_prob": round(final_prob, 4),
        "suppression_ratio": round(suppression_ratio, 2),
        "correct_final_rank": correct_rank,
        "category": category,
        "layer_probs": [round(p, 4) for p in layer_probs],
    })

    print(f"{prompt:<45} {predicted:<12} {hall_type:<28} "
          f"Block {peak_layer:<3} {relative_depth:.3f}  "
          f"{suppression_ratio:.1f}x")

n_correct = sum(1 for r in layer_results
                if r["hallucination_type"] == "CORRECT")
n_2a = sum(1 for r in layer_results
           if r["hallucination_type"] == "TYPE2A_SUPPRESSION")
n_2b = sum(1 for r in layer_results
           if r["hallucination_type"] == "TYPE2B_GAP")

type2a_cases = [r for r in layer_results
                if r["hallucination_type"] == "TYPE2A_SUPPRESSION"]
avg_rel = (
    sum(r["peak_layer_relative"] for r in type2a_cases) /
    len(type2a_cases) if type2a_cases else 0
)
avg_supp = (
    sum(r["suppression_ratio"] for r in type2a_cases) /
    len(type2a_cases) if type2a_cases else 0
)

print(f"\nSUMMARY: Correct={n_correct} ({n_correct/20*100:.0f}%) | "
      f"Type2a={n_2a} ({n_2a/20*100:.0f}%) | "
      f"Type2b={n_2b} ({n_2b/20*100:.0f}%)")
if type2a_cases:
    print(f"Type2a avg rel depth: {avg_rel:.3f} | "
          f"avg suppression: {avg_supp:.1f}x")

# ── Experiment 3: Activation Patching ────────────────────────────

print(f"\n--- EXPERIMENT 3: Activation Patching ---")
print(f"{'Prompt':<45} {'Answer':<12} {'Baseline':<12} "
      f"{'Patched':<12} {'Flipped'}")
print("-" * 95)

patch_results = []

for prompt, answer in PATCH_EXPERIMENTS:
    token_id = get_token_id(model, answer)

    with torch.no_grad():
        baseline_logits, cache = model.run_with_cache(prompt)

    layer_probs = get_layer_probs(model, cache, token_id, n_layers)
    peak_layer = layer_probs.index(max(layer_probs))

    baseline_pred = model.to_string(
        baseline_logits[0, -1].argmax()
    ).strip()

    if peak_layer >= n_layers - 2:
        source_layer = n_layers // 2
    else:
        source_layer = peak_layer

    peak_activation = cache[
        f"blocks.{source_layer}.hook_resid_post"
    ].clone()

    def patch_hook(value, hook):
        value[0, -1, :] = peak_activation[0, -1, :]
        return value

    best_result = None
    best_prob = -1

    target_layers = list(range(source_layer + 1, n_layers))
    if not target_layers:
        target_layers = [n_layers - 1]

    for target in target_layers:
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                prompt,
                fwd_hooks=[(
                    f"blocks.{target}.hook_resid_post",
                    patch_hook
                )]
            )

        patched_pred = model.to_string(
            patched_logits[0, -1].argmax()
        ).strip()
        patched_prob = get_token_prob(model, patched_logits, answer)

        if patched_prob > best_prob:
            best_prob = patched_prob
            best_result = {
                "patched_pred": patched_pred,
                "target_layer": target,
                "patched_prob": patched_prob,
            }

    if best_result is None:
        best_result = {
            "patched_pred": baseline_pred,
            "target_layer": n_layers - 1,
            "patched_prob": 0.0,
        }

    flipped = best_result["patched_pred"].lower() == answer.lower()
    flipped_str = "YES CAUSAL" if flipped else "no"

    patch_results.append({
        "prompt": prompt,
        "answer": answer,
        "baseline_pred": baseline_pred,
        "patched_pred": best_result["patched_pred"],
        "source_layer": source_layer,
        "target_layer": best_result["target_layer"],
        "flipped": flipped,
        "patched_prob": round(best_result["patched_prob"], 4),
    })

    print(f"{prompt:<45} {answer:<12} {baseline_pred:<12} "
          f"{best_result['patched_pred']:<12} {flipped_str}")

n_flipped = sum(1 for r in patch_results if r["flipped"])
print(f"\nPatching result: {n_flipped}/{len(patch_results)} "
      f"flipped to correct")

# ── Experiment 4: Logit Blending Intervention ─────────────────────

print(f"\n--- EXPERIMENT 4: Logit Blending Intervention ---")
print(f"{'Alpha':<8} {'Correct':<10} {'Accuracy':<12} {'vs Baseline'}")
print("-" * 40)

alpha_correct = {a: 0 for a in ALPHAS}

for prompt, answer, category in PROMPTS:
    token_id = get_token_id(model, answer)

    with torch.no_grad():
        final_logits, cache = model.run_with_cache(prompt)

    layer_probs = get_layer_probs(model, cache, token_id, n_layers)
    peak_layer = layer_probs.index(max(layer_probs))

    resid = cache[
        f"blocks.{peak_layer}.hook_resid_post"
    ][0, -1]
    resid_normed = model.ln_final(resid.unsqueeze(0))[0]
    peak_logits = resid_normed @ model.W_U + model.b_U

    for alpha in ALPHAS:
        if alpha == 0.0:
            pred_logits = final_logits[0, -1]
        else:
            pred_logits = (
                alpha * peak_logits +
                (1 - alpha) * final_logits[0, -1]
            )

        pred = model.to_string(pred_logits.argmax()).strip()
        if pred.lower() == answer.lower():
            alpha_correct[alpha] += 1

baseline_acc = alpha_correct[0.0] / len(PROMPTS)
max_correct = max(alpha_correct.values())

for alpha in ALPHAS:
    acc = alpha_correct[alpha] / len(PROMPTS)
    delta = acc - baseline_acc
    delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
    best_mark = " BEST" if alpha_correct[alpha] == max_correct else ""
    print(f"α={alpha:<6} {alpha_correct[alpha]}/{len(PROMPTS):<8} "
          f"{acc:.1%}        {delta_str}{best_mark}")

best_alpha = max(ALPHAS, key=lambda a: alpha_correct[a])
best_acc = alpha_correct[best_alpha] / len(PROMPTS)
improvement = best_acc - baseline_acc

print(f"\nBaseline: {baseline_acc:.1%} | Best: {best_acc:.1%} | "
      f"Improvement: +{improvement:.1%} at α={best_alpha}")

# ── Save All Results ──────────────────────────────────────────────

output = {
    "model": MODEL_NAME,
    "model_params": MODEL_PARAMS,
    "n_layers": n_layers,
    "n_heads": model.cfg.n_heads,
    "d_model": model.cfg.d_model,
    "timestamp": datetime.now().isoformat(),
    "experiment_1_layer_analysis": {
        "summary": {
            "correct": n_correct,
            "type2a": n_2a,
            "type2b": n_2b,
            "avg_rel_depth_type2a": round(avg_rel, 3),
            "avg_suppression_type2a": round(avg_supp, 2),
        },
        "results": layer_results,
    },
    "experiment_3_patching": {
        "summary": {
            "n_flipped": n_flipped,
            "total": len(patch_results),
        },
        "results": patch_results,
    },
    "experiment_4_intervention": {
        "summary": {
            "baseline_accuracy": baseline_acc,
            "best_alpha": best_alpha,
            "best_accuracy": best_acc,
            "improvement": improvement,
        },
        "alpha_results": {
            str(a): {
                "correct": alpha_correct[a],
                "accuracy": alpha_correct[a] / len(PROMPTS)
            }
            for a in ALPHAS
        },
    },
}

filename = f"results/full_{MODEL_SAFE}.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nAll results saved to {filename}")
print(f"\n{'='*60}")
print(f"COMPLETE: {MODEL_NAME}")
print(f"Correct: {n_correct/20*100:.0f}% | "
      f"Type2a: {n_2a/20*100:.0f}% | "
      f"Intervention: +{improvement:.1%}")
print(f"{'='*60}\n")