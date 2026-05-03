"""
Experiment 7 — TruthfulQA Benchmark
=====================================
Goal: Validate logit blending intervention on a real benchmark.
Tests our intervention on questions we never used to develop it.
This is out-of-distribution validation.

Models: GPT-2 XL (strong suppression family)
        GPT-Neo 2.7B (weak suppression family)
        
Expected: GPT-2 XL improves with intervention
          GPT-Neo 2.7B does not improve
          
This confirms our architectural family finding on real data.
"""

import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

ALPHAS = [0.0, 0.3, 0.5]
N_SAMPLES = 200

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

def run_benchmark(model_name, params, n_samples=N_SAMPLES):
    print(f"\n{'='*70}")
    print(f"Loading {model_name}...")

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}")

    print(f"Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    samples = list(dataset)[:n_samples]
    print(f"Loaded {len(samples)} samples")

    alpha_correct = {a: 0 for a in ALPHAS}
    alpha_type2a = {a: 0 for a in ALPHAS}
    total = 0
    skipped = 0

    results = []

    for item in tqdm(samples, desc=f"Running {model_name}"):
        question = item["question"]
        correct_answers = item["correct_answers"]

        if not correct_answers:
            skipped += 1
            continue

        best_answer = correct_answers[0]
        words = best_answer.strip().split()
        if not words:
            skipped += 1
            continue

        first_word = words[0].rstrip(".,;:!?")
        if len(first_word) < 2:
            skipped += 1
            continue

        prompt = f"Q: {question}\nA:"

        try:
            token_id = get_token_id(model, first_word)
        except Exception:
            skipped += 1
            continue

        try:
            with torch.no_grad():
                final_logits, cache = model.run_with_cache(prompt)
        except Exception as e:
            skipped += 1
            continue

        final_probs = torch.softmax(final_logits[0, -1], dim=-1)
        correct_rank = (
            final_probs > final_probs[token_id]
        ).sum().item() + 1

        layer_probs = get_layer_probs(model, cache, token_id, n_layers)
        peak_layer = layer_probs.index(max(layer_probs))
        peak_prob = max(layer_probs)
        final_prob = layer_probs[-1]
        suppression_ratio = peak_prob / (final_prob + 1e-10)

        resid = cache[
            f"blocks.{peak_layer}.hook_resid_post"
        ][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        peak_logits = resid_normed @ model.W_U + model.b_U

        row = {
            "question": question,
            "correct_answer": best_answer,
            "first_word": first_word,
            "peak_layer": peak_layer,
            "suppression_ratio": round(suppression_ratio, 3),
            "predictions": {}
        }

        for alpha in ALPHAS:
            if alpha == 0.0:
                pred_logits = final_logits[0, -1]
            else:
                pred_logits = (
                    alpha * peak_logits +
                    (1 - alpha) * final_logits[0, -1]
                )

            pred = model.to_string(pred_logits.argmax()).strip()
            correct = (
                first_word.lower() in pred.lower() or
                pred.lower() in best_answer.lower()
            )

            if correct:
                alpha_correct[alpha] += 1
            if correct_rank <= 10:
                alpha_type2a[alpha] += 1

            row["predictions"][str(alpha)] = {
                "predicted": pred,
                "correct": correct,
            }

        results.append(row)
        total += 1

    print(f"\nProcessed: {total} | Skipped: {skipped}")
    print(f"\n{'Alpha':<8} {'Correct':<12} {'Accuracy':<12} {'vs Baseline'}")
    print("-" * 45)

    baseline_acc = alpha_correct[0.0] / total if total > 0 else 0

    for alpha in ALPHAS:
        acc = alpha_correct[alpha] / total if total > 0 else 0
        delta = acc - baseline_acc
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        best = " BEST" if alpha_correct[alpha] == max(
            alpha_correct.values()
        ) else ""
        print(f"a={alpha:<6} {alpha_correct[alpha]}/{total:<10} "
              f"{acc:.1%}       {delta_str}{best}")

    best_alpha = max(ALPHAS, key=lambda a: alpha_correct[a])
    best_acc = alpha_correct[best_alpha] / total if total > 0 else 0
    improvement = best_acc - baseline_acc

    output = {
        "model": model_name,
        "model_params": params,
        "n_layers": n_layers,
        "benchmark": "TruthfulQA",
        "n_samples": total,
        "timestamp": datetime.now().isoformat(),
        "baseline_accuracy": baseline_acc,
        "best_alpha": best_alpha,
        "best_accuracy": best_acc,
        "improvement": improvement,
        "alpha_results": {
            str(a): {
                "correct": alpha_correct[a],
                "accuracy": (
                    alpha_correct[a] / total if total > 0 else 0
                )
            }
            for a in ALPHAS
        },
        "results": results[:50],
    }

    filename = f"results/truthfulqa_{model_name.replace('/', '_')}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBaseline: {baseline_acc:.1%} | "
          f"Best: {best_acc:.1%} | "
          f"Improvement: +{improvement:.1%}")
    print(f"Saved to {filename}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "baseline": baseline_acc,
        "best": best_acc,
        "improvement": improvement,
    }

# ── Run both models ───────────────────────────────────────────────

all_results = []

r1 = run_benchmark("gpt2-xl", "1.5B")
all_results.append(r1)

r2 = run_benchmark("EleutherAI/gpt-neo-2.7B", "2.7B")
all_results.append(r2)

# ── Final comparison ──────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"TRUTHFULQA BENCHMARK FINAL COMPARISON")
print(f"{'='*70}")
print(f"\n{'Model':<25} {'Baseline':<12} {'Best':<12} {'Improvement'}")
print("-" * 55)

for r in all_results:
    model_short = r["model"].split("/")[-1]
    print(f"{model_short:<25} {r['baseline']:.1%}       "
          f"{r['best']:.1%}       +{r['improvement']:.1%}")

print(f"\nPredicted outcome:")
print(f"  GPT-2 XL (strong suppression):   intervention improves")
print(f"  GPT-Neo 2.7B (weak suppression): intervention no effect")

with open("results/truthfulqa_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved.")
print("Experiment 7 complete.")