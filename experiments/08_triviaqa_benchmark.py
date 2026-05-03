"""
Experiment 8 — TriviaQA Benchmark
===================================
TriviaQA is a more natural factual QA benchmark than our
20-prompt capitals dataset. 10,000+ real trivia questions.
This is the strongest out-of-distribution validation.
"""

import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

ALPHAS = [0.0, 0.1, 0.3, 0.5]
N_SAMPLES = 300

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

def run_benchmark(model_name, params):
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

    print(f"Loading TriviaQA...")
    dataset = load_dataset(
        "trivia_qa", "rc.nocontext", split=f"validation[:{N_SAMPLES}]"
    )
    print(f"Loaded {len(dataset)} samples")

    alpha_correct = {a: 0 for a in ALPHAS}
    total = 0
    skipped = 0
    results = []

    for item in tqdm(dataset, desc=f"Running {model_name}"):
        question = item["question"]
        answers = item["answer"]["aliases"]

        if not answers:
            skipped += 1
            continue

        best_answer = answers[0]
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
        except Exception:
            skipped += 1
            continue

        layer_probs = get_layer_probs(model, cache, token_id, n_layers)
        peak_layer = layer_probs.index(max(layer_probs))

        resid = cache[
            f"blocks.{peak_layer}.hook_resid_post"
        ][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        peak_logits = resid_normed @ model.W_U + model.b_U

        row = {"question": question, "answer": best_answer, "predictions": {}}

        for alpha in ALPHAS:
            if alpha == 0.0:
                pred_logits = final_logits[0, -1]
            else:
                pred_logits = (
                    alpha * peak_logits +
                    (1 - alpha) * final_logits[0, -1]
                )

            pred = model.to_string(pred_logits.argmax()).strip()
            correct = any(
                a.lower()[:6] in pred.lower()
                for a in answers
            )

            if correct:
                alpha_correct[alpha] += 1

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
        "benchmark": "TriviaQA",
        "n_samples": total,
        "timestamp": datetime.now().isoformat(),
        "baseline_accuracy": baseline_acc,
        "best_alpha": best_alpha,
        "best_accuracy": best_acc,
        "improvement": improvement,
        "alpha_results": {
            str(a): {
                "correct": alpha_correct[a],
                "accuracy": alpha_correct[a] / total if total > 0 else 0
            }
            for a in ALPHAS
        },
    }

    filename = f"results/triviaqa_{model_name.replace('/', '_')}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBaseline: {baseline_acc:.1%} | Best: {best_acc:.1%} | "
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

all_results = []
r1 = run_benchmark("gpt2-xl", "1.5B")
all_results.append(r1)
r2 = run_benchmark("EleutherAI/gpt-neo-2.7B", "2.7B")
all_results.append(r2)

print(f"\n{'='*70}")
print(f"TRIVIAQA FINAL COMPARISON")
print(f"{'='*70}")
print(f"\n{'Model':<25} {'Baseline':<12} {'Best':<12} {'Improvement'}")
print("-" * 55)
for r in all_results:
    print(f"{r['model'].split('/')[-1]:<25} {r['baseline']:.1%}"
          f"       {r['best']:.1%}       +{r['improvement']:.1%}")

with open("results/triviaqa_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nExperiment 8 complete.")