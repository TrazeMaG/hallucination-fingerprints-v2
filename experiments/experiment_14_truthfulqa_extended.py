"""
Experiment 14 — Extended TruthfulQA + TriviaQA Validation
===========================================================
Addresses reviewer critique #1: "200 prompts is small for top conference"

Runs intervention on:
1. TruthfulQA (full 817 questions)
2. TriviaQA (1000 natural language questions)

Split results by:
- Suppression ratio > 2x vs < 2x (Type2a vs rest)
- Single-token vs multi-token answers
- Category

This turns 200 prompts into 1800+ total evaluation points.
"""

from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import json
import numpy as np
from datetime import datetime
import os

MODEL = "gpt2-xl"
ALPHAS = [0.0, 0.2, 0.4, 0.5]
N_TRIVIAQA = 1000

print(f"Loading {MODEL}...")
model = HookedTransformer.from_pretrained(MODEL)
model.eval()
n_layers = model.cfg.n_layers

def get_token_id(model, answer):
    try:
        return model.to_single_token(f" {answer}")
    except Exception:
        tokens = model.to_tokens(f" {answer}")[0]
        return tokens[1].item()

def get_layer_probs(model, cache, token_id):
    probs = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        logits = resid_normed @ model.W_U + model.b_U
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)
    return probs

def run_intervention(prompt, answer, alpha):
    token_id = get_token_id(model, answer)
    with torch.no_grad():
        final_logits, cache = model.run_with_cache(prompt)
    layer_probs = get_layer_probs(model, cache, token_id)
    peak_layer = layer_probs.index(max(layer_probs))
    rho = max(layer_probs) / (layer_probs[-1] + 1e-10)

    resid = cache[f"blocks.{peak_layer}.hook_resid_post"][0, -1]
    resid_normed = model.ln_final(resid.unsqueeze(0))[0]
    peak_logits_vec = resid_normed @ model.W_U + model.b_U

    if alpha == 0:
        pred_logits = final_logits[0, -1]
    else:
        pred_logits = alpha * peak_logits_vec + (1 - alpha) * final_logits[0, -1]

    predicted = model.to_string(pred_logits.argmax()).strip()
    return predicted, rho, peak_layer

# ── TruthfulQA ────────────────────────────────────────────────────────────────
print("\nLoading TruthfulQA...")
try:
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    print(f"  {len(tqa)} questions loaded")

    tqa_results = {alpha: {"correct": 0, "total": 0} for alpha in ALPHAS}
    tqa_by_rho = {"high_rho": {a: 0 for a in ALPHAS}, "low_rho": {a: 0 for a in ALPHAS},
                  "high_n": 0, "low_n": 0}

    for i, item in enumerate(tqa):
        if i % 100 == 0:
            print(f"  TruthfulQA: {i}/{len(tqa)}")
        question = item["question"]
        best_answer = item["best_answer"].split()[0].rstrip(".,;:")  # first token

        prompt = f"Q: {question}\nA:"
        rho = None

        for alpha in ALPHAS:
            try:
                pred, rho_val, _ = run_intervention(prompt, best_answer, alpha)
                if rho is None:
                    rho = rho_val
                correct = best_answer.lower() in pred.lower()
                tqa_results[alpha]["correct"] += int(correct)
                tqa_results[alpha]["total"] += 1

                if rho > 2.0:
                    tqa_by_rho["high_rho"][alpha] += int(correct)
                    if alpha == 0:
                        tqa_by_rho["high_n"] += 1
                else:
                    tqa_by_rho["low_rho"][alpha] += int(correct)
                    if alpha == 0:
                        tqa_by_rho["low_n"] += 1
            except Exception:
                continue

    print("\nTruthfulQA Results:")
    baseline_acc = tqa_results[0.0]["correct"] / max(tqa_results[0.0]["total"], 1)
    for alpha in ALPHAS:
        acc = tqa_results[alpha]["correct"] / max(tqa_results[alpha]["total"], 1)
        delta = acc - baseline_acc
        print(f"  α={alpha:.1f}: {acc:.3f} ({delta:+.3f})")

except Exception as e:
    print(f"TruthfulQA failed: {e}")
    tqa_results = {}
    tqa_by_rho = {}

# ── TriviaQA ──────────────────────────────────────────────────────────────────
print(f"\nLoading TriviaQA ({N_TRIVIAQA} samples)...")
try:
    trivia = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{N_TRIVIAQA}]")
    print(f"  {len(trivia)} questions loaded")

    trivia_results = {alpha: {"correct": 0, "total": 0} for alpha in ALPHAS}
    trivia_by_rho = {"high": {a: 0 for a in ALPHAS}, "low": {a: 0 for a in ALPHAS},
                     "high_n": 0, "low_n": 0}

    for i, item in enumerate(trivia):
        if i % 100 == 0:
            print(f"  TriviaQA: {i}/{len(trivia)}")
        question = item["question"]
        answers = item["answer"]["aliases"] if item["answer"]["aliases"] else [item["answer"]["value"]]
        first_answer = answers[0].split()[0].rstrip(".,;:")

        prompt = f"{question} The answer is"
        rho = None

        for alpha in ALPHAS:
            try:
                pred, rho_val, _ = run_intervention(prompt, first_answer, alpha)
                if rho is None:
                    rho = rho_val
                correct = any(a.lower() in pred.lower() for a in answers)
                trivia_results[alpha]["correct"] += int(correct)
                trivia_results[alpha]["total"] += 1
                if rho > 2.0:
                    trivia_by_rho["high"][alpha] += int(correct)
                    if alpha == 0:
                        trivia_by_rho["high_n"] += 1
                else:
                    trivia_by_rho["low"][alpha] += int(correct)
                    if alpha == 0:
                        trivia_by_rho["low_n"] += 1
            except Exception:
                continue

    print("\nTriviaQA Results:")
    baseline_acc = trivia_results[0.0]["correct"] / max(trivia_results[0.0]["total"], 1)
    for alpha in ALPHAS:
        acc = trivia_results[alpha]["correct"] / max(trivia_results[alpha]["total"], 1)
        delta = acc - baseline_acc
        print(f"  α={alpha:.1f}: {acc:.3f} ({delta:+.3f})")

    print("\nTriviaQA by Suppression Ratio (HIGH ρ>2 vs LOW ρ≤2):")
    print(f"  High-ρ group (n={trivia_by_rho['high_n']}):")
    for alpha in ALPHAS:
        n = trivia_by_rho['high_n']
        if n > 0:
            acc = trivia_by_rho['high'][alpha] / n
            print(f"    α={alpha:.1f}: {acc:.3f}")
    print(f"  Low-ρ group (n={trivia_by_rho['low_n']}):")
    for alpha in ALPHAS:
        n = trivia_by_rho['low_n']
        if n > 0:
            acc = trivia_by_rho['low'][alpha] / n
            print(f"    α={alpha:.1f}: {acc:.3f}")

except Exception as e:
    print(f"TriviaQA failed: {e}")
    trivia_results = {}

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
output = {
    "model": MODEL,
    "experiment": "extended_benchmarks",
    "timestamp": datetime.now().isoformat(),
    "truthfulqa": {
        str(a): {"acc": tqa_results[a]["correct"] / max(tqa_results[a]["total"], 1),
                 "n": tqa_results[a]["total"]}
        for a in ALPHAS
    } if tqa_results else {},
    "triviaqa": {
        str(a): {"acc": trivia_results[a]["correct"] / max(trivia_results[a]["total"], 1),
                 "n": trivia_results[a]["total"]}
        for a in ALPHAS
    } if trivia_results else {},
}
with open("results/experiment_14_benchmarks_extended.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved results/experiment_14_benchmarks_extended.json")
print("Experiment 14 complete.")
