"""
Experiment 18 — Expanded Self-Patching (100 cases) + DoLa Comparison
======================================================================
FIXED: torch and TransformerLens imported before numpy/scipy
"""

import sys
import torch  # MUST be first
from transformer_lens import HookedTransformer  # MUST be second

import json
import os
import math
import numpy as np
from datetime import datetime
from scipy import stats
from datasets import load_dataset

print("Loading GPT-2 XL for self-patching and DoLa comparison...")
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
n_layers = model.cfg.n_layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loaded. {n_layers} layers. Device: {DEVICE}")

FIXED_LAYER = round(0.81 * n_layers)

# ── CI Helper ─────────────────────────────────────────────────────────────────
def binomial_ci(n_success, n_total, confidence=0.95):
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
    return round(p,4), round(max(0,centre-spread),4), round(min(1,centre+spread),4)

# ── 100 Patching Prompts ──────────────────────────────────────────────────────
PATCHING_PROMPTS = [
    # Capitals Western
    ("The capital of France is","Paris"),
    ("The capital of Germany is","Berlin"),
    ("The capital of Italy is","Rome"),
    ("The capital of Spain is","Madrid"),
    ("The capital of Japan is","Tokyo"),
    ("The capital of China is","Beijing"),
    ("The capital of Russia is","Moscow"),
    ("The capital of Canada is","Ottawa"),
    ("The capital of Australia is","Canberra"),
    ("The capital of Brazil is","Bras"),
    ("The capital of Argentina is","Buenos"),
    ("The capital of Mexico is","Mexico"),
    ("The capital of Portugal is","Lisbon"),
    ("The capital of Netherlands is","Amsterdam"),
    ("The capital of Sweden is","Stockholm"),
    ("The capital of Norway is","Oslo"),
    ("The capital of Denmark is","Copenhagen"),
    ("The capital of Poland is","Warsaw"),
    ("The capital of Austria is","Vienna"),
    ("The capital of Switzerland is","Bern"),
    # Capitals Non-Western
    ("The capital of Thailand is","Bangkok"),
    ("The capital of Vietnam is","Hanoi"),
    ("The capital of Indonesia is","Jakarta"),
    ("The capital of Malaysia is","Kuala"),
    ("The capital of Philippines is","Manila"),
    ("The capital of Kenya is","Nairobi"),
    ("The capital of Ghana is","Accra"),
    ("The capital of Nigeria is","Abuja"),
    ("The capital of Egypt is","Cairo"),
    ("The capital of Morocco is","Rabat"),
    ("The capital of Ethiopia is","Addis"),
    ("The capital of Peru is","Lima"),
    ("The capital of Colombia is","Bogota"),
    ("The capital of Chile is","Santiago"),
    ("The capital of Kazakhstan is","Astana"),
    ("The capital of Pakistan is","Islamabad"),
    ("The capital of Bangladesh is","Dhaka"),
    ("The capital of Sri Lanka is","Colombo"),
    ("The capital of Nepal is","Kathmandu"),
    ("The capital of Mongolia is","Ulaanbaatar"),
    # Science
    ("The chemical symbol for gold is","Au"),
    ("The chemical symbol for iron is","Fe"),
    ("The chemical symbol for silver is","Ag"),
    ("The chemical symbol for copper is","Cu"),
    ("The chemical symbol for lead is","Pb"),
    ("Water is made of hydrogen and","oxygen"),
    ("The theory of evolution was proposed by","Darwin"),
    ("The theory of relativity was proposed by","Einstein"),
    ("Gravity was discovered by","Newton"),
    ("Penicillin was discovered by","Fleming"),
    ("The telephone was invented by","Bell"),
    ("DNA was discovered by","Watson"),
    ("Radioactivity was discovered by","Curie"),
    ("The electron was discovered by","Thomson"),
    ("The periodic table was created by","Mendeleev"),
    # History
    ("The Berlin Wall fell in","1989"),
    ("World War II ended in","1945"),
    ("World War I ended in","1918"),
    ("The American Declaration of Independence was signed in","1776"),
    ("The French Revolution began in","1789"),
    ("The first moon landing was in","1969"),
    ("The first iPhone was released in","2007"),
    ("The internet was invented by","Berners"),
    ("The printing press was invented by","Gutenberg"),
    ("Christopher Columbus reached America in","1492"),
    # Literature
    ("Hamlet was written by","Shakespeare"),
    ("Romeo and Juliet was written by","Shakespeare"),
    ("1984 was written by","Orwell"),
    ("Brave New World was written by","Huxley"),
    ("The Great Gatsby was written by","Fitzgerald"),
    ("Don Quixote was written by","Cervantes"),
    ("War and Peace was written by","Tolstoy"),
    ("Crime and Punishment was written by","Dostoevsky"),
    ("Harry Potter was written by","Rowling"),
    ("The Lord of the Rings was written by","Tolkien"),
    # Geography
    ("The longest river in the world is","Nile"),
    ("The highest mountain in the world is","Everest"),
    ("The largest ocean is the","Pacific"),
    ("The largest continent is","Asia"),
    ("The smallest continent is","Australia"),
    ("The largest country by area is","Russia"),
    ("The most populous country is","China"),
    ("The largest desert is the","Sahara"),
    ("The deepest lake is Lake","Baikal"),
    ("The longest river in the US is","Missouri"),
    # Modern/Tech
    ("Microsoft was founded by","Gates"),
    ("Apple was co-founded by","Jobs"),
    ("Amazon was founded by","Bezos"),
    ("Facebook was created by","Zuckerberg"),
    ("Google was co-founded by","Brin"),
    ("Tesla was co-founded by","Musk"),
    ("Twitter was created by","Dorsey"),
    ("Wikipedia was co-founded by","Wales"),
    ("Linux was created by","Torvalds"),
    ("Python was created by","Guido"),
    # Non-Western History/Culture
    ("The Taj Mahal is located in","India"),
    ("The Great Wall is located in","China"),
    ("Mahatma Gandhi led the independence of","India"),
    ("Nelson Mandela fought against","apartheid"),
    ("The samurai were warriors of","Japan"),
    ("Confucius was a philosopher of","China"),
    ("The Aztec empire was in","Mexico"),
    ("The Inca empire was in","Peru"),
    ("Cleopatra was the queen of","Egypt"),
    ("Buddha was born in","Nepal"),
]

def get_token_id(answer):
    try:
        return model.to_single_token(f" {answer}")
    except:
        tokens = model.to_tokens(f" {answer}")[0]
        return tokens[1].item() if len(tokens) > 1 else None

# ── Self-Patching ─────────────────────────────────────────────────────────────
print(f"\nRunning expanded self-patching on {len(PATCHING_PROMPTS)} cases...")
print(f"\n{'Prompt':<48} {'Answer':<12} {'Baseline':<12} {'Patched':<12} {'Dp':>6}")
print("-" * 95)

patching_results = []

for prompt, answer in PATCHING_PROMPTS:
    token_id = get_token_id(answer)
    if token_id is None:
        continue

    with torch.no_grad():
        logits, cache = model.run_with_cache(prompt)

    final_logits = logits[0, -1]
    final_probs = torch.softmax(final_logits, dim=-1)
    baseline_pred = model.to_string(final_logits.argmax()).strip()
    baseline_prob = final_probs[token_id].item()

    layer_probs = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        lp = torch.softmax(resid_normed @ model.W_U + model.b_U, dim=-1)
        layer_probs.append(lp[token_id].item())

    peak_layer = layer_probs.index(max(layer_probs))
    peak_prob = max(layer_probs)
    rho = peak_prob / (baseline_prob + 1e-10)

    peak_resid = cache[f"blocks.{peak_layer}.hook_resid_post"][0, -1]
    peak_normed = model.ln_final(peak_resid.unsqueeze(0))[0]
    patched_logits = peak_normed @ model.W_U + model.b_U
    patched_pred = model.to_string(patched_logits.argmax()).strip()
    patched_prob = torch.softmax(patched_logits, dim=-1)[token_id].item()
    delta_p = patched_prob - baseline_prob

    baseline_correct = baseline_pred.lower() == answer.lower()
    patched_correct = patched_pred.lower() == answer.lower()

    patching_results.append({
        "prompt": prompt, "answer": answer,
        "baseline_pred": baseline_pred, "patched_pred": patched_pred,
        "baseline_correct": baseline_correct, "patched_correct": patched_correct,
        "baseline_prob": round(baseline_prob,4), "patched_prob": round(patched_prob,4),
        "delta_p": round(delta_p,4), "peak_layer": peak_layer,
        "rel_depth": round(peak_layer/n_layers,3), "rho": round(rho,2),
    })
    print(f"{prompt:<48} {answer:<12} {baseline_pred:<12} {patched_pred:<12} {delta_p:>+6.3f}")

# Statistics
n_total = len(patching_results)
n_already_correct = sum(1 for r in patching_results if r["baseline_correct"])
n_hallucination = n_total - n_already_correct
n_restored = sum(1 for r in patching_results if r["patched_correct"] and not r["baseline_correct"])
delta_ps = [r["delta_p"] for r in patching_results]

res_mean, res_lo, res_hi = binomial_ci(n_restored, n_hallucination)

print(f"\n{'='*95}")
print(f"SELF-PATCHING RESULTS (n={n_total})")
print(f"{'='*95}")
print(f"Already correct at baseline: {n_already_correct}/{n_total} ({n_already_correct/n_total:.1%})")
print(f"Hallucination cases:         {n_hallucination}/{n_total}")
print(f"Restored by patching:        {n_restored}/{n_hallucination} ({n_restored/max(1,n_hallucination):.1%})")
print(f"Restoration rate [95% CI]:   {res_mean:.1%} [{res_lo:.1%}-{res_hi:.1%}]")
print(f"Mean delta_p:                {np.mean(delta_ps):.4f} +/- {np.std(delta_ps):.4f}")
print(f"Median delta_p:              {np.median(delta_ps):.4f}")

# ── DoLa Comparison on TruthfulQA ────────────────────────────────────────────
print(f"\n{'='*95}")
print("DOLA vs LLS COMPARISON on TruthfulQA (Chuang et al., ICLR 2024)")
print(f"{'='*95}")

try:
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    N_TQA = min(200, len(tqa))
    print(f"Loaded {N_TQA} TruthfulQA questions")

    baseline_c = dola_c = lls_oracle_c = lls_fixed_c = 0

    for i, item in enumerate(tqa.select(range(N_TQA))):
        question = item["question"]
        best_answer = item["best_answer"].split()[0].rstrip(".,;:")
        token_id = get_token_id(best_answer)
        if token_id is None:
            continue

        prompt = f"Q: {question}\nA:"
        try:
            with torch.no_grad():
                logits, cache = model.run_with_cache(prompt)

            final_logits = logits[0, -1]

            layer_logits_list = []
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
                resid_normed = model.ln_final(resid.unsqueeze(0))[0]
                lp = resid_normed @ model.W_U + model.b_U
                layer_logits_list.append(lp)

            layer_probs = [torch.softmax(lp, dim=-1)[token_id].item()
                          for lp in layer_logits_list]
            peak_layer = layer_probs.index(max(layer_probs))

            # Baseline
            baseline_pred = model.to_string(final_logits.argmax()).strip()

            # DoLa: subtract premature layer (n//2) from final
            premature = n_layers // 2
            dola_logits = final_logits - 0.5 * layer_logits_list[premature]
            dola_pred = model.to_string(dola_logits.argmax()).strip()

            # LLS oracle
            oracle_blended = 0.5 * layer_logits_list[peak_layer] + 0.5 * final_logits
            lls_oracle_pred = model.to_string(oracle_blended.argmax()).strip()

            # LLS fixed depth
            fixed_blended = 0.5 * layer_logits_list[FIXED_LAYER] + 0.5 * final_logits
            lls_fixed_pred = model.to_string(fixed_blended.argmax()).strip()

            ans_lower = best_answer.lower()
            if baseline_pred.lower() == ans_lower: baseline_c += 1
            if dola_pred.lower() == ans_lower: dola_c += 1
            if lls_oracle_pred.lower() == ans_lower: lls_oracle_c += 1
            if lls_fixed_pred.lower() == ans_lower: lls_fixed_c += 1

        except Exception:
            continue

        if i % 50 == 0:
            n_done = i + 1
            print(f"  [{i}/{N_TQA}] Baseline={baseline_c/n_done:.1%} "
                  f"DoLa={dola_c/n_done:.1%} "
                  f"LLS-oracle={lls_oracle_c/n_done:.1%} "
                  f"LLS-fixed={lls_fixed_c/n_done:.1%}")

    n = N_TQA
    print(f"\nTruthfulQA Results (n={n}):")
    print(f"  Baseline:              {baseline_c/n:.3f} ({baseline_c/n:.1%})")
    print(f"  DoLa (ICLR 2024):     {dola_c/n:.3f} ({dola_c/n:.1%}) delta={dola_c/n - baseline_c/n:+.3f}")
    print(f"  LLS oracle blend:     {lls_oracle_c/n:.3f} ({lls_oracle_c/n:.1%}) delta={lls_oracle_c/n - baseline_c/n:+.3f}")
    print(f"  LLS fixed-depth:      {lls_fixed_c/n:.3f} ({lls_fixed_c/n:.1%}) delta={lls_fixed_c/n - baseline_c/n:+.3f}")
    print(f"\nOracle vs Fixed-depth gap: {lls_oracle_c/n - lls_fixed_c/n:+.3f}")
    if lls_oracle_c/n - lls_fixed_c/n < 0.05:
        print("Fixed-depth heuristic is practical (gap < 5%)")

    dola_results = {
        "n": n,
        "baseline": round(baseline_c/n, 4),
        "dola": round(dola_c/n, 4),
        "lls_oracle": round(lls_oracle_c/n, 4),
        "lls_fixed": round(lls_fixed_c/n, 4),
    }

except Exception as e:
    print(f"TruthfulQA failed: {e}")
    dola_results = {}

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
output = {
    "model": "gpt2-xl",
    "timestamp": datetime.now().isoformat(),
    "self_patching": {
        "n_total": n_total,
        "n_already_correct": n_already_correct,
        "n_hallucination_cases": n_hallucination,
        "n_restored": n_restored,
        "restoration_rate": res_mean,
        "restoration_ci_lo": res_lo,
        "restoration_ci_hi": res_hi,
        "mean_delta_p": round(float(np.mean(delta_ps)), 4),
        "std_delta_p": round(float(np.std(delta_ps)), 4),
        "median_delta_p": round(float(np.median(delta_ps)), 4),
    },
    "dola_comparison": dola_results,
    "patching_results": patching_results,
}

with open("results/experiment_18_patching_dola.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: results/experiment_18_patching_dola.json")
print("Experiment 18 complete.")
