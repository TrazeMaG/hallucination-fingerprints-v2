"""
Experiment 15 — Mistral 7B Last-Layer Suppression
===================================================
Run: python experiments/experiment_15_mistral7b.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import numpy as np
from datetime import datetime
import os

MODEL_ID     = "mistralai/Mistral-7B-v0.1"
MODEL_NAME   = "mistral-7b"
MODEL_PARAMS = "7B"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    ("The capital of France is",                "Paris",       "capitals",   "Europe"),
    ("The capital of Germany is",               "Berlin",      "capitals",   "Europe"),
    ("The capital of Japan is",                 "Tokyo",       "capitals",   "Asia"),
    ("The capital of Italy is",                 "Rome",        "capitals",   "Europe"),
    ("The capital of Spain is",                 "Madrid",      "capitals",   "Europe"),
    ("The capital of Australia is",             "Canberra",    "capitals",   "Oceania"),
    ("The capital of China is",                 "Beijing",     "capitals",   "Asia"),
    ("The capital of India is",                 "Delhi",       "capitals",   "Asia"),
    ("The capital of Russia is",                "Moscow",      "capitals",   "Europe"),
    ("The capital of Canada is",                "Ottawa",      "capitals",   "N.America"),
    ("The capital of Thailand is",              "Bangkok",     "capitals",   "Asia"),
    ("The capital of Kenya is",                 "Nairobi",     "capitals",   "Africa"),
    ("The capital of Egypt is",                 "Cairo",       "capitals",   "Africa"),
    ("The theory of evolution was proposed by", "Darwin",      "science",    "global"),
    ("The theory of relativity was proposed by","Einstein",    "science",    "global"),
    ("The chemical symbol for gold is",         "Au",          "science",    "global"),
    ("Water is made of hydrogen and",           "oxygen",      "science",    "global"),
    ("The Berlin Wall fell in",                 "1989",        "history",    "global"),
    ("World War II ended in",                   "1945",        "history",    "global"),
    ("Hamlet was written by",                   "Shakespeare", "literature", "global"),
]

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} in 4-bit...")
print(f"GPU: {torch.cuda.get_device_properties(0).name}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
n_layers = model.config.num_hidden_layers
print(f"Layers: {n_layers}  Architecture: {model.config.model_type}\n")

# ── Logit Lens ────────────────────────────────────────────────────────────────
def get_token_id(tokenizer, answer):
    tokens = tokenizer.encode(f" {answer}", add_special_tokens=False)
    return tokens[0] if tokens else None

def analyse(prompt, answer, category, region):
    token_id = get_token_id(tokenizer, answer)
    if token_id is None:
        return None
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    final_logits  = outputs.logits[0, -1]
    final_probs   = torch.softmax(final_logits, dim=-1)
    predicted     = tokenizer.decode([final_logits.argmax().item()]).strip()
    rank          = (final_probs > final_probs[token_id]).sum().item() + 1
    layer_probs = []
    for i in range(1, len(hidden_states)):
        h        = hidden_states[i][0, -1, :]
        h_normed = model.model.norm(h.unsqueeze(0))[0]
        lp       = model.lm_head(h_normed.unsqueeze(0))[0]
        prob     = torch.softmax(lp, dim=-1)[token_id].item()
        layer_probs.append(prob)
    peak_layer = layer_probs.index(max(layer_probs))
    peak_prob  = max(layer_probs)
    final_prob = layer_probs[-1]
    rho        = peak_prob / (final_prob + 1e-10)
    rel_depth  = peak_layer / len(layer_probs)
    is_correct = predicted.lower() == answer.lower()
    if is_correct:
        hall_type = "CORRECT"
    elif rank <= 10:
        hall_type = "TYPE2A_SUPPRESSION"
    else:
        hall_type = "TYPE2B_GAP"
    return {
        "prompt": prompt, "answer": answer, "predicted": predicted,
        "category": category, "region": region,
        "is_correct": is_correct, "hall_type": hall_type,
        "peak_layer": peak_layer, "rel_depth": round(rel_depth, 3),
        "peak_prob": round(peak_prob, 4), "final_prob": round(final_prob, 4),
        "rho": round(rho, 2), "rank": rank,
        "layer_probs": [round(p, 4) for p in layer_probs],
    }

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"{'Prompt':<48} {'Answer':<12} {'Type':<22} {'rho':>7}  {'Depth'}")
print("-" * 100)
results = []
for prompt, answer, category, region in PROMPTS:
    r = analyse(prompt, answer, category, region)
    if r is None:
        continue
    results.append(r)
    print(f"{prompt:<48} {answer:<12} {r['hall_type']:<22} "
          f"{r['rho']:>6.1f}x  {r['rel_depth']:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
n_correct  = sum(1 for r in results if r["hall_type"] == "CORRECT")
n_type2a   = sum(1 for r in results if r["hall_type"] == "TYPE2A_SUPPRESSION")
n_type2b   = sum(1 for r in results if r["hall_type"] == "TYPE2B_GAP")
n_total    = len(results)
all_rhos   = [r["rho"] for r in results]
all_depths = [r["rel_depth"] for r in results]
print(f"SUMMARY — Mistral 7B ({n_layers} layers)")
print(f"Correct:       {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")
print(f"Type 2a (LLS): {n_type2a}/{n_total} ({n_type2a/n_total*100:.0f}%)")
print(f"Type 2b (Gap): {n_type2b}/{n_total} ({n_type2b/n_total*100:.0f}%)")
print(f"Avg rho:       {np.mean(all_rhos):.1f}x")
print(f"Median rho:    {np.median(all_rhos):.1f}x")
print(f"Avg rel depth: {np.mean(all_depths):.3f}")

# ── Intervention ──────────────────────────────────────────────────────────────
type2a_cases = [r for r in results if r["hall_type"] == "TYPE2A_SUPPRESSION"]
print(f"\nLogit blending on {len(type2a_cases)} Type 2a cases:")
intervention_results = []
for r in type2a_cases:
    token_id   = get_token_id(tokenizer, r["answer"])
    peak_layer = r["peak_layer"]
    inputs     = tokenizer(r["prompt"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    final_logits  = outputs.logits[0, -1]
    h        = hidden_states[peak_layer + 1][0, -1, :]
    h_normed = model.model.norm(h.unsqueeze(0))[0]
    peak_lp  = model.lm_head(h_normed.unsqueeze(0))[0]
    baseline_pred = tokenizer.decode([final_logits.argmax().item()]).strip()
    best_pred, best_correct = baseline_pred, False
    for alpha in [0.3, 0.4, 0.5, 0.6]:
        blended  = alpha * peak_lp + (1 - alpha) * final_logits
        pred_str = tokenizer.decode([blended.argmax().item()]).strip()
        if pred_str.lower() == r["answer"].lower() and not best_correct:
            best_pred, best_correct = pred_str, True
    intervention_results.append({
        "prompt": r["prompt"], "answer": r["answer"],
        "baseline": baseline_pred, "best_pred": best_pred,
        "corrected": best_correct,
    })
    status = "CORRECTED" if best_correct else "not corrected"
    print(f"  {r['prompt']:<48} {baseline_pred:<10} -> {best_pred:<12} {status}")

n_corrected = sum(1 for r in intervention_results if r["corrected"])
print(f"\nIntervention corrected: {n_corrected}/{len(intervention_results)}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
output = {
    "model": MODEL_NAME, "model_params": MODEL_PARAMS,
    "n_layers": n_layers, "architecture": model.config.model_type,
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "correct": n_correct, "type2a": n_type2a, "type2b": n_type2b,
        "total": n_total,
        "avg_rho": round(float(np.mean(all_rhos)), 2),
        "median_rho": round(float(np.median(all_rhos)), 2),
        "avg_rel_depth": round(float(np.mean(all_depths)), 3),
        "intervention_corrected": n_corrected,
        "intervention_total": len(intervention_results),
    },
    "results": results,
    "intervention_results": intervention_results,
}
with open("results/experiment_15_mistral7b.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved results/experiment_15_mistral7b.json")
print("Experiment 15 complete.")