"""
Experiment 16 — LLaMA 3 8B Last-Layer Suppression
===================================================
Extends LLS analysis to LLaMA 3 8B (Meta, 2024).
Runs in 4-bit quantization on 10GB VRAM (RTX 3080).

Requirements:
    pip install transformers bitsandbytes accelerate
    huggingface-cli login   (need HF token with LLaMA 3 access)
    Accept license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B

LLaMA 3 architecture differences from GPT-2:
    - Grouped Query Attention (GQA)
    - RoPE positional encoding
    - SwiGLU activation
    - No bias in linear layers
    - 32 layers, 4096 hidden dim
    - Trained on 15T tokens (vs GPT-2's 40GB WebText)

If LLS appears in LLaMA 3 → mechanism exists at frontier scale
If absent → confirms architectural dependency (RoPE = local-like)
Either result is a major paper contribution.

Run:
    python experiments/experiment_16_llama3_8b.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import numpy as np
from datetime import datetime
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME   = "llama3-8b"
MODEL_PARAMS = "8B"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Same 20 core facts used across ALL models in this paper ───────────────────
PROMPTS = [
    # Western capitals
    ("The capital of France is",               "Paris",       "capitals",  "Europe"),
    ("The capital of Germany is",              "Berlin",      "capitals",  "Europe"),
    ("The capital of Japan is",                "Tokyo",       "capitals",  "Asia"),
    ("The capital of Italy is",                "Rome",        "capitals",  "Europe"),
    ("The capital of Spain is",                "Madrid",      "capitals",  "Europe"),
    ("The capital of Australia is",            "Canberra",    "capitals",  "Oceania"),
    ("The capital of China is",                "Beijing",     "capitals",  "Asia"),
    ("The capital of India is",                "Delhi",       "capitals",  "Asia"),
    ("The capital of Russia is",               "Moscow",      "capitals",  "Europe"),
    ("The capital of Canada is",               "Ottawa",      "capitals",  "N.America"),
    # Non-Western
    ("The capital of Thailand is",             "Bangkok",     "capitals",  "Asia"),
    ("The capital of Kenya is",                "Nairobi",     "capitals",  "Africa"),
    ("The capital of Egypt is",                "Cairo",       "capitals",  "Africa"),
    # Science
    ("The theory of evolution was proposed by","Darwin",      "science",   "global"),
    ("The theory of relativity was proposed by","Einstein",   "science",   "global"),
    ("The chemical symbol for gold is",        "Au",          "science",   "global"),
    ("Water is made of hydrogen and",          "oxygen",      "science",   "global"),
    # History
    ("The Berlin Wall fell in",                "1989",        "history",   "global"),
    ("World War II ended in",                  "1945",        "history",   "global"),
    # Literature
    ("Hamlet was written by",                  "Shakespeare", "literature","global"),
]

# ── Load Model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} in 4-bit quantization...")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory/1e9:.1f} GB")

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
    torch_dtype=torch.float16,
)
model.eval()

if DEVICE == "cuda":
    print(f"VRAM used after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}")
print(f"Architecture: {model.config.model_type}")
print(f"Hidden size: {model.config.hidden_size}")

# ── Logit Lens for LLaMA 3 ───────────────────────────────────────────────────
def get_token_id(tokenizer, answer):
    tokens = tokenizer.encode(f" {answer}", add_special_tokens=False)
    return tokens[0] if tokens else None

def logit_lens_llama(model, tokenizer, prompt, answer):
    """
    Apply logit lens to LLaMA 3.
    Uses output_hidden_states=True to extract all layer hidden states.
    Projects each through final RMSNorm + lm_head.
    """
    token_id = get_token_id(tokenizer, answer)
    if token_id is None:
        return None

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    final_logits  = outputs.logits[0, -1]
    final_probs   = torch.softmax(final_logits, dim=-1)
    predicted_id  = final_logits.argmax().item()
    predicted     = tokenizer.decode([predicted_id]).strip()

    correct_final_prob = final_probs[token_id].item()
    correct_final_rank = (final_probs > final_probs[token_id]).sum().item() + 1

    # Apply logit lens: project each layer's hidden state through
    # the final layer norm (model.model.norm) and lm_head
    layer_probs = []
    for layer_idx in range(1, len(hidden_states)):  # skip embedding [0]
        h = hidden_states[layer_idx][0, -1, :]

        # LLaMA uses RMSNorm as final norm
        h_normed = model.model.norm(h.unsqueeze(0))[0]

        # Project to vocabulary
        logits_at_layer = model.lm_head(h_normed.unsqueeze(0))[0]
        prob = torch.softmax(logits_at_layer, dim=-1)[token_id].item()
        layer_probs.append(prob)

    peak_layer = layer_probs.index(max(layer_probs))
    peak_prob  = max(layer_probs)
    final_prob = layer_probs[-1]
    rho        = peak_prob / (final_prob + 1e-10)
    rel_depth  = peak_layer / len(layer_probs)

    is_correct = predicted.lower() == answer.lower()
    if is_correct:
        hall_type = "CORRECT"
    elif correct_final_rank <= 10:
        hall_type = "TYPE2A_SUPPRESSION"
    else:
        hall_type = "TYPE2B_GAP"

    return {
        "prompt":      prompt,
        "answer":      answer,
        "predicted":   predicted,
        "is_correct":  is_correct,
        "hall_type":   hall_type,
        "peak_layer":  peak_layer,
        "rel_depth":   round(rel_depth, 3),
        "peak_prob":   round(peak_prob, 4),
        "final_prob":  round(final_prob, 4),
        "rho":         round(rho, 2),
        "rank":        correct_final_rank,
        "layer_probs": [round(p, 4) for p in layer_probs],
    }

# ── Run Analysis ──────────────────────────────────────────────────────────────
print(f"\nAnalysing {len(PROMPTS)} prompts on {MODEL_NAME}...\n")
print(f"{'Prompt':<48} {'Answer':<12} {'Type':<22} {'ρ':>7}  {'Depth'}")
print("-" * 105)

results = []
for prompt, answer, category, region in PROMPTS:
    r = logit_lens_llama(model, tokenizer, prompt, answer)
    if r is None:
        print(f"{prompt:<48} {answer:<12} TOKENIZATION ERROR")
        continue
    r["category"] = category
    r["region"]   = region
    results.append(r)
    print(f"{prompt:<48} {answer:<12} {r['hall_type']:<22} "
          f"{r['rho']:>6.1f}x  {r['rel_depth']:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 105)
print(f"SUMMARY — {MODEL_NAME} ({MODEL_PARAMS}, {n_layers} layers)")
print("=" * 105)

n_correct = sum(1 for r in results if r["hall_type"] == "CORRECT")
n_type2a  = sum(1 for r in results if r["hall_type"] == "TYPE2A_SUPPRESSION")
n_type2b  = sum(1 for r in results if r["hall_type"] == "TYPE2B_GAP")
n_total   = len(results)

print(f"\nCorrect:          {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")
print(f"Type 2a (LLS):    {n_type2a}/{n_total} ({n_type2a/n_total*100:.0f}%)")
print(f"Type 2b (Gap):    {n_type2b}/{n_total} ({n_type2b/n_total*100:.0f}%)")

all_rhos   = [r["rho"] for r in results]
all_depths = [r["rel_depth"] for r in results]
type2a_cases = [r for r in results if r["hall_type"] == "TYPE2A_SUPPRESSION"]

print(f"\nAll cases:")
print(f"  Avg ρ:         {np.mean(all_rhos):.1f}x")
print(f"  Median ρ:      {np.median(all_rhos):.1f}x")
print(f"  Max ρ:         {max(all_rhos):.1f}x")
print(f"  Avg rel depth: {np.mean(all_depths):.3f}")

if type2a_cases:
    t2a_rhos   = [r["rho"] for r in type2a_cases]
    t2a_depths = [r["rel_depth"] for r in type2a_cases]
    print(f"\nType 2a cases only:")
    print(f"  Avg ρ:         {np.mean(t2a_rhos):.1f}x")
    print(f"  Avg rel depth: {np.mean(t2a_depths):.3f}")

# Full comparison table across all models in paper
print(f"\n{'='*105}")
print("CROSS-MODEL COMPARISON TABLE (paste into paper)")
print(f"{'='*105}")
print(f"{'Model':<20} {'Params':<8} {'Layers':<8} {'Correct':<10} {'Type2a':<10} {'Avg ρ':<10} {'Depth':<8} {'Arch'}")
print("-" * 90)
print(f"{'GPT-2 Small':<20} {'124M':<8} {'12':<8} {'10%':<10} {'10%':<10} {'2.4x':<10} {'0.79':<8} GPT-2 (global attn)")
print(f"{'GPT-2 Medium':<20} {'345M':<8} {'24':<8} {'20%':<10} {'40%':<10} {'15.6x':<10} {'0.86':<8} GPT-2 (global attn)")
print(f"{'GPT-2 Large':<20} {'774M':<8} {'36':<8} {'15%':<10} {'15%':<10} {'17.1x':<10} {'0.86':<8} GPT-2 (global attn)")
print(f"{'GPT-2 XL':<20} {'1.5B':<8} {'48':<8} {'35%':<10} {'50%':<10} {'20.8x':<10} {'0.81':<8} GPT-2 (global attn)")
print(f"{'Phi-2':<20} {'2.7B':<8} {'32':<8} {'70%':<10} {'20%':<10} {'10.8x':<10} {'0.875':<8} Phi (global attn)")
print(f"{'Qwen 1.5-1.8B':<20} {'1.8B':<8} {'24':<8} {'10%':<10} {'70%':<10} {'2.5x':<10} {'0.905':<8} Qwen (global attn)")
print(f"{'GPT-Neo 125M':<20} {'125M':<8} {'12':<8} {'5%':<10} {'25%':<10} {'1.0x':<10} {'0.917':<8} GPT-Neo (local attn)")
print(f"{'GPT-Neo 1.3B':<20} {'1.3B':<8} {'24':<8} {'40%':<10} {'45%':<10} {'1.0x':<10} {'0.958':<8} GPT-Neo (local attn)")
print(f"{'GPT-Neo 2.7B':<20} {'2.7B':<8} {'32':<8} {'15%':<10} {'65%':<10} {'1.0x':<10} {'0.969':<8} GPT-Neo (local attn)")
print(f"{'Pythia 2.8B':<20} {'2.8B':<8} {'32':<8} {'40%':<10} {'50%':<10} {'1.1x':<10} {'0.950':<8} Pythia (rotary attn)")
print(f"{'Mistral 7B':<20} {'7B':<8} {'?':<8} {'?':<10} {'?':<10} {'?':<10} {'?':<8} Mistral (GQA+SWA) <-- fill")
print(f"{'LLaMA 3 8B':<20} {'8B':<8} {str(n_layers):<8} {str(n_correct/n_total*100)[:4]+'%':<10} "
      f"{str(n_type2a/n_total*100)[:4]+'%':<10} "
      f"{str(round(np.mean(all_rhos),1))+'x':<10} "
      f"{str(round(np.mean(all_depths),3)):<8} LLaMA (GQA+RoPE)")

# ── Logit Blending Intervention ───────────────────────────────────────────────
print(f"\n{'='*105}")
print("LOGIT BLENDING INTERVENTION — Type 2a cases")
print(f"{'='*105}\n")

intervention_results = []
for r in type2a_cases:
    prompt   = r["prompt"]
    answer   = r["answer"]
    token_id = get_token_id(tokenizer, answer)
    peak_layer = r["peak_layer"]

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_logits  = outputs.logits[0, -1]

    h = hidden_states[peak_layer + 1][0, -1, :]
    h_normed    = model.model.norm(h.unsqueeze(0))[0]
    peak_logits = model.lm_head(h_normed.unsqueeze(0))[0]

    best_alpha   = 0.0
    best_pred    = tokenizer.decode([final_logits.argmax().item()]).strip()
    best_correct = False

    for alpha in [0.3, 0.4, 0.5, 0.6]:
        blended  = alpha * peak_logits + (1 - alpha) * final_logits
        pred_str = tokenizer.decode([blended.argmax().item()]).strip()
        if pred_str.lower() == answer.lower() and not best_correct:
            best_alpha   = alpha
            best_pred    = pred_str
            best_correct = True

    baseline_pred = tokenizer.decode([final_logits.argmax().item()]).strip()
    intervention_results.append({
        "prompt":    prompt,
        "answer":    answer,
        "baseline":  baseline_pred,
        "best_pred": best_pred,
        "corrected": best_correct,
    })
    status = "✓ CORRECTED" if best_correct else "✗ not corrected"
    print(f"{prompt:<48} {baseline_pred:<12} → {best_pred:<12} {status}")

n_corrected = sum(1 for r in intervention_results if r["corrected"])
print(f"\nIntervention: {n_corrected}/{len(intervention_results)} Type2a cases corrected")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
output = {
    "model":        MODEL_NAME,
    "model_id":     MODEL_ID,
    "model_params": MODEL_PARAMS,
    "n_layers":     n_layers,
    "architecture": model.config.model_type,
    "timestamp":    datetime.now().isoformat(),
    "summary": {
        "correct":                n_correct,
        "type2a":                 n_type2a,
        "type2b":                 n_type2b,
        "total":                  n_total,
        "avg_rho":                round(float(np.mean(all_rhos)), 2),
        "median_rho":             round(float(np.median(all_rhos)), 2),
        "avg_rel_depth":          round(float(np.mean(all_depths)), 3),
        "intervention_corrected": n_corrected,
        "intervention_total":     len(intervention_results),
    },
    "results":              results,
    "intervention_results": intervention_results,
}

with open(f"results/experiment_16_{MODEL_NAME.replace('-','_')}.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/experiment_16_{MODEL_NAME.replace('-','_')}.json")
print("Experiment 16 complete.")
print("\nSend results/experiment_16_llama3_8b.json back to Nikhil.")
