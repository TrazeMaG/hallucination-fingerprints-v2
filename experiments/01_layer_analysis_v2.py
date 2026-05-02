"""
Experiment 1 — Layer Analysis v2 (GPT-2 Medium)
=================================================
Same analysis as GPT-2 Small but on GPT-2 Medium (345M, 24 layers).
Goal: Does Last-Layer Suppression persist at larger scale?

Key question: Does suppression happen at the absolute final layer
or at a consistent relative depth?
"""

from transformer_lens import HookedTransformer
import torch
import json
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────

MODEL_NAME = "gpt2-xl"
MODEL_PARAMS = "1.5B"

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
    ("The speed of light is", "299", "science"),
    ("Albert Einstein discovered", "relativity", "science"),
    ("Shakespeare wrote", "Hamlet", "literature"),
    ("The first president of the United States was", "Washington", "history"),
    ("The theory of evolution was proposed by", "Darwin", "science"),
    ("The chemical symbol for gold is", "Au", "science"),
]

# ── Load Model ────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()
n_layers = model.cfg.n_layers
print(f"Model loaded.")
print(f"  Layers:    {n_layers}")
print(f"  Heads:     {model.cfg.n_heads}")
print(f"  Dimension: {model.cfg.d_model}")

# ── Core Functions ────────────────────────────────────────────────

def get_layer_prob(model, cache, layer, token_id):
    """
    Project residual stream at given layer to vocabulary space.
    Returns probability of target token at that layer.
    This is the logit lens technique.
    """
    resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
    resid_normed = model.ln_final(resid.unsqueeze(0))[0]
    logits = resid_normed @ model.W_U + model.b_U
    prob = torch.softmax(logits, dim=-1)[token_id].item()
    return prob


def analyse_prompt(model, cache_fn, prompt, correct_answer, n_layers):
    """
    Full hallucination analysis for a single prompt.
    Returns structured results including layer-by-layer probabilities,
    hallucination type, suppression ratio, and peak factual layer.
    """

    correct_token = f" {correct_answer}"

    try:
        token_id = model.to_single_token(correct_token)
    except Exception:
        tokens = model.to_tokens(correct_token)[0]
        token_id = tokens[1].item()

    with torch.no_grad():
        logits, cache = model.run_with_cache(prompt)

    final_logits = logits[0, -1]
    final_probs = torch.softmax(final_logits, dim=-1)
    predicted_token = model.to_string(final_logits.argmax())

    correct_final_prob = final_probs[token_id].item()
    correct_final_rank = (final_probs > final_probs[token_id]).sum().item() + 1

    layer_probs = []
    for layer in range(n_layers):
        prob = get_layer_prob(model, cache, layer, token_id)
        layer_probs.append(prob)

    peak_layer = layer_probs.index(max(layer_probs))
    peak_prob = max(layer_probs)
    final_prob = layer_probs[-1]
    suppression_ratio = peak_prob / (final_prob + 1e-10)

    # Relative depth — where is the peak as fraction of total layers?
    relative_peak = peak_layer / n_layers

    is_correct = predicted_token.strip().lower() == correct_answer.lower()

    if is_correct:
        hall_type = "CORRECT"
    elif correct_final_rank <= 10:
        hall_type = "TYPE2A_LAST_LAYER_SUPPRESSION"
    else:
        hall_type = "TYPE2B_KNOWLEDGE_GAP"

    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "predicted": predicted_token.strip(),
        "is_correct": is_correct,
        "hallucination_type": hall_type,
        "peak_layer": peak_layer,
        "peak_layer_relative": round(relative_peak, 3),
        "peak_prob": round(peak_prob, 4),
        "final_prob": round(final_prob, 4),
        "suppression_ratio": round(suppression_ratio, 2),
        "correct_final_rank": correct_final_rank,
        "n_layers": n_layers,
        "layer_probs": [round(p, 4) for p in layer_probs],
    }


# ── Run Analysis ──────────────────────────────────────────────────

print(f"\nAnalysing {len(PROMPTS)} prompts...\n")
print(f"{'Prompt':<45} {'Predicted':<12} {'Type':<30} {'Peak':<10} {'Rel':<8} {'Ratio'}")
print("-" * 120)

results = []

for prompt, correct, category in PROMPTS:
    result = analyse_prompt(model, None, prompt, correct, n_layers)
    result["category"] = category
    results.append(result)

    print(f"{prompt:<45} "
          f"{result['predicted']:<12} "
          f"{result['hallucination_type']:<30} "
          f"Block {result['peak_layer']:<5} "
          f"{result['peak_layer_relative']:<8} "
          f"{result['suppression_ratio']:.1f}x")

# ── Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 120)
print(f"SUMMARY — {MODEL_NAME} ({MODEL_PARAMS}, {n_layers} layers)")
print("=" * 120)

n_correct = sum(1 for r in results if r["hallucination_type"] == "CORRECT")
n_type2a = sum(1 for r in results if r["hallucination_type"] == "TYPE2A_LAST_LAYER_SUPPRESSION")
n_type2b = sum(1 for r in results if r["hallucination_type"] == "TYPE2B_KNOWLEDGE_GAP")

print(f"Total prompts:           {len(results)}")
print(f"Correct:                 {n_correct} ({n_correct/len(results)*100:.1f}%)")
print(f"Type 2a (Suppression):   {n_type2a} ({n_type2a/len(results)*100:.1f}%)")
print(f"Type 2b (Gap):           {n_type2b} ({n_type2b/len(results)*100:.1f}%)")

suppression_cases = [
    r for r in results
    if r["hallucination_type"] == "TYPE2A_LAST_LAYER_SUPPRESSION"
]

if suppression_cases:
    avg_peak = sum(r["peak_layer"] for r in suppression_cases) / len(suppression_cases)
    avg_rel = sum(r["peak_layer_relative"] for r in suppression_cases) / len(suppression_cases)
    avg_ratio = sum(r["suppression_ratio"] for r in suppression_cases) / len(suppression_cases)
    print(f"\nType 2a cases:")
    print(f"  Average peak layer:          Block {avg_peak:.1f} of {n_layers}")
    print(f"  Average relative depth:      {avg_rel:.3f} (0=first, 1=last)")
    print(f"  Average suppression ratio:   {avg_ratio:.1f}x")
    print(f"  Suppression always at:       Block {n_layers} (final)")

all_cases = [r for r in results if not r["is_correct"]]
if all_cases:
    avg_ratio_all = sum(r["suppression_ratio"] for r in all_cases) / len(all_cases)
    max_ratio = max(r["suppression_ratio"] for r in all_cases)
    max_case = max(all_cases, key=lambda r: r["suppression_ratio"])
    print(f"\nAll hallucination cases:")
    print(f"  Average suppression ratio:   {avg_ratio_all:.1f}x")
    print(f"  Maximum suppression ratio:   {max_ratio:.1f}x")
    print(f"  Most suppressed prompt:      '{max_case['prompt']}'")

# ── Compare with GPT-2 Small ──────────────────────────────────────

print(f"\nCross-model comparison so far:")
print(f"  GPT-2 Small  (124M, 12 layers): avg peak Block 9.5,  2 Type2a cases")
print(f"  GPT-2 Medium (345M, {n_layers} layers): avg peak Block ?,   ? Type2a cases")
print(f"  (fill in from results above)")

# ── Save Results ──────────────────────────────────────────────────

output = {
    "model": MODEL_NAME,
    "model_params": MODEL_PARAMS,
    "n_layers": n_layers,
    "timestamp": datetime.now().isoformat(),
    "n_prompts": len(results),
    "summary": {
        "correct": n_correct,
        "type2a_suppression": n_type2a,
        "type2b_gap": n_type2b,
    },
    "results": results
}

with open(f"results/experiment_01_{MODEL_NAME.replace('-', '_')}.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/experiment_01_{MODEL_NAME.replace('-', '_')}.json")
print(f"Experiment 1 ({MODEL_NAME}) complete.")