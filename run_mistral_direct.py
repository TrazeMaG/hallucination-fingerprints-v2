"""
Mistral 7B Direct — using HuggingFace with 4-bit quantization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from datetime import datetime

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

print("Loading Mistral 7B in 4-bit...")

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Loaded. Layers: {n_layers}")

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

def get_token_id(tokenizer, answer):
    ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    return ids[0]

def analyse_prompt(model, tokenizer, prompt, answer, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    final_logits = outputs.logits[0, -1].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    predicted_id = final_logits.argmax()
    predicted = tokenizer.decode(predicted_id).strip()

    token_id = get_token_id(tokenizer, answer)
    correct_rank = (final_probs > final_probs[token_id]).sum().item() + 1

    hidden_states = outputs.hidden_states
    layer_probs = []

    lm_head = model.lm_head
    norm = model.model.norm

    for layer_idx in range(1, len(hidden_states)):
        hs = hidden_states[layer_idx][0, -1].float()
        with torch.no_grad():
            normed = norm(hs.unsqueeze(0).unsqueeze(0))
            logits = lm_head(normed)[0, 0].float()
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        layer_probs.append(prob)

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

    return {
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
        "category": "",
        "layer_probs": [round(p, 4) for p in layer_probs],
    }

print(f"\nAnalysing {len(PROMPTS)} prompts...\n")
print(f"{'Prompt':<45} {'Predicted':<12} {'Type':<25} {'Peak':<8} {'Rel':<6} {'Ratio'}")
print("-" * 105)

results = []

for prompt, answer, category in PROMPTS:
    result = analyse_prompt(model, tokenizer, prompt, answer, n_layers)
    result["category"] = category
    results.append(result)

    print(f"{prompt:<45} "
          f"{result['predicted']:<12} "
          f"{result['hallucination_type']:<25} "
          f"Block {result['peak_layer']:<3} "
          f"{result['peak_layer_relative']:<6} "
          f"{result['suppression_ratio']:.1f}x")

n_correct = sum(1 for r in results if r["hallucination_type"] == "CORRECT")
n_2a = sum(1 for r in results if r["hallucination_type"] == "TYPE2A_SUPPRESSION")
n_2b = sum(1 for r in results if r["hallucination_type"] == "TYPE2B_GAP")

type2a = [r for r in results if r["hallucination_type"] == "TYPE2A_SUPPRESSION"]
avg_rel = sum(r["peak_layer_relative"] for r in type2a) / len(type2a) if type2a else 0
avg_sup = sum(r["suppression_ratio"] for r in type2a) / len(type2a) if type2a else 0

print(f"\n{'='*105}")
print(f"MISTRAL 7B SUMMARY (32 layers, 7B params)")
print(f"{'='*105}")
print(f"Correct:  {n_correct}/20 ({n_correct/20*100:.0f}%)")
print(f"Type 2a:  {n_2a}/20 ({n_2a/20*100:.0f}%)")
print(f"Type 2b:  {n_2b}/20 ({n_2b/20*100:.0f}%)")
if type2a:
    print(f"Avg relative depth: {avg_rel:.3f}")
    print(f"Avg suppression:    {avg_sup:.1f}x")

output = {
    "model": "mistralai/Mistral-7B-v0.1",
    "model_params": "7B",
    "n_layers": n_layers,
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "correct": n_correct,
        "type2a": n_2a,
        "type2b": n_2b,
        "avg_rel_depth": round(avg_rel, 3),
        "avg_suppression": round(avg_sup, 2),
    },
    "results": results,
}

with open("results/full_mistral_7b.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to results/full_mistral_7b.json")
print("Mistral 7B complete.")