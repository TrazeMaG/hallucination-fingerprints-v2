from transformer_lens import HookedTransformer
import torch

print("Loading GPT-2...")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

prompt = "The capital of France is"
print(f"\nPrompt: '{prompt}'")

logits, cache = model.run_with_cache(prompt)

print("\nTop 5 predictions:")
top5 = torch.topk(logits[0, -1], 5)
for token_id, prob in zip(top5.indices, top5.values.softmax(dim=-1)):
    print(f"  '{model.to_string(token_id)}': {prob:.4f}")

print("\nModel config:")
print(f"  Total layers: {model.cfg.n_layers}")
print(f"  Heads per layer: {model.cfg.n_heads}")
print(f"  Model dimension: {model.cfg.d_model}")
print(f"  Vocab size: {model.cfg.d_vocab}")

print("\nCache keys (first 10):")
for i, key in enumerate(list(cache.keys())[:10]):
    print(f"  {key}")

print("\nTransformerLens verified and working.")