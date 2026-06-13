print("Step 1: starting")
import sys
sys.stdout.flush()

print("Step 2: importing json, os, numpy")
sys.stdout.flush()
import json
import os
import numpy as np

print("Step 3: importing scipy")
sys.stdout.flush()
from scipy import stats

print("Step 4: importing torch")
sys.stdout.flush()
import torch

print("Step 5: importing TransformerLens")
sys.stdout.flush()
from transformer_lens import HookedTransformer

print("Step 6: importing datasets")
sys.stdout.flush()
from datasets import load_dataset

print("Step 7: all imports done")
print(f"CUDA available: {torch.cuda.is_available()}")
sys.stdout.flush()

print("Step 8: loading GPT-2 XL...")
sys.stdout.flush()
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
n_layers = model.cfg.n_layers

print(f"Step 9: model loaded. {n_layers} layers")
sys.stdout.flush()
