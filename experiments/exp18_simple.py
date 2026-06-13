import torch
from transformer_lens import HookedTransformer
print("STARTED")
import json, os, math
import numpy as np
from scipy import stats
from datasets import load_dataset
from datetime import datetime

print("All imports done")
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
n_layers = model.cfg.n_layers
print("Model loaded:", n_layers, "layers")