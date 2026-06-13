print("Step 1: starting")
import sys
sys.stdout.flush()

print("Step 2: torch first")
sys.stdout.flush()
import torch

print("Step 3: TransformerLens")
sys.stdout.flush()
from transformer_lens import HookedTransformer
print("TL imported ok")
sys.stdout.flush()

print("Step 4: now numpy and scipy")
sys.stdout.flush()
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
print("All imports done")
print(f"numpy: {np.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
sys.stdout.flush()
