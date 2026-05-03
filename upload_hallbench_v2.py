from datasets import Dataset
import json

# Load our benchmark
from hallscope.hallbench import HALLBENCH_V2

data = {
    "prompt": [item[0] for item in HALLBENCH_V2],
    "answer": [item[1] for item in HALLBENCH_V2],
    "category": [item[2] for item in HALLBENCH_V2],
    "tier": [item[3] for item in HALLBENCH_V2],
    "tier_description": [
        "High suppression — single token factual recall" if item[3] == "tier1"
        else "Borderline — sometimes survives suppression" if item[3] == "tier2"
        else "Knowledge gap — model likely never learned this"
        for item in HALLBENCH_V2
    ],
}

ds = Dataset.from_dict(data)
print(f"Dataset created: {len(ds)} examples")
print(f"Tiers: {ds.unique('tier')}")
ds.push_to_hub("Trazemag/hallbench-v2")
print("Uploaded to HuggingFace!")