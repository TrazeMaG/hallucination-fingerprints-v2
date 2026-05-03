# HallScope

Hallucination interpretability library for transformer language models.

Built on top of TransformerLens. Companion to the paper:
"Last-Layer Suppression: A Universal Causal Mechanism of Factual 
Hallucination in Transformer Language Models"

## Install

```bash
pip install hallscope
```

## Usage

```python
from hallscope import HallScope

hs = HallScope("gpt2-xl")

# Analyse a prompt
report = hs.analyse("The capital of France is", "Paris")
print(report)
# HallScope Analysis
#   Model:              gpt2-xl
#   Predicted:          a
#   Correct answer:     Paris
#   Hallucination type: TYPE2A_SUPPRESSION
#   Peak factual layer: Block 41 (85% depth)
#   Suppression ratio:  18.6x

# Correct it
corrected = hs.correct("The capital of France is")
print(corrected)  # Paris

# Run benchmark
from hallscope.benchmark import get_capitals_benchmark
prompts, answers = get_capitals_benchmark()
results = hs.benchmark(prompts, answers)
print(results)
```

## Models Tested

| Model | Suppression | Intervention |
|-------|-------------|--------------|
| GPT-2 XL | 20.8x | +45% |
| Phi-2 | 10.8x | +5% |
| Qwen 1.5 1.8B | 2.5x | +40% |
| GPT-Neo 2.7B | 1.0x | +0% |
| Pythia 2.8B | 1.1x | +0% |