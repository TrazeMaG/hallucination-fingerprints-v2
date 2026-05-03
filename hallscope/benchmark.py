"""
HallScope Benchmark — Standard evaluation utilities
"""

CAPITALS_BENCHMARK = [
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Australia is", "Canberra"),
    ("The capital of China is", "Beijing"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of Canada is", "Ottawa"),
    ("The capital of Brazil is", "Brasilia"),
    ("The capital of India is", "Delhi"),
    ("The capital of Argentina is", "Buenos"),
    ("The Berlin Wall fell in", "1989"),
    ("Water is made of hydrogen and", "oxygen"),
    ("Albert Einstein discovered", "relativity"),
    ("The theory of evolution was proposed by", "Darwin"),
    ("The chemical symbol for gold is", "Au"),
    ("Shakespeare wrote", "Hamlet"),
    ("The first president of the United States was", "Washington"),
    ("The speed of light is approximately", "299"),
]

def get_capitals_benchmark():
    prompts = [p for p, a in CAPITALS_BENCHMARK]
    answers = [a for p, a in CAPITALS_BENCHMARK]
    return prompts, answers