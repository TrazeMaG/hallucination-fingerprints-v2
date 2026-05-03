"""
HallBench v2 — The standard benchmark for Last-Layer Suppression research.

Designed to distinguish between hallucination types rather than
just measuring overall accuracy. Three tiers:

Tier 1 — High suppression facts: single-token answers, clear relation
          words, facts that GPT-2 family models consistently suppress.
          These are drawn from our validated experiment results.

Tier 2 — Borderline facts: answers that sometimes survive suppression
          depending on model size. Tests the survival threshold.

Tier 3 — Knowledge gap facts: facts outside typical training data.
          Tests that intervention does not hallucinate wrong answers.
"""

TIER1_HIGH_SUPPRESSION = [
    # Capital cities — proven Type2a in GPT-2 XL
    ("The capital of France is", "Paris", "capitals"),
    ("The capital of Germany is", "Berlin", "capitals"),
    ("The capital of Russia is", "Moscow", "capitals"),
    ("The capital of China is", "Beijing", "capitals"),
    ("The capital of India is", "Delhi", "capitals"),
    ("The capital of Brazil is", "Brasilia", "capitals"),
    ("The capital of Argentina is", "Buenos", "capitals"),
    ("The capital of Canada is", "Ottawa", "capitals"),
    ("The capital of Egypt is", "Cairo", "capitals"),
    ("The capital of South Africa is", "Pretoria", "capitals"),
    ("The capital of Mexico is", "Mexico", "capitals"),
    ("The capital of Turkey is", "Ankara", "capitals"),
    ("The capital of Poland is", "Warsaw", "capitals"),
    ("The capital of Sweden is", "Stockholm", "capitals"),
    ("The capital of Norway is", "Oslo", "capitals"),
    # Science — proven Type2a
    ("The theory of evolution was proposed by", "Darwin", "science"),
    ("Albert Einstein discovered", "relativity", "science"),
    ("The chemical symbol for gold is", "Au", "science"),
    ("The chemical symbol for iron is", "Fe", "science"),
    ("The chemical symbol for silver is", "Ag", "science"),
    ("Water is made of hydrogen and", "oxygen", "science"),
    ("The speed of light is approximately", "299", "science"),
    ("The powerhouse of the cell is the", "mitochondria", "biology"),
    ("DNA stands for deoxyribonucleic", "acid", "biology"),
    ("The force of gravity was described by", "Newton", "science"),
]

TIER2_BORDERLINE = [
    # Facts that survive in larger models but not smaller
    ("The capital of Australia is", "Canberra", "capitals"),
    ("The capital of Japan is", "Tokyo", "capitals"),
    ("The capital of Italy is", "Rome", "capitals"),
    ("The capital of Spain is", "Madrid", "capitals"),
    ("The Berlin Wall fell in", "1989", "history"),
    ("The first moon landing was in", "1969", "history"),
    ("The periodic table was created by", "Mendeleev", "science"),
    ("Shakespeare wrote", "Hamlet", "literature"),
    ("The first president of the United States was", "Washington", "history"),
    ("The Eiffel Tower is located in", "Paris", "geography"),
    ("The Amazon river is in", "South", "geography"),
    ("The Great Wall is in", "China", "geography"),
    ("Napoleon was exiled to", "Elba", "history"),
    ("The French Revolution began in", "1789", "history"),
    ("Gravity was described in Principia by", "Newton", "science"),
]

TIER3_KNOWLEDGE_GAP = [
    # Facts unlikely to be in training data or too specific
    ("The capital of Bhutan is", "Thimphu", "capitals"),
    ("The capital of Suriname is", "Paramaribo", "capitals"),
    ("The capital of Kyrgyzstan is", "Bishkek", "capitals"),
    ("The capital of Eritrea is", "Asmara", "capitals"),
    ("The capital of Vanuatu is", "Port", "capitals"),
    ("The chemical symbol for tungsten is", "W", "science"),
    ("The chemical symbol for osmium is", "Os", "science"),
    ("The speed of sound in water is", "1480", "science"),
    ("The Planck constant is approximately", "6.626", "science"),
    ("The Treaty of Westphalia was signed in", "1648", "history"),
]

# Combined full benchmark
HALLBENCH_V2 = (
    [(p, a, c, "tier1") for p, a, c in TIER1_HIGH_SUPPRESSION] +
    [(p, a, c, "tier2") for p, a, c in TIER2_BORDERLINE] +
    [(p, a, c, "tier3") for p, a, c in TIER3_KNOWLEDGE_GAP]
)

def get_hallbench_v2():
    """
    Returns the full HallBench v2 benchmark.
    
    Returns:
        List of (prompt, answer, category, tier) tuples.
    """
    return HALLBENCH_V2

def get_tier(tier_number):
    """
    Returns a specific tier of the benchmark.
    
    Args:
        tier_number: 1, 2, or 3
    
    Returns:
        List of (prompt, answer, category, tier) tuples.
    """
    tier_map = {
        1: TIER1_HIGH_SUPPRESSION,
        2: TIER2_BORDERLINE,
        3: TIER3_KNOWLEDGE_GAP,
    }
    data = tier_map.get(tier_number, [])
    tier_name = f"tier{tier_number}"
    return [(p, a, c, tier_name) for p, a, c in data]

def get_prompts_and_answers(tier_number=None):
    """
    Returns prompts and answers for benchmarking.
    
    Args:
        tier_number: 1, 2, or 3. None returns all tiers.
    
    Returns:
        (prompts, answers) tuple of lists.
    """
    if tier_number:
        data = get_tier(tier_number)
    else:
        data = HALLBENCH_V2
    
    prompts = [p for p, a, c, t in data]
    answers = [a for p, a, c, t in data]
    return prompts, answers