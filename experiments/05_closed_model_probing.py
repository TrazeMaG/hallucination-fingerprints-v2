"""
Experiment 5 — Closed Model Behavioural Probing
=================================================
Goal: Find behavioural signatures of Last-Layer Suppression
in closed models (GPT-3.5-turbo) without internal access.

Method 1: Logit Bias Probing
Suppress high-frequency structural tokens ("the", "a", "in")
that win at the final layer in open models.
If the correct answer emerges more often — suppression exists.

Method 2: Chain of Thought Conflict Detection
Compare direct answer vs reasoned answer.
If model reasons correctly but answers wrongly — suppression.

Method 3: Consistency Probing
Ask the same question multiple ways.
If accuracy varies dramatically — surface-form dependency exists.
"""

import openai
import json
from datetime import datetime
import time

# ── Setup ─────────────────────────────────────────────────────────

API_KEY = input("Enter your OpenAI API key: ").strip()
client = openai.OpenAI(api_key=API_KEY)

MODEL = "gpt-3.5-turbo"

PROMPTS = [
    ("What is the capital of France?", "Paris", "capitals"),
    ("What is the capital of Germany?", "Berlin", "capitals"),
    ("What is the capital of Japan?", "Tokyo", "capitals"),
    ("What is the capital of Australia?", "Canberra", "capitals"),
    ("What is the capital of Brazil?", "Brasilia", "capitals"),
    ("What is the capital of China?", "Beijing", "capitals"),
    ("What is the capital of India?", "New Delhi", "capitals"),
    ("What is the capital of Russia?", "Moscow", "capitals"),
    ("What is the capital of Canada?", "Ottawa", "capitals"),
    ("What is the capital of Argentina?", "Buenos Aires", "capitals"),
    ("In what year did the Berlin Wall fall?", "1989", "history"),
    ("What is water made of?", "hydrogen and oxygen", "science"),
    ("Who proposed the theory of evolution?", "Darwin", "science"),
    ("What is the chemical symbol for gold?", "Au", "science"),
    ("Who was the first president of the United States?", "Washington", "history"),
    ("Who wrote Hamlet?", "Shakespeare", "literature"),
    ("What did Albert Einstein discover?", "relativity", "science"),
    ("What is the capital of Italy?", "Rome", "capitals"),
    ("What is the capital of Spain?", "Madrid", "capitals"),
    ("What is the capital of Canada?", "Ottawa", "capitals"),
]

def call_api(prompt, system=None, max_tokens=10,
             logit_bias=None, temperature=0):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logit_bias:
        kwargs["logit_bias"] = logit_bias

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2)
        return ""

def answer_correct(response, correct):
    return correct.lower() in response.lower()

# ── Method 1: Direct vs Suppressed ───────────────────────────────

print(f"\n{'='*90}")
print(f"METHOD 1: Logit Bias Probing — {MODEL}")
print(f"Suppressing structural tokens to reveal hidden factual knowledge")
print(f"{'='*90}")
print(f"{'Question':<45} {'Direct':<20} {'Suppressed':<20} {'Change'}")
print("-" * 90)

# Token IDs for common structural tokens in GPT tokenizer
# "the"=1the, "a"=264, "an"=459, "it"=433, "I"=40
SUPPRESS_TOKENS = {
    "1": -50,    # common completions
    "2": -50,
    "The": -30,
    "It": -30,
    "A": -30,
}

# Use tiktoken to get actual token IDs
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    suppress_ids = {}
    for word in [" the", " a", " an", " it", " is", " was", "The"]:
        ids = enc.encode(word)
        for id_ in ids:
            suppress_ids[id_] = -50
except:
    suppress_ids = {
        1: -30, 257: -30, 262: -30,
        319: -30, 287: -30, 284: -30
    }

method1_results = []

system_direct = "Answer with just the answer, nothing else. Be concise."

for question, correct, category in PROMPTS[:15]:
    time.sleep(0.5)

    direct = call_api(question, system=system_direct, max_tokens=15)
    time.sleep(0.5)
    suppressed = call_api(
        question,
        system=system_direct,
        max_tokens=15,
        logit_bias=suppress_ids
    )

    direct_correct = answer_correct(direct, correct)
    suppressed_correct = answer_correct(suppressed, correct)

    changed = direct_correct != suppressed_correct
    change_str = ""
    if not direct_correct and suppressed_correct:
        change_str = "IMPROVED +"
    elif direct_correct and not suppressed_correct:
        change_str = "DEGRADED -"
    else:
        change_str = "same"

    method1_results.append({
        "question": question,
        "correct": correct,
        "direct": direct,
        "suppressed": suppressed,
        "direct_correct": direct_correct,
        "suppressed_correct": suppressed_correct,
        "improved": not direct_correct and suppressed_correct,
        "degraded": direct_correct and not suppressed_correct,
    })

    q_short = question[:43]
    d_short = direct[:18]
    s_short = suppressed[:18]
    print(f"{q_short:<45} {d_short:<20} {s_short:<20} {change_str}")

n_improved = sum(1 for r in method1_results if r["improved"])
n_degraded = sum(1 for r in method1_results if r["degraded"])
n_direct_correct = sum(1 for r in method1_results if r["direct_correct"])
n_suppressed_correct = sum(
    1 for r in method1_results if r["suppressed_correct"]
)

print(f"\nDirect accuracy:     {n_direct_correct}/15")
print(f"Suppressed accuracy: {n_suppressed_correct}/15")
print(f"Improved:            {n_improved} cases")
print(f"Degraded:            {n_degraded} cases")

# ── Method 2: CoT Conflict Detection ─────────────────────────────

print(f"\n{'='*90}")
print(f"METHOD 2: Chain of Thought Conflict Detection")
print(f"If model reasons correctly but answers directly wrongly — suppression")
print(f"{'='*90}")
print(f"{'Question':<40} {'Direct':<15} {'CoT Answer':<20} {'Conflict'}")
print("-" * 90)

method2_results = []

system_cot = """Think step by step, then give your final answer.
Format: REASONING: [your reasoning] ANSWER: [final answer only]"""

for question, correct, category in PROMPTS[:15]:
    time.sleep(0.5)
    direct = call_api(question, system=system_direct, max_tokens=10)

    time.sleep(0.5)
    cot_response = call_api(
        question, system=system_cot, max_tokens=100
    )

    cot_answer = ""
    if "ANSWER:" in cot_response:
        cot_answer = cot_response.split("ANSWER:")[-1].strip()
    else:
        cot_answer = cot_response

    direct_correct = answer_correct(direct, correct)
    cot_correct = answer_correct(cot_answer, correct)
    conflict = not direct_correct and cot_correct

    method2_results.append({
        "question": question,
        "correct": correct,
        "direct": direct,
        "cot_answer": cot_answer,
        "direct_correct": direct_correct,
        "cot_correct": cot_correct,
        "conflict": conflict,
    })

    conflict_str = "CONFLICT - SUPPRESSION" if conflict else "consistent"
    q_short = question[:38]
    print(f"{q_short:<40} {direct[:13]:<15} {cot_answer[:18]:<20} "
          f"{conflict_str}")

n_conflicts = sum(1 for r in method2_results if r["conflict"])
n_cot_correct = sum(1 for r in method2_results if r["cot_correct"])
n_direct_correct2 = sum(
    1 for r in method2_results if r["direct_correct"]
)

print(f"\nDirect accuracy:  {n_direct_correct2}/15")
print(f"CoT accuracy:     {n_cot_correct}/15")
print(f"Conflicts:        {n_conflicts} cases")
print(f"(conflict = model knows answer in CoT but not direct)")

# ── Method 3: Prompt Sensitivity ─────────────────────────────────

print(f"\n{'='*90}")
print(f"METHOD 3: Prompt Sensitivity")
print(f"Same fact, different phrasing — does accuracy vary?")
print(f"{'='*90}")

sensitivity_prompts = [
    [
        ("What is the capital of Australia?", "Canberra"),
        ("Australia's capital city is?", "Canberra"),
        ("Name the capital of Australia.", "Canberra"),
        ("The capital city of Australia is", "Canberra"),
    ],
    [
        ("Who proposed the theory of evolution?", "Darwin"),
        ("The theory of evolution was proposed by", "Darwin"),
        ("Name the scientist who proposed evolution.", "Darwin"),
        ("Evolution theory was developed by which scientist?", "Darwin"),
    ],
    [
        ("What is the capital of Canada?", "Ottawa"),
        ("Canada's capital is?", "Ottawa"),
        ("Name the capital city of Canada.", "Ottawa"),
        ("The capital of Canada is", "Ottawa"),
    ],
]

method3_results = []

for fact_group in sensitivity_prompts:
    print(f"\nFact: '{fact_group[0][1]}'")
    group_results = []
    for prompt, correct in fact_group:
        time.sleep(0.5)
        response = call_api(prompt, system=system_direct, max_tokens=15)
        correct_flag = answer_correct(response, correct)
        group_results.append({
            "prompt": prompt,
            "response": response,
            "correct": correct_flag,
        })
        marker = "✓" if correct_flag else "✗"
        print(f"  {marker} '{prompt[:60]}' → '{response[:20]}'")

    accuracy = sum(1 for r in group_results if r["correct"])
    print(f"  Accuracy: {accuracy}/{len(fact_group)} phrasings")
    method3_results.append({
        "fact": fact_group[0][1],
        "accuracy": accuracy,
        "total": len(fact_group),
        "results": group_results
    })

# ── Summary ───────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"CLOSED MODEL PROBING SUMMARY — {MODEL}")
print(f"{'='*90}")

print(f"\nMethod 1 — Logit Bias Probing:")
print(f"  Direct accuracy:     {n_direct_correct}/15 "
      f"({n_direct_correct/15*100:.0f}%)")
print(f"  Suppressed accuracy: {n_suppressed_correct}/15 "
      f"({n_suppressed_correct/15*100:.0f}%)")
print(f"  Cases improved:      {n_improved}")
change = (n_suppressed_correct - n_direct_correct) / 15 * 100
print(f"  Net change:          {change:+.0f}%")

print(f"\nMethod 2 — CoT Conflict Detection:")
print(f"  Direct accuracy:  {n_direct_correct2}/15 "
      f"({n_direct_correct2/15*100:.0f}%)")
print(f"  CoT accuracy:     {n_cot_correct}/15 "
      f"({n_cot_correct/15*100:.0f}%)")
print(f"  Conflicts:        {n_conflicts}/15")
print(f"  (model knew answer via reasoning but not direct output)")

print(f"\nMethod 3 — Prompt Sensitivity:")
for r in method3_results:
    print(f"  '{r['fact']}': {r['accuracy']}/{r['total']} phrasings correct")

if n_improved > 0:
    print(f"\nBEHAVIOURAL EVIDENCE OF SUPPRESSION IN {MODEL}:")
    print(f"Suppressing structural tokens improved accuracy in "
          f"{n_improved} cases.")
    print(f"This is consistent with Last-Layer Suppression.")

if n_conflicts > 0:
    print(f"\nCHAIN OF THOUGHT CONFLICTS DETECTED: {n_conflicts} cases")
    print(f"Model reasoned correctly but output wrongly.")
    print(f"Consistent with suppression of internally retrieved facts.")

# ── Save ──────────────────────────────────────────────────────────

output = {
    "model": MODEL,
    "timestamp": datetime.now().isoformat(),
    "method1_logit_bias": {
        "direct_accuracy": n_direct_correct / 15,
        "suppressed_accuracy": n_suppressed_correct / 15,
        "improved": n_improved,
        "degraded": n_degraded,
        "results": method1_results,
    },
    "method2_cot_conflict": {
        "direct_accuracy": n_direct_correct2 / 15,
        "cot_accuracy": n_cot_correct / 15,
        "conflicts": n_conflicts,
        "results": method2_results,
    },
    "method3_sensitivity": {
        "results": method3_results,
    },
}

with open("results/experiment_05_closed_model.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/experiment_05_closed_model.json")
print("Experiment 5 complete.")