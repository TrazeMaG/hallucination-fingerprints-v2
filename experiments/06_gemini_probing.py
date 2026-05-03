"""
Experiment 6 — Gemini Behavioural Probing
==========================================
Goal: Find behavioural signatures of Last-Layer Suppression
in Google Gemini 2.5 Flash without internal access.
"""

import google.generativeai as genai
import json
import time
from datetime import datetime

API_KEY = input("Enter your Gemini API key: ").strip()
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

FACTS = [
    ("The capital of France is", "What is the capital of France?", "Paris"),
    ("The capital of Germany is", "What is the capital of Germany?", "Berlin"),
    ("The capital of Japan is", "What is the capital of Japan?", "Tokyo"),
    ("The capital of Australia is", "What is the capital of Australia?", "Canberra"),
    ("The capital of Russia is", "What is the capital of Russia?", "Moscow"),
    ("The capital of China is", "What is the capital of China?", "Beijing"),
    ("The capital of Brazil is", "What is the capital of Brazil?", "Brasilia"),
    ("The capital of India is", "What is the capital of India?", "New Delhi"),
    ("The capital of Canada is", "What is the capital of Canada?", "Ottawa"),
    ("The capital of Argentina is", "What is the capital of Argentina?", "Buenos Aires"),
    ("The Berlin Wall fell in", "In what year did the Berlin Wall fall?", "1989"),
    ("Water is made of hydrogen and", "What is water made of?", "oxygen"),
    ("The theory of evolution was proposed by", "Who proposed the theory of evolution?", "Darwin"),
    ("The chemical symbol for gold is", "What is the chemical symbol for gold?", "Au"),
    ("The first president of the United States was", "Who was the first president of the United States?", "Washington"),
]

SAFETY_OFF = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def call_gemini(prompt, temperature=0.0):
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=50,
            ),
            safety_settings=SAFETY_OFF
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return ""
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(2)
        return ""

def is_correct(response, answer):
    return answer.lower() in response.lower()

# ── Method 1: Completion vs Question ─────────────────────────────

print(f"\n{'='*95}")
print(f"METHOD 1: Completion Mode vs Question Mode")
print(f"Key test: does Gemini hallucinate on sentence completion")
print(f"but answer correctly when asked directly?")
print(f"{'='*95}")
print(f"{'Fact':<35} {'Completion':<20} {'Question':<20} {'Conflict'}")
print("-" * 95)

method1_results = []

for completion, question, answer in FACTS:
    time.sleep(1)

    comp_prompt = f"Complete this sentence with just the answer, nothing else: '{completion}'"
    comp_response = call_gemini(comp_prompt)

    time.sleep(1)

    q_prompt = f"Answer with just the answer, nothing else: {question}"
    q_response = call_gemini(q_prompt)

    comp_correct = is_correct(comp_response, answer)
    q_correct = is_correct(q_response, answer)
    conflict = not comp_correct and q_correct

    method1_results.append({
        "completion_prompt": completion,
        "question": question,
        "answer": answer,
        "completion_response": comp_response,
        "question_response": q_response,
        "completion_correct": comp_correct,
        "question_correct": q_correct,
        "conflict": conflict,
    })

    conflict_str = "CONFLICT - SUPPRESSION SIGNAL" if conflict else "consistent"
    fact_short = completion[:33]
    print(f"{fact_short:<35} {comp_response[:18]:<20} {q_response[:18]:<20} {conflict_str}")

n_conflicts = sum(1 for r in method1_results if r["conflict"])
n_comp_correct = sum(1 for r in method1_results if r["completion_correct"])
n_q_correct = sum(1 for r in method1_results if r["question_correct"])

print(f"\nCompletion accuracy: {n_comp_correct}/{len(FACTS)}")
print(f"Question accuracy:   {n_q_correct}/{len(FACTS)}")
print(f"Conflicts:           {n_conflicts}/{len(FACTS)}")

# ── Method 2: CoT Conflict ────────────────────────────────────────

print(f"\n{'='*95}")
print(f"METHOD 2: Direct vs Chain of Thought Conflict")
print(f"{'='*95}")
print(f"{'Question':<40} {'Direct':<20} {'CoT':<20} {'Conflict'}")
print("-" * 95)

method2_results = []

for completion, question, answer in FACTS[:10]:
    time.sleep(1)

    direct_prompt = f"Answer in one word or number only: {question}"
    direct = call_gemini(direct_prompt)

    time.sleep(1)

    cot_prompt = f"Think step by step about this question, then give your final answer on the last line starting with 'ANSWER:'\n\n{question}"
    cot_response = call_gemini(cot_prompt, temperature=0.0)

    cot_answer = ""
    if "ANSWER:" in cot_response.upper():
        cot_answer = cot_response.upper().split("ANSWER:")[-1].strip()
    else:
        cot_answer = cot_response.split("\n")[-1].strip()

    direct_correct = is_correct(direct, answer)
    cot_correct = is_correct(cot_answer, answer)
    conflict = not direct_correct and cot_correct

    method2_results.append({
        "question": question,
        "answer": answer,
        "direct": direct,
        "cot_answer": cot_answer,
        "direct_correct": direct_correct,
        "cot_correct": cot_correct,
        "conflict": conflict,
    })

    conflict_str = "CONFLICT" if conflict else "consistent"
    print(f"{question[:38]:<40} {direct[:18]:<20} {cot_answer[:18]:<20} {conflict_str}")

n_cot_conflicts = sum(1 for r in method2_results if r["conflict"])
print(f"\nDirect accuracy: {sum(1 for r in method2_results if r['direct_correct'])}/10")
print(f"CoT accuracy:    {sum(1 for r in method2_results if r['cot_correct'])}/10")
print(f"Conflicts:       {n_cot_conflicts}/10")

# ── Method 3: Prompt Sensitivity ─────────────────────────────────

print(f"\n{'='*95}")
print(f"METHOD 3: Prompt Sensitivity")
print(f"{'='*95}")

sensitivity_groups = [
    {
        "fact": "Canberra",
        "prompts": [
            "The capital of Australia is",
            "What is the capital of Australia?",
            "Australia's capital city is",
            "Name the capital of Australia.",
        ]
    },
    {
        "fact": "Darwin",
        "prompts": [
            "The theory of evolution was proposed by",
            "Who proposed the theory of evolution?",
            "Evolution was proposed by",
            "Name the scientist who proposed evolution.",
        ]
    },
    {
        "fact": "1989",
        "prompts": [
            "The Berlin Wall fell in",
            "In what year did the Berlin Wall fall?",
            "The Berlin Wall came down in",
            "When did the Berlin Wall fall?",
        ]
    },
]

method3_results = []

for group in sensitivity_groups:
    fact = group["fact"]
    print(f"\nFact: '{fact}'")
    group_correct = 0
    group_results = []

    for prompt in group["prompts"]:
        time.sleep(1)
        if prompt.endswith("?"):
            full_prompt = f"Answer in one word only: {prompt}"
        else:
            full_prompt = f"Complete with just the answer: '{prompt}'"
        response = call_gemini(full_prompt)
        correct = is_correct(response, fact)
        if correct:
            group_correct += 1
        group_results.append({
            "prompt": prompt,
            "response": response,
            "correct": correct,
        })
        marker = "Y" if correct else "N"
        print(f"  [{marker}] '{prompt[:55]}' -> '{response[:20]}'")

    print(f"  Accuracy: {group_correct}/{len(group['prompts'])} phrasings")
    method3_results.append({
        "fact": fact,
        "accuracy": group_correct,
        "total": len(group["prompts"]),
        "results": group_results,
    })

# ── Summary ───────────────────────────────────────────────────────

print(f"\n{'='*95}")
print(f"GEMINI 2.5 FLASH BEHAVIOURAL PROBING SUMMARY")
print(f"{'='*95}")

print(f"\nMethod 1 - Completion vs Question:")
print(f"  Completion accuracy: {n_comp_correct}/{len(FACTS)} ({n_comp_correct/len(FACTS)*100:.0f}%)")
print(f"  Question accuracy:   {n_q_correct}/{len(FACTS)} ({n_q_correct/len(FACTS)*100:.0f}%)")
print(f"  Conflicts:           {n_conflicts}/{len(FACTS)}")
if n_q_correct > n_comp_correct:
    diff = n_q_correct - n_comp_correct
    print(f"  {diff} facts answered correctly as questions but wrongly as completions")
    print(f"  Consistent with Last-Layer Suppression in completion mode")

print(f"\nMethod 2 - Direct vs CoT:")
print(f"  CoT conflicts: {n_cot_conflicts}/10")

print(f"\nMethod 3 - Prompt Sensitivity:")
for r in method3_results:
    sensitivity = r["total"] - r["accuracy"]
    print(f"  '{r['fact']}': {r['accuracy']}/{r['total']} ({'sensitive' if sensitivity > 0 else 'robust'})")

# ── Save ──────────────────────────────────────────────────────────

output = {
    "model": "gemini-2.5-flash",
    "timestamp": datetime.now().isoformat(),
    "method1_completion_vs_question": {
        "completion_accuracy": n_comp_correct / len(FACTS),
        "question_accuracy": n_q_correct / len(FACTS),
        "conflicts": n_conflicts,
        "results": method1_results,
    },
    "method2_cot_conflict": {
        "conflicts": n_cot_conflicts,
        "results": method2_results,
    },
    "method3_sensitivity": {
        "results": method3_results,
    },
}

with open("results/experiment_06_gemini_probing.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to results/experiment_06_gemini_probing.json")
print("Experiment 6 complete.")