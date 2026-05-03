"""
HallReport — Structured analysis results
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class HallReport:
    """
    Complete hallucination analysis report for a single prompt.

    Attributes:
        prompt: The input prompt.
        predicted: The model's predicted next token.
        correct_answer: The expected correct answer (if provided).
        is_correct: Whether the prediction matches the correct answer.
        hallucination_type: One of:
            - "CORRECT": Prediction is correct.
            - "TYPE2A_SUPPRESSION": Correct answer exists in top-10
              but was suppressed by final layer.
            - "TYPE2B_GAP": Correct answer not in top-10 — model
              likely never learned this fact.
        peak_layer: Layer index where correct answer probability peaks.
        peak_layer_relative: Peak layer as fraction of total layers.
            Consistently ~0.83 in strong suppression models.
        peak_prob: Probability of correct answer at peak layer.
        final_prob: Probability of correct answer at final layer.
        suppression_ratio: peak_prob / final_prob.
            >2x = strong suppression (intervention likely helps).
            ~1x = weak/no suppression (intervention unlikely to help).
        correct_final_rank: Rank of correct answer at final layer.
        layer_probs: List of correct answer probabilities at each layer.
        n_layers: Total number of layers in the model.
        model_name: Name of the model used.
        survival_probability: Estimated probability that the correct
            answer would survive suppression (based on peak_prob).
        cache: Raw TransformerLens cache (for advanced analysis).
    """

    prompt: str
    predicted: str
    correct_answer: Optional[str]
    is_correct: bool
    hallucination_type: str
    peak_layer: int
    peak_layer_relative: float
    peak_prob: float
    final_prob: float
    suppression_ratio: float
    correct_final_rank: int
    layer_probs: List[float]
    n_layers: int
    model_name: str
    survival_probability: float
    cache: object = field(repr=False)

    def __str__(self):
        lines = [
            f"HallScope Analysis",
            f"  Model:              {self.model_name}",
            f"  Prompt:             {self.prompt}",
            f"  Predicted:          {self.predicted}",
            f"  Correct answer:     {self.correct_answer}",
            f"  Is correct:         {self.is_correct}",
            f"  Hallucination type: {self.hallucination_type}",
            f"  Peak factual layer: Block {self.peak_layer} "
            f"({self.peak_layer_relative:.0%} depth)",
            f"  Peak probability:   {self.peak_prob:.4f}",
            f"  Final probability:  {self.final_prob:.4f}",
            f"  Suppression ratio:  {self.suppression_ratio:.1f}x",
            f"  Correct answer rank:{self.correct_final_rank}",
            f"  Survival estimate:  {self.survival_probability:.2f}",
        ]
        return "\n".join(lines)

    @property
    def is_type2a(self) -> bool:
        return self.hallucination_type == "TYPE2A_SUPPRESSION"

    @property
    def is_type2b(self) -> bool:
        return self.hallucination_type == "TYPE2B_GAP"

    @property
    def strong_suppression(self) -> bool:
        return self.suppression_ratio > 2.0

    @property
    def intervention_recommended(self) -> bool:
        return self.is_type2a and self.strong_suppression