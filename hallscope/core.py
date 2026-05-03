"""
HallScope Core — Main analysis engine
"""

import torch
from transformer_lens import HookedTransformer
from .report import HallReport


class HallScope:
    """
    Main HallScope class. Wraps any TransformerLens-supported model
    and provides hallucination analysis, correction, and benchmarking.

    Usage:
        hs = HallScope("gpt2-xl")
        report = hs.analyse("The capital of France is", "Paris")
        corrected = hs.correct("The capital of France is")
        comparison = hs.compare(
            "The capital of France is",
            ["gpt2", "gpt2-medium", "gpt2-xl"]
        )
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Load a model for hallucination analysis.

        Args:
            model_name: Any TransformerLens-supported model name.
                       e.g. "gpt2-xl", "EleutherAI/gpt-neo-2.7B",
                            "microsoft/phi-2", "Qwen/Qwen1.5-1.8B"
            device: "cuda" or "cpu". Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"HallScope: Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        print(f"HallScope: Ready. {self.n_layers} layers, "
              f"{self.model.cfg.d_model}d")

    def _get_token_id(self, answer: str) -> int:
        try:
            return self.model.to_single_token(f" {answer}")
        except Exception:
            tokens = self.model.to_tokens(f" {answer}")[0]
            return tokens[1].item()

    def _get_layer_probs(self, cache, token_id: int) -> list:
        probs = []
        for layer in range(self.n_layers):
            resid = cache[
                f"blocks.{layer}.hook_resid_post"
            ][0, -1]
            resid_normed = self.model.ln_final(resid.unsqueeze(0))[0]
            logits = resid_normed @ self.model.W_U + self.model.b_U
            prob = torch.softmax(logits, dim=-1)[token_id].item()
            probs.append(prob)
        return probs

    def analyse(self, prompt: str, correct_answer: str = None) -> "HallReport":
        """
        Run full hallucination analysis on a prompt.

        Args:
            prompt: The text prompt to analyse.
            correct_answer: The expected correct answer (optional).
                           If not provided, analysis is done on top
                           prediction only.

        Returns:
            HallReport object with full analysis results.
        """
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(prompt)

        final_probs = torch.softmax(logits[0, -1], dim=-1)
        predicted = self.model.to_string(logits[0, -1].argmax()).strip()

        if correct_answer:
            token_id = self._get_token_id(correct_answer)
            layer_probs = self._get_layer_probs(cache, token_id)
            correct_final_prob = final_probs[token_id].item()
            correct_final_rank = (
                final_probs > final_probs[token_id]
            ).sum().item() + 1
        else:
            token_id = logits[0, -1].argmax().item()
            layer_probs = self._get_layer_probs(cache, token_id)
            correct_final_prob = final_probs[token_id].item()
            correct_final_rank = 1

        peak_layer = layer_probs.index(max(layer_probs))
        peak_prob = max(layer_probs)
        final_prob = layer_probs[-1]
        suppression_ratio = peak_prob / (final_prob + 1e-10)
        relative_depth = peak_layer / self.n_layers

        is_correct = (
            predicted.lower() == correct_answer.lower()
            if correct_answer else True
        )

        if is_correct:
            hall_type = "CORRECT"
        elif correct_final_rank <= 10:
            hall_type = "TYPE2A_SUPPRESSION"
        else:
            hall_type = "TYPE2B_GAP"

        survival_probability = min(
            1.0, peak_prob / 0.15
        ) if peak_prob < 0.15 else 1.0

        return HallReport(
            prompt=prompt,
            predicted=predicted,
            correct_answer=correct_answer,
            is_correct=is_correct,
            hallucination_type=hall_type,
            peak_layer=peak_layer,
            peak_layer_relative=round(relative_depth, 3),
            peak_prob=round(peak_prob, 4),
            final_prob=round(final_prob, 4),
            suppression_ratio=round(suppression_ratio, 2),
            correct_final_rank=correct_final_rank,
            layer_probs=layer_probs,
            n_layers=self.n_layers,
            model_name=self.model_name,
            survival_probability=round(survival_probability, 3),
            cache=cache,
        )

    def correct(self, prompt: str, alpha: float = 0.5) -> str:
        """
        Generate a corrected prediction using logit blending.
        Blends peak-layer logits with final-layer logits to
        bypass Last-Layer Suppression.

        Args:
            prompt: The text prompt.
            alpha: Blend weight for peak layer (0=no correction,
                   1=peak layer only). Default 0.5 works well for
                   strong suppression models (GPT-2, Phi-2, Qwen).

        Returns:
            Corrected predicted token as string.
        """
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(prompt)

        predicted_token_id = logits[0, -1].argmax().item()
        layer_probs = self._get_layer_probs(
            cache, predicted_token_id
        )
        peak_layer = layer_probs.index(max(layer_probs))

        resid = cache[
            f"blocks.{peak_layer}.hook_resid_post"
        ][0, -1]
        resid_normed = self.model.ln_final(resid.unsqueeze(0))[0]
        peak_logits = resid_normed @ self.model.W_U + self.model.b_U

        blended = alpha * peak_logits + (1 - alpha) * logits[0, -1]
        return self.model.to_string(blended.argmax()).strip()

    def compare(self, prompt: str, models: list) -> dict:
        """
        Compare hallucination analysis across multiple models.

        Args:
            prompt: The text prompt to analyse.
            models: List of model names to compare.

        Returns:
            Dictionary of model name to HallReport.
        """
        current_model = self.model_name
        results = {}

        for model_name in models:
            if model_name == current_model:
                report = self.analyse(prompt)
            else:
                hs = HallScope(model_name)
                report = hs.analyse(prompt)
                del hs

            results[model_name] = {
                "suppression_ratio": report.suppression_ratio,
                "peak_layer": report.peak_layer,
                "peak_layer_relative": report.peak_layer_relative,
                "hallucination_type": report.hallucination_type,
                "predicted": report.predicted,
            }

        return results

    def benchmark(
        self,
        prompts: list,
        answers: list,
        alphas: list = None
    ) -> dict:
        """
        Run benchmark evaluation with intervention.

        Args:
            prompts: List of prompt strings.
            answers: List of correct answer strings.
            alphas: List of alpha values to test.

        Returns:
            Dictionary of results per alpha value.
        """
        if alphas is None:
            alphas = [0.0, 0.1, 0.3, 0.5]

        alpha_correct = {a: 0 for a in alphas}

        for prompt, answer in zip(prompts, answers):
            token_id = self._get_token_id(answer)

            with torch.no_grad():
                final_logits, cache = self.model.run_with_cache(prompt)

            layer_probs = self._get_layer_probs(cache, token_id)
            peak_layer = layer_probs.index(max(layer_probs))

            resid = cache[
                f"blocks.{peak_layer}.hook_resid_post"
            ][0, -1]
            resid_normed = self.model.ln_final(resid.unsqueeze(0))[0]
            peak_logits = (
                resid_normed @ self.model.W_U + self.model.b_U
            )

            for alpha in alphas:
                if alpha == 0.0:
                    pred_logits = final_logits[0, -1]
                else:
                    pred_logits = (
                        alpha * peak_logits +
                        (1 - alpha) * final_logits[0, -1]
                    )
                pred = self.model.to_string(
                    pred_logits.argmax()
                ).strip()
                if pred.lower() == answer.lower():
                    alpha_correct[alpha] += 1

        n = len(prompts)
        return {
            str(a): {
                "correct": alpha_correct[a],
                "accuracy": alpha_correct[a] / n,
            }
            for a in alphas
        }