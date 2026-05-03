"""
HallScope — Hallucination Interpretability Library
====================================================
The standard tool for hallucination analysis in transformer
language models.

pip install hallscope

Usage:
    from hallscope import HallScope
    
    hs = HallScope("gpt2-xl")
    report = hs.analyse("The capital of France is")
    print(report.suppression_ratio)      # 18.6x
    print(report.hallucination_type)     # TYPE2A_SUPPRESSION
    corrected = hs.correct("The capital of France is")
    print(corrected)                     # Paris
"""

from .core import HallScope
from .report import HallReport

__version__ = "0.1.0"
__author__ = "Nikhil Upadhyay"
__all__ = ["HallScope", "HallReport"]