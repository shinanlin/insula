"""Semantic / lexical-semantic analyses for Lexical Delay.

See README.md and design.md for the claim ladder and analysis plan.
"""

from src.semantic.features import build_stimulus_table
from src.semantic.rsa import pairwise_rdm, rsa_spearman

__all__ = [
    "build_stimulus_table",
    "pairwise_rdm",
    "rsa_spearman",
]
