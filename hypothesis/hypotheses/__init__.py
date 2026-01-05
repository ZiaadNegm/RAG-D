"""
Hypothesis implementations.

Each hypothesis is implemented as a class inheriting from HypothesisTest.
See template.py for the structure and examples.
"""

from hypothesis.hypotheses.template import (
    HypothesisTest,
    HypothesisResult,
    H5_DispersionMisses,
    H4_ConcentrationRedundancy,
    run_hypothesis,
    AVAILABLE_HYPOTHESES,
)

__all__ = [
    'HypothesisTest',
    'HypothesisResult',
    'H5_DispersionMisses',
    'H4_ConcentrationRedundancy',
    'run_hypothesis',
    'AVAILABLE_HYPOTHESES',
]
