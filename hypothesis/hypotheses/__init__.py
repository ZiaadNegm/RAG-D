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

from hypothesis.hypotheses.h3_phase2_recall import H3_Phase2_Recall
from hypothesis.hypotheses.h3_colocation import H3_Colocation, FallbackAnalysisResult

__all__ = [
    'HypothesisTest',
    'HypothesisResult',
    'H5_DispersionMisses',
    'H4_ConcentrationRedundancy',
    'H3_Phase2_Recall',
    'H3_Colocation',
    'FallbackAnalysisResult',
    'run_hypothesis',
    'AVAILABLE_HYPOTHESES',
]
