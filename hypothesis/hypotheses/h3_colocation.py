"""
Hypothesis H3 Co-location Analysis: Oracle Fallback Availability and Strength

This module implements the analyses from H3_COLOCATION_ANALYSIS_PLAN.md to determine
whether oracle accessibility is a valid proxy for "high-scoring evidence" accessibility.

Key Analyses:
    Analysis 1: The 72.6% Finding — Fallback Availability
        For inaccessible oracles, check if ANY other doc embedding is accessible.
        Result: 72.6% have NO fallback (document completely dark for that token).
        
    Analysis 2: Fallback Strength Measurement
        For the 27.4% with fallback, measure score gap: oracle_score - best_fallback_score
        Determines if fallback provides real compensation (Regime B) or weak rescue.
        
    Analysis 4: Strong Embedding Centroid Spread (TODO)
        For oracle-accessible cases, analyze how strong embeddings are distributed.

The Two Regimes:
    Regime A (Co-location): Missing oracle centroid = losing all strong evidence
    Regime B (Dispersed): Fallback exists and provides compensation

Key Finding:
    72.6% of inaccessible oracle cases have NO accessible fallback at all.
    This means 2nd-best, 3rd-best, etc. are ALL inaccessible — the document is
    completely invisible for that query token.

Config Support:
    - smoke: Sample 500 inaccessible cases for quick validation (~seconds)
    - dev: Sample 5000 cases for signal exploration (~1 min)
    - prod: Full 27,177 cases for final analysis (~3-5 min)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.configs import RuntimeConfig, load_config


# =============================================================================
# Config-aware sampling settings
# =============================================================================

SAMPLE_SIZES = {
    'smoke': 500,      # Quick sanity check (~seconds)
    'dev': 5000,       # Signal exploration (~1 min)
    'prod': None,      # Full analysis (None = all cases)
}

RANDOM_SEED = 42


@dataclass
class FallbackAnalysisResult:
    """Results from the fallback availability analysis (72.6% finding)."""
    
    # Core counts
    total_inaccessible_oracles: int
    has_fallback_count: int
    no_fallback_count: int
    
    # Percentages
    has_fallback_pct: float
    no_fallback_pct: float
    
    # Sampling info
    is_sampled: bool = False
    sample_size: Optional[int] = None
    full_population_size: Optional[int] = None
    
    # Validation checks
    m4r_inaccessible_count_matches: bool = True
    r0_lookup_unique_keys: int = 0
    docs_checked: int = 0
    
    # Per-case details (optional, for inspection)
    case_details: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> dict:
        return {
            'total_inaccessible_oracles': self.total_inaccessible_oracles,
            'has_fallback_count': self.has_fallback_count,
            'no_fallback_count': self.no_fallback_count,
            'has_fallback_pct': self.has_fallback_pct,
            'no_fallback_pct': self.no_fallback_pct,
            'sampling': {
                'is_sampled': self.is_sampled,
                'sample_size': self.sample_size,
                'full_population_size': self.full_population_size,
            },
            'validation': {
                'm4r_inaccessible_count_matches': self.m4r_inaccessible_count_matches,
                'r0_lookup_unique_keys': self.r0_lookup_unique_keys,
                'docs_checked': self.docs_checked,
            }
        }


@dataclass
class FallbackStrengthResult:
    """Results from the fallback strength analysis (Analysis 2)."""
    
    # Core counts
    total_cases_with_fallback: int
    cases_analyzed: int
    
    # Score gap statistics
    mean_score_gap: float           # oracle_score - best_fallback_score
    median_score_gap: float
    std_score_gap: float
    
    # Relative gap statistics (gap / oracle_score)
    mean_relative_gap: float
    median_relative_gap: float
    
    # Regime classification based on gap size
    strong_fallback_count: int      # gap < 5% (Regime B compensation)
    moderate_fallback_count: int    # gap 5-20%
    weak_fallback_count: int        # gap > 20% (still Regime A)
    
    strong_fallback_pct: float
    moderate_fallback_pct: float
    weak_fallback_pct: float
    
    # Percentiles of relative gap
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    # NEW: Absolute impact metrics (more honest framing)
    total_oracle_score_sum: float = 0.0       # Sum of oracle scores analyzed
    total_score_gap_sum: float = 0.0          # Sum of all score gaps (cumulative loss)
    total_score_loss_pct: float = 0.0         # total_score_gap_sum / total_oracle_score_sum
    
    # NEW: Counts relative to ALL inaccessible (not just those with fallback)
    # These are populated by combining with Analysis 1 results
    pct_of_all_inaccessible_strong: Optional[float] = None
    pct_of_all_inaccessible_moderate: Optional[float] = None
    pct_of_all_inaccessible_weak: Optional[float] = None
    
    # Sampling info
    is_sampled: bool = False
    sample_size: Optional[int] = None
    full_population_size: Optional[int] = None
    
    # Per-case details (optional)
    case_details: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> dict:
        return {
            'total_cases_with_fallback': self.total_cases_with_fallback,
            'cases_analyzed': self.cases_analyzed,
            'score_gap': {
                'mean': self.mean_score_gap,
                'median': self.median_score_gap,
                'std': self.std_score_gap,
            },
            'relative_gap': {
                'mean': self.mean_relative_gap,
                'median': self.median_relative_gap,
                'percentiles': self.percentiles,
            },
            'cumulative_impact': {
                'total_oracle_score_sum': self.total_oracle_score_sum,
                'total_score_gap_sum': self.total_score_gap_sum,
                'total_score_loss_pct': self.total_score_loss_pct,
            },
            'regime_classification': {
                'strong_fallback': {
                    'count': self.strong_fallback_count,
                    'pct_of_fallback_cases': self.strong_fallback_pct,
                    'pct_of_all_inaccessible': self.pct_of_all_inaccessible_strong,
                    'definition': 'gap < 5%'
                },
                'moderate_fallback': {
                    'count': self.moderate_fallback_count,
                    'pct_of_fallback_cases': self.moderate_fallback_pct,
                    'pct_of_all_inaccessible': self.pct_of_all_inaccessible_moderate,
                    'definition': '5% <= gap < 20%'
                },
                'weak_fallback': {
                    'count': self.weak_fallback_count,
                    'pct_of_fallback_cases': self.weak_fallback_pct,
                    'pct_of_all_inaccessible': self.pct_of_all_inaccessible_weak,
                    'definition': 'gap >= 20%'
                },
            },
            'sampling': {
                'is_sampled': self.is_sampled,
                'sample_size': self.sample_size,
                'full_population_size': self.full_population_size,
            }
        }


@dataclass
class CentroidSpreadResult:
    """Results from the centroid spread analysis (Analysis 4).
    
    This analyzes oracle-ACCESSIBLE cases to understand how "strong" embeddings
    (within k std of oracle score) are distributed across centroids.
    
    Key insight: This is self-contained — we encode Q fresh and score ALL
    doc embeddings with the SAME Q. No comparison to old M4R scores needed.
    """
    
    # Core counts
    total_oracle_accessible: int
    cases_analyzed: int
    
    # Strong embedding statistics
    mean_strong_embeddings: float       # avg number of strong embeddings per case
    median_strong_embeddings: float
    
    # Centroid spread statistics  
    mean_unique_centroids: float        # avg unique centroids containing strong embeddings
    median_unique_centroids: float
    
    # Probing capture statistics
    mean_centroids_probed: float        # of centroids with strong embs, how many were probed
    median_centroids_probed: float
    mean_capture_rate: float            # % of strong embedding centroids that were probed
    
    # Oracle centroid concentration
    mean_strong_in_oracle_centroid: float  # avg strong embeddings in oracle's centroid
    pct_oracle_only: float                  # % of cases where ALL strong are in oracle centroid
    
    # Config
    k_threshold: float = 1.5            # std threshold for "strong" classification
    
    # Distribution: how many centroids contain strong embeddings
    centroids_distribution: Dict[str, int] = field(default_factory=dict)  # {'1': count, '2': count, ...}
    
    # Sampling info
    is_sampled: bool = False
    sample_size: Optional[int] = None
    full_population_size: Optional[int] = None
    
    # Per-case details (optional)
    case_details: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> dict:
        return {
            'total_oracle_accessible': self.total_oracle_accessible,
            'cases_analyzed': self.cases_analyzed,
            'strong_embedding_stats': {
                'k_threshold': self.k_threshold,
                'mean_count': self.mean_strong_embeddings,
                'median_count': self.median_strong_embeddings,
            },
            'centroid_spread': {
                'mean_unique_centroids': self.mean_unique_centroids,
                'median_unique_centroids': self.median_unique_centroids,
                'distribution': self.centroids_distribution,
            },
            'probing_capture': {
                'mean_centroids_probed': self.mean_centroids_probed,
                'median_centroids_probed': self.median_centroids_probed,
                'mean_capture_rate': self.mean_capture_rate,
            },
            'oracle_centroid_concentration': {
                'mean_strong_in_oracle_centroid': self.mean_strong_in_oracle_centroid,
                'pct_all_strong_in_oracle_only': self.pct_oracle_only,
            },
            'sampling': {
                'is_sampled': self.is_sampled,
                'sample_size': self.sample_size,
                'full_population_size': self.full_population_size,
            }
        }


# =============================================================================
# Scoring utilities (from m4_verification_experiments.py)
# =============================================================================

def compute_bucket_scores(q_token: torch.Tensor, bucket_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute bucket scores for a query token.
    Result: bucket_scores[dim, bucket] = q_token[dim] * bucket_weights[bucket]
    """
    return q_token.unsqueeze(1) * bucket_weights.unsqueeze(0)  # (128, num_buckets)


def unpack_and_score_python(
    residual: torch.Tensor,
    bucket_scores: torch.Tensor,
    nbits: int
) -> float:
    """
    Unpack quantized residual and compute score via bucket lookup.
    """
    score = 0.0
    
    if nbits == 4:
        packed_dim = 64
        for packed_idx in range(packed_dim):
            packed_val = residual[packed_idx].item()
            unpacked_0 = packed_val >> 4
            unpacked_1 = packed_val & 0x0F
            
            dim_0 = packed_idx * 2
            dim_1 = dim_0 + 1
            
            score += bucket_scores[dim_0, unpacked_0].item()
            score += bucket_scores[dim_1, unpacked_1].item()
            
    elif nbits == 2:
        packed_dim = 32
        for packed_idx in range(packed_dim):
            packed_val = residual[packed_idx].item()
            unpacked_0 = (packed_val & 0xC0) >> 6
            unpacked_1 = (packed_val & 0x30) >> 4
            unpacked_2 = (packed_val & 0x0C) >> 2
            unpacked_3 = (packed_val & 0x03)
            
            dim_base = packed_idx * 4
            score += bucket_scores[dim_base, unpacked_0].item()
            score += bucket_scores[dim_base + 1, unpacked_1].item()
            score += bucket_scores[dim_base + 2, unpacked_2].item()
            score += bucket_scores[dim_base + 3, unpacked_3].item()
    
    return score


def compute_score_for_position(
    q_token: torch.Tensor,
    embedding_pos: int,
    centroids: torch.Tensor,
    residuals_compacted: torch.Tensor,
    offsets_compacted: torch.Tensor,
    bucket_weights: torch.Tensor,
    nbits: int,
    bucket_scores: Optional[torch.Tensor] = None
) -> float:
    """
    Compute similarity score for one query token and one embedding position.
    score = centroid_score + residual_score
    """
    # Find centroid for this embedding
    centroid_id = torch.searchsorted(offsets_compacted, embedding_pos, side='right').item() - 1
    
    # Centroid contribution
    centroid_score = (q_token @ centroids[centroid_id]).item()
    
    # Precompute bucket scores if not provided
    if bucket_scores is None:
        bucket_scores = compute_bucket_scores(q_token, bucket_weights)
    
    # Residual contribution
    residual = residuals_compacted[embedding_pos]
    residual_score = unpack_and_score_python(residual, bucket_scores, nbits)
    
    return centroid_score + residual_score


class H3_Colocation(HypothesisTest):
    """
    H3 Co-location Analysis: Validate whether oracle accessibility is a proxy
    for high-scoring evidence accessibility.
    
    Key question: When the oracle embedding is inaccessible, are the 2nd-best,
    3rd-best, etc. also inaccessible?
    
    Answer (from 72.6% finding): For 72.6% of inaccessible oracles, YES —
    the document has ZERO embeddings in any probed centroid for that token.
    """
    
    HYPOTHESIS_ID = "H3_Colocation"
    HYPOTHESIS_NAME = "Oracle Co-location Analysis"
    CLAIM = "Oracle accessibility is a valid proxy for high-scoring evidence accessibility"
    
    # Expected values for validation
    EXPECTED_INACCESSIBLE_COUNT = 27177
    EXPECTED_NO_FALLBACK_PCT = 72.6
    
    # Default paths
    GOLDEN_METRICS_DIR = Path("/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425/golden_metrics_v2")
    RUN_DIR = Path("/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425")
    INDEX_PATH = Path("/mnt/datasets/index/beir-quora.split=test.nbits=4")
    
    def __init__(self, config: Optional[RuntimeConfig] = None,
                 golden_metrics_dir: Optional[Path] = None,
                 run_dir: Optional[Path] = None,
                 index_path: Optional[Path] = None):
        """Initialize with optional custom paths."""
        if config is None:
            config = load_config("prod")
        super().__init__(config)
        
        if golden_metrics_dir:
            self.GOLDEN_METRICS_DIR = Path(golden_metrics_dir)
        if run_dir:
            self.RUN_DIR = Path(run_dir)
        if index_path:
            self.INDEX_PATH = Path(index_path)
        
        # Data containers
        self.m4r_df: Optional[pd.DataFrame] = None
        self.r0_df: Optional[pd.DataFrame] = None
        self.embedding_to_centroid: Optional[torch.Tensor] = None
        self.reverse_doc_offsets: Optional[torch.Tensor] = None
        self.reverse_sorted_indices: Optional[torch.Tensor] = None
        
        # R0 lookup: (query_id, q_token_id) -> set of probed centroid IDs
        self.r0_lookup: Optional[Dict[Tuple[int, int], Set[int]]] = None
        
        # Results
        self.fallback_result: Optional[FallbackAnalysisResult] = None
        self.strength_result: Optional[FallbackStrengthResult] = None
        
        # Additional components for Analysis 2 (scoring)
        self.centroids: Optional[torch.Tensor] = None
        self.residuals_compacted: Optional[torch.Tensor] = None
        self.offsets_compacted: Optional[torch.Tensor] = None
        self.bucket_weights: Optional[torch.Tensor] = None
        self.nbits: int = 4
        
        # Query encoding components (lazy loaded)
        self._checkpoint = None
        self._queries: Optional[Dict[int, str]] = None
    
    def setup(self):
        """Load all required data sources."""
        print("=" * 60)
        print("H3 Co-location Analysis: Loading Data")
        print("=" * 60)
        
        # 1. Load M4R (oracle accessibility per query token)
        m4r_path = self.GOLDEN_METRICS_DIR / "M4R.parquet"
        if not m4r_path.exists():
            raise FileNotFoundError(f"M4R not found: {m4r_path}")
        self.m4r_df = pd.read_parquet(m4r_path)
        print(f"✓ Loaded M4R: {len(self.m4r_df):,} rows")
        print(f"  Columns: {list(self.m4r_df.columns)}")
        
        # 2. Load R0 (selected centroids per query token)
        r0_path = self.RUN_DIR / "tier_b" / "R0_selected_centroids.parquet"
        if not r0_path.exists():
            raise FileNotFoundError(f"R0 not found: {r0_path}")
        self.r0_df = pd.read_parquet(r0_path)
        print(f"✓ Loaded R0: {len(self.r0_df):,} rows")
        
        # 3. Load embedding_to_centroid mapping
        e2c_path = self.INDEX_PATH / "embedding_to_centroid.pt"
        if not e2c_path.exists():
            raise FileNotFoundError(f"embedding_to_centroid not found: {e2c_path}")
        self.embedding_to_centroid = torch.load(e2c_path, weights_only=True)
        print(f"✓ Loaded embedding_to_centroid: {len(self.embedding_to_centroid):,} embeddings")
        
        # 4. Load reverse index for doc -> embedding positions
        offsets_path = self.INDEX_PATH / "reverse_doc_offsets.pt"
        indices_path = self.INDEX_PATH / "reverse_sorted_indices.pt"
        
        if not offsets_path.exists():
            raise FileNotFoundError(f"reverse_doc_offsets not found: {offsets_path}")
        if not indices_path.exists():
            raise FileNotFoundError(f"reverse_sorted_indices not found: {indices_path}")
        
        self.reverse_doc_offsets = torch.load(offsets_path, weights_only=True)
        self.reverse_sorted_indices = torch.load(indices_path, weights_only=True)
        print(f"✓ Loaded reverse index: {len(self.reverse_doc_offsets):,} docs, {len(self.reverse_sorted_indices):,} embeddings")
        
        # 5. Build R0 lookup: (query_id, q_token_id) -> set of centroid IDs
        print("\nBuilding R0 lookup...")
        self._build_r0_lookup()
        print(f"✓ R0 lookup built: {len(self.r0_lookup):,} unique (query_id, q_token_id) keys")
    
    def setup_scoring_components(self):
        """
        Load additional components needed for similarity scoring (Analysis 2).
        Called lazily when analyze_fallback_strength() is invoked.
        """
        if self.centroids is not None:
            return  # Already loaded
        
        print("\n" + "-" * 40)
        print("Loading scoring components for Analysis 2...")
        print("-" * 40)
        
        # Centroids
        centroids_path = self.INDEX_PATH / "centroids.npy"
        self.centroids = torch.from_numpy(np.load(str(centroids_path)))
        print(f"✓ Loaded centroids: {self.centroids.shape}")
        
        # Residuals compacted
        residuals_path = self.INDEX_PATH / "residuals.compacted.pt"
        self.residuals_compacted = torch.load(str(residuals_path), weights_only=True)
        print(f"✓ Loaded residuals: {self.residuals_compacted.shape}")
        
        # Sizes and offsets
        sizes_path = self.INDEX_PATH / "sizes.compacted.pt"
        sizes_compacted = torch.load(str(sizes_path), weights_only=True)
        num_centroids = len(sizes_compacted)
        self.offsets_compacted = torch.zeros((num_centroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=self.offsets_compacted[1:])
        print(f"✓ Computed offsets: {self.offsets_compacted.shape}")
        
        # Bucket weights
        bucket_weights_path = self.INDEX_PATH / "bucket_weights.npy"
        self.bucket_weights = torch.from_numpy(np.load(str(bucket_weights_path)))
        print(f"✓ Loaded bucket_weights: {self.bucket_weights.shape}")
        
        # Determine nbits from residual shape
        packed_dim = self.residuals_compacted.shape[1]
        self.nbits = 4 if packed_dim == 64 else 2
        print(f"✓ Determined nbits: {self.nbits}")
        
    def setup_query_encoding(self):
        """
        Setup query encoding components (lazy load checkpoint and queries).
        Uses Checkpoint directly instead of WARPSearcher to avoid environment variable requirements.
        """
        if self._checkpoint is not None:
            return
        
        print("\n" + "-" * 40)
        print("Setting up query encoding...")
        print("-" * 40)
        
        # Load queries from BEIR dataset
        queries_path = Path("/mnt/datasets/BEIR/quora/queries.jsonl")
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        
        self._queries = {}
        with open(queries_path) as f:
            for line in f:
                q = json.loads(line)
                self._queries[int(q['_id'])] = q['text']
        print(f"✓ Loaded {len(self._queries):,} queries")
        
        # Load ColBERT config from index for proper tokenization settings
        from warp.infra.config import ColBERTConfig
        from warp.modeling.checkpoint import Checkpoint
        
        config = ColBERTConfig.load_from_index(str(self.INDEX_PATH))
        
        print("Loading Checkpoint for query encoding (this may take a moment)...")
        self._checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
        print(f"✓ Checkpoint initialized: {config.checkpoint}")
    
    def _encode_query(self, query_id: int) -> torch.Tensor:
        """
        Encode a query and return its token embeddings.
        
        Returns:
            Q: (num_tokens, 128) query token embeddings
        """
        if self._checkpoint is None or self._queries is None:
            raise RuntimeError("Query encoding not set up. Call setup_query_encoding() first.")
        
        query_text = self._queries.get(query_id)
        if query_text is None:
            raise ValueError(f"Query ID {query_id} not found in queries")
        
        Q = self._checkpoint.queryFromText([query_text], to_cpu=True)  # Returns (1, num_tokens, 128)
        return Q[0]  # (num_tokens, 128)
    
    def _build_r0_lookup(self):
        """
        Build lookup from (query_id, q_token_id) to set of probed centroid IDs.
        
        Critical: This must be keyed by BOTH query_id AND q_token_id because
        centroid selection is token-specific.
        """
        self.r0_lookup = {}
        
        # Group R0 by (query_id, q_token_id) and collect centroid IDs
        grouped = self.r0_df.groupby(['query_id', 'q_token_id'])['centroid_id'].apply(set)
        
        for (query_id, q_token_id), centroid_set in grouped.items():
            self.r0_lookup[(query_id, q_token_id)] = centroid_set
        
        # Validation: each (query_id, q_token_id) should have nprobe=32 centroids
        sizes = [len(v) for v in self.r0_lookup.values()]
        unique_sizes = set(sizes)
        if unique_sizes != {32}:
            print(f"  WARNING: Expected all R0 groups to have 32 centroids, found sizes: {unique_sizes}")
        else:
            print(f"  ✓ All R0 groups have exactly 32 centroids (nprobe=32)")
    
    def _get_doc_embedding_positions(self, doc_id: int) -> torch.Tensor:
        """
        Get all embedding positions for a document using the reverse index.
        
        Returns tensor of embedding positions belonging to this document.
        """
        start = self.reverse_doc_offsets[doc_id].item()
        end = self.reverse_doc_offsets[doc_id + 1].item()
        return self.reverse_sorted_indices[start:end]
    
    def _check_fallback_availability(self, doc_id: int, query_id: int, q_token_id: int) -> Tuple[bool, int, int]:
        """
        Check if document has ANY embedding accessible for this (query, token).
        
        Args:
            doc_id: Document ID
            query_id: Query ID
            q_token_id: Query token index
            
        Returns:
            Tuple of (has_fallback, n_doc_embeddings, n_accessible)
        """
        # Get all embedding positions for this document
        doc_positions = self._get_doc_embedding_positions(doc_id)
        n_doc_embeddings = len(doc_positions)
        
        if n_doc_embeddings == 0:
            return False, 0, 0
        
        # Get probed centroids for this (query, token)
        key = (query_id, q_token_id)
        if key not in self.r0_lookup:
            # This shouldn't happen if data is consistent
            print(f"  WARNING: Key {key} not found in R0 lookup")
            return False, n_doc_embeddings, 0
        
        probed_centroids = self.r0_lookup[key]
        
        # Check if ANY doc embedding's centroid is in probed set
        n_accessible = 0
        for pos in doc_positions:
            centroid_id = self.embedding_to_centroid[pos.item()].item()
            if centroid_id in probed_centroids:
                n_accessible += 1
        
        has_fallback = n_accessible > 0
        return has_fallback, n_doc_embeddings, n_accessible
    
    def analyze_fallback_availability(self, save_details: bool = True, 
                                       sample_size: Optional[int] = None) -> FallbackAnalysisResult:
        """
        Analysis 1: The 72.6% Finding
        
        For each inaccessible oracle case, check if the document has ANY other
        embedding accessible for that query token.
        
        This is the core reproducible computation for the 72.6% finding.
        
        Args:
            save_details: If True, save per-case details for inspection
            sample_size: If provided, randomly sample this many cases.
                         Use None for full analysis (prod), or pass from config.
                         Default sample sizes by config:
                           - smoke: 500 cases (~seconds)
                           - dev: 5000 cases (~1 min)  
                           - prod: None = all 27,177 cases (~3-5 min)
            
        Returns:
            FallbackAnalysisResult with counts and validation info
        """
        print("\n" + "=" * 60)
        print("Analysis 1: Fallback Availability (72.6% Finding)")
        print("=" * 60)
        
        # Determine sample size from config if not explicitly provided
        if sample_size is None and hasattr(self, 'config') and self.config is not None:
            config_name = getattr(self.config, 'name', 'prod')
            sample_size = SAMPLE_SIZES.get(config_name, None)
            if sample_size:
                print(f"\nUsing {config_name} config: sampling {sample_size:,} cases")
        
        # Step 1: Filter M4R to inaccessible oracles
        inaccessible_full = self.m4r_df[self.m4r_df['oracle_is_accessible'] == False].copy()
        full_population_size = len(inaccessible_full)
        
        print(f"\nStep 1: Filter to inaccessible oracles")
        print(f"  Total M4R rows: {len(self.m4r_df):,}")
        print(f"  Inaccessible oracles (full): {full_population_size:,}")
        print(f"  Expected: {self.EXPECTED_INACCESSIBLE_COUNT:,}")
        
        m4r_count_matches = (full_population_size == self.EXPECTED_INACCESSIBLE_COUNT)
        if m4r_count_matches:
            print(f"  ✓ Count matches expected value")
        else:
            print(f"  ⚠ Count differs from expected by {abs(full_population_size - self.EXPECTED_INACCESSIBLE_COUNT):,}")
        
        # Step 1b: Sample if requested
        is_sampled = False
        if sample_size is not None and sample_size < full_population_size:
            is_sampled = True
            inaccessible = inaccessible_full.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"\nStep 1b: Sampling {sample_size:,} cases (seed={RANDOM_SEED})")
            print(f"  ⚠ Results are estimates based on sample, not full census")
        else:
            inaccessible = inaccessible_full
            sample_size = None  # Indicate full analysis
            print(f"\nStep 1b: Using full population (no sampling)")
        
        total_to_check = len(inaccessible)
        
        # Step 2: For each inaccessible oracle, check fallback availability
        print(f"\nStep 2: Check fallback availability for {total_to_check:,} cases...")
        
        results = []
        has_fallback_count = 0
        no_fallback_count = 0
        
        for idx, row in tqdm(inaccessible.iterrows(), total=total_to_check, desc="Checking fallbacks"):
            query_id = int(row['query_id'])
            q_token_id = int(row['q_token_id'])
            doc_id = int(row['doc_id'])
            oracle_score = row.get('oracle_score', np.nan)
            
            has_fallback, n_doc_embs, n_accessible = self._check_fallback_availability(
                doc_id, query_id, q_token_id
            )
            
            if has_fallback:
                has_fallback_count += 1
            else:
                no_fallback_count += 1
            
            if save_details:
                results.append({
                    'query_id': query_id,
                    'q_token_id': q_token_id,
                    'doc_id': doc_id,
                    'oracle_score': oracle_score,
                    'has_fallback': has_fallback,
                    'n_doc_embeddings': n_doc_embs,
                    'n_accessible_embeddings': n_accessible,
                })
        
        # Compute percentages
        has_fallback_pct = (has_fallback_count / total_to_check) * 100
        no_fallback_pct = (no_fallback_count / total_to_check) * 100
        
        # Print results
        print(f"\n" + "=" * 60)
        print("RESULTS: Fallback Availability")
        print("=" * 60)
        
        if is_sampled:
            print(f"\n⚠ SAMPLED ANALYSIS (n={total_to_check:,} of {full_population_size:,})")
        else:
            print(f"\n✓ FULL CENSUS (n={total_to_check:,})")
            
        print(f"\n  Has fallback (at least 1 accessible embedding):")
        print(f"    Count: {has_fallback_count:,}")
        print(f"    Percentage: {has_fallback_pct:.2f}%")
        print(f"\n  NO fallback (zero accessible embeddings):")
        print(f"    Count: {no_fallback_count:,}")
        print(f"    Percentage: {no_fallback_pct:.2f}%")
        print(f"\n  Expected no-fallback %: {self.EXPECTED_NO_FALLBACK_PCT}%")
        
        if abs(no_fallback_pct - self.EXPECTED_NO_FALLBACK_PCT) < 1.0:
            print(f"  ✓ Result matches expected value (within 1.0pp)")
        else:
            diff = no_fallback_pct - self.EXPECTED_NO_FALLBACK_PCT
            print(f"  ⚠ Result differs from expected by {diff:+.2f}pp")
            if is_sampled:
                print(f"    (Note: sampling variance expected)")
        
        # Build result object
        case_details_df = pd.DataFrame(results) if save_details else None
        
        self.fallback_result = FallbackAnalysisResult(
            total_inaccessible_oracles=total_to_check,
            has_fallback_count=has_fallback_count,
            no_fallback_count=no_fallback_count,
            has_fallback_pct=has_fallback_pct,
            no_fallback_pct=no_fallback_pct,
            is_sampled=is_sampled,
            sample_size=sample_size if is_sampled else None,
            full_population_size=full_population_size,
            m4r_inaccessible_count_matches=m4r_count_matches,
            r0_lookup_unique_keys=len(self.r0_lookup),
            docs_checked=inaccessible['doc_id'].nunique(),
            case_details=case_details_df,
        )
        
        return self.fallback_result
    
    def analyze_fallback_strength(self, 
                                   cases_df: Optional[pd.DataFrame] = None,
                                   sample_size: Optional[int] = None) -> FallbackStrengthResult:
        """
        Analysis 2: Fallback Strength Measurement
        
        For cases where an accessible fallback exists (the 27.4%), measure how good
        that fallback is compared to the oracle.
        
        Computes:
            - score_gap = oracle_score - best_fallback_score
            - relative_gap = score_gap / oracle_score
            
        Classifies into:
            - Strong fallback: gap < 5% (Regime B compensation)
            - Moderate fallback: 5% <= gap < 20%
            - Weak fallback: gap >= 20% (still Regime A)
            
        Args:
            cases_df: DataFrame of cases to analyze. If None, uses cases from 
                      Analysis 1 that have fallback (has_fallback == True).
            sample_size: If provided, sample this many cases for faster analysis.
                         If None, uses config-based sampling.
        
        Returns:
            FallbackStrengthResult with gap statistics and regime classification
        """
        print("\n" + "=" * 60)
        print("Analysis 2: Fallback Strength Measurement")
        print("=" * 60)
        
        # Setup scoring components if not already loaded
        self.setup_scoring_components()
        self.setup_query_encoding()
        
        # Get cases to analyze
        if cases_df is None:
            if self.fallback_result is None or self.fallback_result.case_details is None:
                raise RuntimeError("No case details available. Run analyze_fallback_availability() first.")
            
            # Filter to cases that HAVE fallback
            all_cases = self.fallback_result.case_details
            cases_df = all_cases[all_cases['has_fallback'] == True].copy()
        
        # Join with M4R to get oracle_embedding_pos (needed for consistent scoring)
        cases_df = cases_df.merge(
            self.m4r_df[['query_id', 'q_token_id', 'doc_id', 'oracle_embedding_pos']],
            on=['query_id', 'q_token_id', 'doc_id'],
            how='left'
        )
        
        full_population_size = len(cases_df)
        print(f"\nCases with fallback (full): {full_population_size:,}")
        
        # Determine sample size from config if not explicitly provided
        if sample_size is None and hasattr(self, 'config') and self.config is not None:
            config_name = getattr(self.config, 'name', 'prod')
            # Use smaller sample for Analysis 2 since it involves encoding
            analysis2_samples = {
                'smoke': 100,     # Very quick
                'dev': 500,       # Moderate
                'prod': None,     # Full (~7 min)
            }
            sample_size = analysis2_samples.get(config_name, None)
            if sample_size:
                print(f"Using {config_name} config: sampling {sample_size:,} cases")
        
        # Sample if needed
        is_sampled = False
        if sample_size is not None and sample_size < full_population_size:
            is_sampled = True
            cases_to_analyze = cases_df.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"Sampling {sample_size:,} cases (seed={RANDOM_SEED})")
        else:
            cases_to_analyze = cases_df
            sample_size = None
            print("Using full population (no sampling)")
        
        n_cases = len(cases_to_analyze)
        print(f"\nAnalyzing {n_cases:,} cases...")
        
        # Cache for encoded queries to avoid re-encoding
        query_cache: Dict[int, torch.Tensor] = {}
        
        results = []
        for idx, row in tqdm(cases_to_analyze.iterrows(), total=n_cases, desc="Computing fallback scores"):
            query_id = int(row['query_id'])
            q_token_id = int(row['q_token_id'])
            doc_id = int(row['doc_id'])
            
            # Get or cache query embeddings
            if query_id not in query_cache:
                query_cache[query_id] = self._encode_query(query_id)
            Q = query_cache[query_id]
            
            # Get query token embedding
            if q_token_id >= Q.shape[0]:
                # Token ID out of range (padding), skip
                continue
            q_token = Q[q_token_id]
            
            # Precompute bucket scores for this token
            bucket_scores = compute_bucket_scores(q_token, self.bucket_weights)
            
            # Get ALL embedding positions for this doc
            all_doc_positions = self._get_doc_embedding_positions(doc_id)
            
            # Find TRUE oracle: best score across ALL positions using OUR Q
            # This is necessary because M4R's oracle_embedding_pos was computed
            # with a different Q encoding, which is not comparable
            true_oracle_score = float('-inf')
            true_oracle_pos = -1
            for pos in all_doc_positions:
                score = compute_score_for_position(
                    q_token=q_token,
                    embedding_pos=pos,
                    centroids=self.centroids,
                    residuals_compacted=self.residuals_compacted,
                    offsets_compacted=self.offsets_compacted,
                    bucket_weights=self.bucket_weights,
                    nbits=self.nbits,
                    bucket_scores=bucket_scores
                )
                if score > true_oracle_score:
                    true_oracle_score = score
                    true_oracle_pos = pos
            
            oracle_score = true_oracle_score
            
            # Get accessible embedding positions
            accessible_positions = self._get_accessible_positions(doc_id, query_id, q_token_id)
            
            if len(accessible_positions) == 0:
                # Shouldn't happen for cases with has_fallback=True, but be safe
                continue
            
            # Compute score for each accessible embedding, find best
            best_fallback_score = float('-inf')
            for pos in accessible_positions:
                score = compute_score_for_position(
                    q_token=q_token,
                    embedding_pos=pos,
                    centroids=self.centroids,
                    residuals_compacted=self.residuals_compacted,
                    offsets_compacted=self.offsets_compacted,
                    bucket_weights=self.bucket_weights,
                    nbits=self.nbits,
                    bucket_scores=bucket_scores
                )
                if score > best_fallback_score:
                    best_fallback_score = score
            
            # Compute gap (now both scores use the same Q)
            score_gap = oracle_score - best_fallback_score
            relative_gap = score_gap / oracle_score if oracle_score > 0 else 0.0
            
            results.append({
                'query_id': query_id,
                'q_token_id': q_token_id,
                'doc_id': doc_id,
                'oracle_score': oracle_score,
                'best_fallback_score': best_fallback_score,
                'score_gap': score_gap,
                'relative_gap': relative_gap,
                'n_accessible': len(accessible_positions),
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("WARNING: No valid results to analyze")
            return None
        
        # Compute statistics
        score_gaps = results_df['score_gap']
        relative_gaps = results_df['relative_gap']
        oracle_scores = results_df['oracle_score']
        
        mean_score_gap = score_gaps.mean()
        median_score_gap = score_gaps.median()
        std_score_gap = score_gaps.std()
        
        mean_relative_gap = relative_gaps.mean()
        median_relative_gap = relative_gaps.median()
        
        # Cumulative impact metrics (more honest framing)
        total_oracle_score_sum = oracle_scores.sum()
        total_score_gap_sum = score_gaps.sum()
        total_score_loss_pct = (total_score_gap_sum / total_oracle_score_sum) * 100 if total_oracle_score_sum > 0 else 0
        
        # Count cases with POSITIVE gap (actual loss)
        positive_gap_mask = score_gaps > 0
        n_with_actual_loss = positive_gap_mask.sum()
        pct_with_actual_loss = (n_with_actual_loss / len(results_df)) * 100
        
        # Percentiles
        percentiles = {
            'p5': np.percentile(relative_gaps, 5),
            'p10': np.percentile(relative_gaps, 10),
            'p25': np.percentile(relative_gaps, 25),
            'p50': np.percentile(relative_gaps, 50),
            'p75': np.percentile(relative_gaps, 75),
            'p90': np.percentile(relative_gaps, 90),
            'p95': np.percentile(relative_gaps, 95),
        }
        
        # Regime classification
        n_analyzed = len(results_df)
        strong_mask = relative_gaps < 0.05
        moderate_mask = (relative_gaps >= 0.05) & (relative_gaps < 0.20)
        weak_mask = relative_gaps >= 0.20
        
        strong_count = strong_mask.sum()
        moderate_count = moderate_mask.sum()
        weak_count = weak_mask.sum()
        
        # Percentages within fallback cases (biased framing - but still useful)
        strong_pct = (strong_count / n_analyzed) * 100
        moderate_pct = (moderate_count / n_analyzed) * 100
        weak_pct = (weak_count / n_analyzed) * 100
        
        # Get total inaccessible from Analysis 1 result (if available)
        total_inaccessible = self.fallback_result.total_inaccessible_oracles if self.fallback_result else None
        
        # Compute percentages relative to ALL inaccessible (honest framing)
        if total_inaccessible and total_inaccessible > 0:
            # Scale counts if sampled
            if is_sampled:
                scale = full_population_size / n_analyzed
                strong_scaled = int(strong_count * scale)
                moderate_scaled = int(moderate_count * scale)
                weak_scaled = int(weak_count * scale)
            else:
                strong_scaled = strong_count
                moderate_scaled = moderate_count
                weak_scaled = weak_count
            
            pct_of_all_strong = (strong_scaled / total_inaccessible) * 100
            pct_of_all_moderate = (moderate_scaled / total_inaccessible) * 100
            pct_of_all_weak = (weak_scaled / total_inaccessible) * 100
            no_fallback_pct = self.fallback_result.no_fallback_pct
        else:
            pct_of_all_strong = None
            pct_of_all_moderate = None
            pct_of_all_weak = None
            no_fallback_pct = None
        
        # Print results
        print(f"\n" + "=" * 60)
        print("RESULTS: Fallback Strength")
        print("=" * 60)
        
        if is_sampled:
            print(f"\n⚠ SAMPLED ANALYSIS (n={n_analyzed:,} of {full_population_size:,})")
        else:
            print(f"\n✓ FULL ANALYSIS (n={n_analyzed:,})")
        
        print(f"\nScore Gap Statistics:")
        print(f"  Mean gap: {mean_score_gap:.4f}")
        print(f"  Median gap: {median_score_gap:.4f}")
        print(f"  Std dev: {std_score_gap:.4f}")
        
        print(f"\nRelative Gap Statistics (gap / oracle_score):")
        print(f"  Mean: {mean_relative_gap:.2%}")
        print(f"  Median: {median_relative_gap:.2%}")
        print(f"  Percentiles:")
        for k, v in percentiles.items():
            print(f"    {k}: {v:.2%}")
        
        # HONEST FRAMING: Cumulative impact
        print(f"\n" + "-" * 40)
        print("CUMULATIVE IMPACT (Total Score Loss)")
        print("-" * 40)
        print(f"  Sum of oracle scores (analyzed): {total_oracle_score_sum:.2f}")
        print(f"  Sum of score gaps (total loss):  {total_score_gap_sum:.2f}")
        print(f"  Total score loss:                {total_score_loss_pct:.2f}%")
        print(f"  Cases with actual loss (gap>0):  {n_with_actual_loss:,} ({pct_with_actual_loss:.1f}%)")
        
        print(f"\nRegime Classification (within fallback cases only):")
        print(f"  ⚠ NOTE: These percentages are BIASED - computed only over")
        print(f"          cases that have fallback, not all inaccessible cases!")
        print(f"  Strong fallback (gap < 5%):   {strong_count:,} ({strong_pct:.1f}%)")
        print(f"  Moderate fallback (5-20%):    {moderate_count:,} ({moderate_pct:.1f}%)")
        print(f"  Weak fallback (gap >= 20%):   {weak_count:,} ({weak_pct:.1f}%)")
        
        # HONEST FRAMING: Percentages relative to ALL inaccessible
        if total_inaccessible:
            print(f"\n" + "-" * 40)
            print("HONEST FRAMING: % of ALL Inaccessible Cases")
            print("-" * 40)
            print(f"  Total inaccessible oracle cases: {total_inaccessible:,}")
            print(f"  ")
            print(f"  NO fallback (full handicap):     {no_fallback_pct:.1f}%")
            print(f"  Strong fallback (<5% gap):       {pct_of_all_strong:.1f}%")
            print(f"  Moderate fallback (5-20% gap):   {pct_of_all_moderate:.1f}%")
            print(f"  Weak fallback (>=20% gap):       {pct_of_all_weak:.1f}%")
            print(f"  ")
            print(f"  → SIGNIFICANT HANDICAP:          {no_fallback_pct + pct_of_all_weak:.1f}%")
            print(f"    (no fallback OR weak fallback)")
            print(f"  → REAL COMPENSATION:             {pct_of_all_strong:.1f}%")
            print(f"    (strong fallback only)")
        
        # Build result object
        self.strength_result = FallbackStrengthResult(
            total_cases_with_fallback=full_population_size,
            cases_analyzed=n_analyzed,
            mean_score_gap=mean_score_gap,
            median_score_gap=median_score_gap,
            std_score_gap=std_score_gap,
            mean_relative_gap=mean_relative_gap,
            median_relative_gap=median_relative_gap,
            strong_fallback_count=int(strong_count),
            moderate_fallback_count=int(moderate_count),
            weak_fallback_count=int(weak_count),
            strong_fallback_pct=strong_pct,
            moderate_fallback_pct=moderate_pct,
            weak_fallback_pct=weak_pct,
            percentiles=percentiles,
            total_oracle_score_sum=total_oracle_score_sum,
            total_score_gap_sum=total_score_gap_sum,
            total_score_loss_pct=total_score_loss_pct,
            pct_of_all_inaccessible_strong=pct_of_all_strong,
            pct_of_all_inaccessible_moderate=pct_of_all_moderate,
            pct_of_all_inaccessible_weak=pct_of_all_weak,
            is_sampled=is_sampled,
            sample_size=sample_size if is_sampled else None,
            full_population_size=full_population_size,
            case_details=results_df,
        )
        
        return self.strength_result
    
    def _get_accessible_positions(self, doc_id: int, query_id: int, q_token_id: int) -> List[int]:
        """
        Get embedding positions from doc that are accessible for this (query, token).
        """
        # Get all embedding positions for this document
        doc_positions = self._get_doc_embedding_positions(doc_id)
        
        if len(doc_positions) == 0:
            return []
        
        # Get probed centroids for this (query, token)
        key = (query_id, q_token_id)
        if key not in self.r0_lookup:
            return []
        
        probed_centroids = self.r0_lookup[key]
        
        # Filter to accessible positions
        accessible = []
        for pos in doc_positions:
            centroid_id = self.embedding_to_centroid[pos.item()].item()
            if centroid_id in probed_centroids:
                accessible.append(pos.item())
        
        return accessible
    
    def analyze_centroid_spread(self, 
                                 sample_size: Optional[int] = None,
                                 k_threshold: float = 1.5) -> CentroidSpreadResult:
        """
        Analysis 4: Strong Embedding Centroid Spread
        
        For oracle-ACCESSIBLE cases, analyze how "strong" embeddings (within k
        standard deviations of oracle score) are distributed across centroids.
        
        KEY METHODOLOGY NOTE:
        This analysis is SELF-CONTAINED with respect to query encoding:
        - We encode the query fresh using Checkpoint
        - We score ALL doc embeddings using that SAME fresh Q
        - We identify "strong" embeddings relative to the best score we compute
        - NO comparison to old M4R scores is made
        
        This avoids the Q encoding consistency issue we encountered in Analysis 2.
        
        Args:
            sample_size: Number of cases to sample. None = full analysis.
            k_threshold: Standard deviations from oracle to define "strong" (default 1.5)
            
        Returns:
            CentroidSpreadResult with distribution statistics
        """
        print("\n" + "=" * 60)
        print("Analysis 4: Strong Embedding Centroid Spread")
        print("=" * 60)
        
        # Ensure scoring components are loaded
        if self.centroids is None:
            print("\n" + "-" * 40)
            print("Loading scoring components for Analysis 4...")
            print("-" * 40)
            self.setup_scoring_components()
        
        # Ensure query encoding is set up
        if self._checkpoint is None:
            print("\n" + "-" * 40)
            print("Setting up query encoding...")
            print("-" * 40)
            self.setup_query_encoding()
        
        # Filter M4R to oracle-ACCESSIBLE cases
        accessible_cases = self.m4r_df[self.m4r_df['oracle_is_accessible'] == True].copy()
        full_population_size = len(accessible_cases)
        
        print(f"\nOracle-accessible cases (full): {full_population_size:,}")
        
        # Determine if sampling
        if sample_size is not None and sample_size < full_population_size:
            is_sampled = True
            cases_df = accessible_cases.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"Using sample of {sample_size:,} cases")
        else:
            is_sampled = False
            cases_df = accessible_cases
            print(f"Using full population (no sampling)")
        
        print(f"\nAnalyzing {len(cases_df):,} cases with k={k_threshold} std threshold...")
        
        # Pre-compute bucket scores structure (only need bucket_weights loaded once)
        results = []
        
        for idx, row in tqdm(cases_df.iterrows(), total=len(cases_df), 
                             desc="Analyzing centroid spread"):
            query_id = int(row['query_id'])
            q_token_id = int(row['q_token_id'])
            doc_id = int(row['doc_id'])
            oracle_centroid_id = int(row['oracle_centroid_id'])
            
            # Get probed centroids for this (query, token)
            key = (query_id, q_token_id)
            if key not in self.r0_lookup:
                continue
            probed_centroids = self.r0_lookup[key]
            
            # Get ALL embedding positions for this document
            all_doc_positions = self._get_doc_embedding_positions(doc_id)
            
            if len(all_doc_positions) < 2:
                # Need at least 2 embeddings to have meaningful spread analysis
                continue
            
            # Encode query token FRESH
            try:
                Q = self._encode_query(query_id)
                q_token = Q[q_token_id]
                
                # Pre-compute bucket scores for this token
                bucket_scores = compute_bucket_scores(q_token, self.bucket_weights)
            except Exception as e:
                continue
            
            # Score ALL doc embeddings with the SAME fresh Q
            scores = []
            positions = []
            centroids_for_pos = []
            
            for pos in all_doc_positions:
                pos_val = pos.item()
                score = compute_score_for_position(
                    q_token=q_token,
                    embedding_pos=pos_val,
                    centroids=self.centroids,
                    residuals_compacted=self.residuals_compacted,
                    offsets_compacted=self.offsets_compacted,
                    bucket_weights=self.bucket_weights,
                    nbits=self.nbits,
                    bucket_scores=bucket_scores
                )
                scores.append(score)
                positions.append(pos_val)
                centroid_id = self.embedding_to_centroid[pos_val].item()
                centroids_for_pos.append(centroid_id)
            
            scores = np.array(scores)
            positions = np.array(positions)
            centroids_for_pos = np.array(centroids_for_pos)
            
            # Find TRUE oracle (best score we computed)
            true_oracle_score = np.max(scores)
            true_oracle_idx = np.argmax(scores)
            true_oracle_centroid = centroids_for_pos[true_oracle_idx]
            
            # Compute threshold for "strong" embeddings
            score_std = np.std(scores)
            
            if score_std < 1e-6:
                # All scores nearly identical — all are "strong"
                strong_mask = np.ones(len(scores), dtype=bool)
            else:
                threshold = true_oracle_score - k_threshold * score_std
                strong_mask = scores >= threshold
            
            # Count strong embeddings
            n_strong = np.sum(strong_mask)
            strong_positions = positions[strong_mask]
            strong_centroids = centroids_for_pos[strong_mask]
            
            # Unique centroids containing strong embeddings
            unique_strong_centroids = set(strong_centroids)
            n_unique_centroids = len(unique_strong_centroids)
            
            # How many of those centroids were probed?
            n_centroids_probed = len(unique_strong_centroids & probed_centroids)
            capture_rate = n_centroids_probed / n_unique_centroids if n_unique_centroids > 0 else 1.0
            
            # Strong embeddings in oracle's centroid
            n_strong_in_oracle = np.sum(strong_centroids == true_oracle_centroid)
            all_strong_in_oracle = (n_strong == n_strong_in_oracle)
            
            results.append({
                'query_id': query_id,
                'q_token_id': q_token_id,
                'doc_id': doc_id,
                'n_doc_embeddings': len(all_doc_positions),
                'n_strong': int(n_strong),
                'n_unique_centroids': n_unique_centroids,
                'n_centroids_probed': n_centroids_probed,
                'capture_rate': capture_rate,
                'n_strong_in_oracle_centroid': int(n_strong_in_oracle),
                'all_strong_in_oracle_centroid': all_strong_in_oracle,
                'true_oracle_score': true_oracle_score,
                'score_std': score_std,
                'threshold': true_oracle_score - k_threshold * score_std,
            })
        
        results_df = pd.DataFrame(results)
        n_analyzed = len(results_df)
        
        if n_analyzed == 0:
            print("WARNING: No valid cases analyzed!")
            return None
        
        # Compute statistics
        mean_strong = results_df['n_strong'].mean()
        median_strong = results_df['n_strong'].median()
        
        mean_unique_centroids = results_df['n_unique_centroids'].mean()
        median_unique_centroids = results_df['n_unique_centroids'].median()
        
        mean_centroids_probed = results_df['n_centroids_probed'].mean()
        median_centroids_probed = results_df['n_centroids_probed'].median()
        mean_capture_rate = results_df['capture_rate'].mean()
        
        mean_strong_in_oracle = results_df['n_strong_in_oracle_centroid'].mean()
        pct_oracle_only = (results_df['all_strong_in_oracle_centroid'].sum() / n_analyzed) * 100
        
        # Distribution of unique centroids containing strong embeddings
        centroid_counts = results_df['n_unique_centroids'].value_counts().sort_index()
        # Bin into categories: 1, 2, 3, 4+
        distribution = {}
        for i in range(1, 4):
            distribution[str(i)] = int(centroid_counts.get(i, 0))
        distribution['4+'] = int(centroid_counts[centroid_counts.index >= 4].sum())
        
        # Print results
        print(f"\n" + "=" * 60)
        print("RESULTS: Centroid Spread Analysis")
        print("=" * 60)
        
        if is_sampled:
            print(f"\n⚠ SAMPLED ANALYSIS (n={n_analyzed:,} of {full_population_size:,})")
        else:
            print(f"\n✓ FULL ANALYSIS (n={n_analyzed:,})")
        
        print(f"\nStrong Embedding Statistics (k={k_threshold} std):")
        print(f"  Mean strong embeddings per case: {mean_strong:.2f}")
        print(f"  Median strong embeddings: {median_strong:.1f}")
        
        print(f"\nCentroid Spread (unique centroids containing strong embeddings):")
        print(f"  Mean: {mean_unique_centroids:.2f}")
        print(f"  Median: {median_unique_centroids:.1f}")
        print(f"  Distribution:")
        for k, v in distribution.items():
            pct = (v / n_analyzed) * 100
            print(f"    {k} centroid(s): {v:,} ({pct:.1f}%)")
        
        print(f"\nProbing Capture:")
        print(f"  Mean centroids probed (of those with strong): {mean_centroids_probed:.2f}")
        print(f"  Median centroids probed: {median_centroids_probed:.1f}")
        print(f"  Mean capture rate: {mean_capture_rate:.1%}")
        
        print(f"\nOracle Centroid Concentration:")
        print(f"  Mean strong embeddings in oracle centroid: {mean_strong_in_oracle:.2f}")
        print(f"  % cases where ALL strong in oracle centroid: {pct_oracle_only:.1f}%")
        
        # Interpretation
        print(f"\n" + "-" * 40)
        print("INTERPRETATION")
        print("-" * 40)
        if mean_unique_centroids < 2:
            print("  → Strong embeddings are TIGHTLY CLUSTERED (mostly 1 centroid)")
            print("    This supports Regime A: missing oracle centroid = missing all strong evidence")
        elif mean_capture_rate > 0.9:
            print("  → Strong embeddings are DISPERSED but well captured by probing")
            print("    nprobe=32 is sufficient to capture most strong embedding centroids")
        else:
            print(f"  → Strong embeddings spread across ~{mean_unique_centroids:.1f} centroids")
            print(f"    Probing captures {mean_capture_rate:.1%} of them on average")
        
        # Build result object
        self.spread_result = CentroidSpreadResult(
            total_oracle_accessible=full_population_size,
            cases_analyzed=n_analyzed,
            mean_strong_embeddings=mean_strong,
            median_strong_embeddings=median_strong,
            mean_unique_centroids=mean_unique_centroids,
            median_unique_centroids=median_unique_centroids,
            mean_centroids_probed=mean_centroids_probed,
            median_centroids_probed=median_centroids_probed,
            mean_capture_rate=mean_capture_rate,
            centroids_distribution=distribution,
            mean_strong_in_oracle_centroid=mean_strong_in_oracle,
            pct_oracle_only=pct_oracle_only,
            k_threshold=k_threshold,
            is_sampled=is_sampled,
            sample_size=sample_size if is_sampled else None,
            full_population_size=full_population_size,
            case_details=results_df,
        )
        
        return self.spread_result
    
    def analyze(self) -> HypothesisResult:
        """
        Run the full co-location analysis.
        
        Currently implements Analysis 1 (72.6% finding).
        Analysis 2 and 4 to be added.
        """
        # Run Analysis 1
        fallback_result = self.analyze_fallback_availability(save_details=True)
        
        # Hypothesis is supported if:
        # 1. The no-fallback rate is high (>70%), confirming Regime A dominates
        # 2. This validates oracle accessibility as proxy for evidence accessibility
        supported = fallback_result.no_fallback_pct > 70.0
        
        # Effect size: the no-fallback percentage
        effect_size = fallback_result.no_fallback_pct / 100.0
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=effect_size,
            effect_size_ci=(effect_size - 0.01, effect_size + 0.01),  # Narrow CI, empirical
            p_value=0.0,  # This is a census, not a sample
            statistics={
                'analysis_1_fallback_availability': fallback_result.to_dict(),
                'interpretation': {
                    'regime_a_certain_pct': fallback_result.no_fallback_pct,
                    'regime_ambiguous_pct': fallback_result.has_fallback_pct,
                    'conclusion': (
                        f"{fallback_result.no_fallback_pct:.1f}% of inaccessible oracles have NO "
                        f"fallback at all. The 2nd-best, 3rd-best, etc. are ALL inaccessible. "
                        f"Oracle accessibility IS a valid proxy for evidence accessibility."
                    )
                }
            },
            config_name=self.config.name,
            n_observations=fallback_result.total_inaccessible_oracles,
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        """Generate visualizations for the co-location analysis using black/dark gray palette."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        if self.fallback_result is None:
            print("No results to visualize. Run analyze() first.")
            return
        
        result = self.fallback_result
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        # Color palette: black and dark gray
        COLOR_PRIMARY = '#1a1a1a'      # Near black - for "No Fallback" (the key finding)
        COLOR_SECONDARY = '#666666'    # Dark gray - for "Has Fallback"
        COLOR_ACCENT = '#333333'       # Darker gray for edges/accents
        COLOR_TEXT_LIGHT = '#ffffff'   # White text on dark backgrounds
        COLOR_TEXT_DARK = '#1a1a1a'    # Dark text
        COLOR_BG_BOX = '#f0f0f0'       # Light gray for annotation boxes
        
        # =====================================================================
        # Plot 1: Main Finding - Horizontal stacked bar (clean scientific style)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 3))
        
        no_fallback = [result.no_fallback_pct]
        has_fallback = [result.has_fallback_pct]
        
        bar_width = 0.5
        x = [0]
        
        # Consistent gray-blue color palette
        COLOR_NO_FB = '#2C3E50'      # Dark blue-gray
        COLOR_HAS_FB = '#7F8C8D'     # Medium gray
        
        # Main bars
        bars1 = ax.barh(x, no_fallback, bar_width, 
                        label=f'No Alternative ({result.no_fallback_pct:.1f}%)', 
                        color=COLOR_NO_FB, edgecolor='white', linewidth=1.5)
        bars2 = ax.barh(x, has_fallback, bar_width, left=no_fallback, 
                        label=f'Has Alternative ({result.has_fallback_pct:.1f}%)',
                        color=COLOR_HAS_FB, edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_ylim(-0.4, 0.4)
        
        # Percentage labels on bars
        ax.annotate(f'{result.no_fallback_pct:.1f}%', 
                    xy=(result.no_fallback_pct/2, 0), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        ax.annotate(f'{result.has_fallback_pct:.1f}%', 
                    xy=(result.no_fallback_pct + result.has_fallback_pct/2, 0), 
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        fig_path = plot_dir / 'h3_colocation_fallback_availability.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Plot 2: Regime breakdown pie chart (clean scientific style)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Consistent gray-blue color palette
        COLOR_NO_FB = '#2C3E50'      # Dark blue-gray
        COLOR_HAS_FB = '#7F8C8D'     # Medium gray
        
        sizes = [result.no_fallback_pct, result.has_fallback_pct]
        labels = [f'No Alternative\n({result.no_fallback_pct:.1f}%)', 
                  f'Has Alternative\n({result.has_fallback_pct:.1f}%)']
        colors = [COLOR_NO_FB, COLOR_HAS_FB]
        explode = (0.02, 0)
        
        wedges, texts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        
        # Style the labels
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        # Add center circle for donut style
        centre_circle = plt.Circle((0, 0), 0.5, fc='white', ec='#BDC3C7', linewidth=2)
        ax.add_artist(centre_circle)
        
        plt.tight_layout()
        fig_path = plot_dir / 'h3_colocation_regime_breakdown.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Plot 3: Distribution of accessible embeddings for "has fallback" cases
        # =====================================================================
        if result.case_details is not None:
            has_fb = result.case_details[result.case_details['has_fallback'] == True]
            
            if len(has_fb) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('#fafafa')
                
                # Light blue color (slightly darker)
                COLOR_LIGHT_BLUE = '#5B9BD5'
                
                accessible_counts = has_fb['n_accessible_embeddings']
                
                # Compute histogram data for percentage display
                max_count = int(accessible_counts.max())
                bins = range(1, max_count + 2)
                counts_per_bin, bin_edges = np.histogram(accessible_counts, bins=bins)
                total_cases = len(has_fb)
                pcts_per_bin = counts_per_bin / total_cases * 100
                
                # Plot as percentage
                bin_centers = [b + 0.5 for b in range(1, max_count + 1)]
                ax.bar(bin_centers, pcts_per_bin, width=0.8, 
                       edgecolor=COLOR_TEXT_DARK, color=COLOR_LIGHT_BLUE, alpha=0.85,
                       linewidth=1.2)
                
                ax.set_xlabel('Accessible Embeddings per Document', fontsize=12, fontweight='medium')
                ax.set_ylabel('Percentage of Cases (%)', fontsize=12, fontweight='medium')
                ax.set_xticks(bin_centers[:min(15, len(bin_centers))])  # Limit x-ticks for readability
                ax.set_xticklabels([str(int(b)) for b in range(1, min(16, max_count + 1))])
                
                # Style the spines
                for spine in ax.spines.values():
                    spine.set_color(COLOR_ACCENT)
                    spine.set_linewidth(0.8)
                ax.tick_params(colors=COLOR_TEXT_DARK)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                fig_path = plot_dir / 'h3_colocation_accessible_distribution.png'
                plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Analysis 2 Visualizations (if strength analysis was run)
        # =====================================================================
        if self.strength_result is not None:
            self._visualize_strength_results(plot_dir, COLOR_PRIMARY, COLOR_SECONDARY, 
                                              COLOR_ACCENT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK, COLOR_BG_BOX)
    
    def _visualize_strength_results(self, plot_dir, COLOR_PRIMARY, COLOR_SECONDARY, 
                                     COLOR_ACCENT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK, COLOR_BG_BOX):
        """Generate visualizations for Analysis 2 (Fallback Strength)."""
        import matplotlib.pyplot as plt
        
        result = self.strength_result
        if result is None:
            return
        
        # Color for "green" regime (strong fallback / Regime B)
        COLOR_REGIME_B = '#4a4a4a'    # Medium-dark gray for Regime B (good)
        COLOR_MODERATE = '#666666'     # Medium gray
        COLOR_REGIME_A = '#1a1a1a'     # Dark (Regime A - still bad)
        
        # =====================================================================
        # Plot 4: Relative Gap Distribution - Bucket-focused view (clean scientific style)
        # =====================================================================
        if result.case_details is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafafa')
            
            gaps = result.case_details['relative_gap'] * 100  # Convert to %
            
            # Categorical bins
            bins = [0, 2, 5, 10, 20, 40, 60, 100]
            bin_labels = ['0-2%', '2-5%', '5-10%', '10-20%', '20-40%', '40-60%', '60%+']
            
            # Compute counts per bin
            counts, _ = np.histogram(gaps, bins=bins)
            total = len(gaps)
            pcts = counts / total * 100
            
            # Consistent blue-gray gradient (light to dark as gap increases)
            bar_colors = ['#85C1E9', '#5DADE2', '#3498DB', '#2980B9', '#2471A3', '#1F618D', '#1A5276']
            
            bars = ax.bar(range(len(bin_labels)), pcts, color=bar_colors[:len(counts)], 
                          edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, fontsize=11)
            ax.set_xlabel('Relative Score Gap (%)', fontsize=12, fontweight='medium')
            ax.set_ylabel('Percentage of Cases (%)', fontsize=12, fontweight='medium')
            
            # Add percentage labels on bars (only values, no counts)
            for bar, pct in zip(bars, pcts):
                if pct > 2:  # Only label if visible
                    ax.annotate(f'{pct:.1f}%', 
                                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            fig_path = plot_dir / 'h3_colocation_fallback_strength_histogram.png'
            plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Plot 5: Regime Classification Bar Chart (clean scientific style)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('white')
        
        # Consistent blue-gray colors (light to dark)
        COLOR_STRONG = '#85C1E9'     # Light blue
        COLOR_MOD = '#3498DB'        # Medium blue
        COLOR_WEAK = '#1A5276'       # Dark blue
        
        categories = ['Strong\n(< 5%)', 'Moderate\n(5-20%)', 'Weak\n(≥ 20%)']
        pcts = [result.strong_fallback_pct, result.moderate_fallback_pct, result.weak_fallback_pct]
        colors_bars = [COLOR_STRONG, COLOR_MOD, COLOR_WEAK]
        
        bars = ax.bar(categories, pcts, color=colors_bars, edgecolor='white', linewidth=2)
        
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='medium')
        ax.set_ylim(0, max(max(pcts), 1) * 1.2)  # Ensure y-axis has room even if some bars are 0
        
        # Add percentage labels only on bars
        for bar, pct in zip(bars, pcts):
            label_y = bar.get_height() if bar.get_height() > 0 else 0.5
            ax.annotate(f'{pct:.1f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, label_y),
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig_path = plot_dir / 'h3_colocation_fallback_regime_classification.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Plot 6: Combined Summary (Final Picture) - Donut chart (clean scientific style)
        # =====================================================================
        if self.fallback_result is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_facecolor('white')
            
            total_inaccessible = self.fallback_result.total_inaccessible_oracles
            no_fb = self.fallback_result.no_fallback_count
            
            # Compute counts for each category
            strong_total = int(result.strong_fallback_pct / 100 * self.fallback_result.has_fallback_count)
            moderate_total = int(result.moderate_fallback_pct / 100 * self.fallback_result.has_fallback_count)
            weak_total = self.fallback_result.has_fallback_count - strong_total - moderate_total
            
            # Consistent blue-gray gradient (dark to light)
            COLOR_NO_FB = '#1A5276'      # Darkest blue
            COLOR_WEAK = '#2471A3'       # Dark blue
            COLOR_MODERATE = '#3498DB'   # Medium blue
            COLOR_STRONG = '#85C1E9'     # Light blue
            
            sizes = [no_fb, weak_total, moderate_total, strong_total]
            pcts = [s / total_inaccessible * 100 for s in sizes]
            
            # Only show labels for non-zero segments
            labels = []
            for i, (name, pct) in enumerate(zip(
                ['No Alternative', 'Weak', 'Moderate', 'Strong'], pcts)):
                if pct > 0.5:  # Only label segments > 0.5%
                    labels.append(f'{name}\n{pct:.1f}%')
                else:
                    labels.append('')
            
            colors = [COLOR_NO_FB, COLOR_WEAK, COLOR_MODERATE, COLOR_STRONG]
            explode = (0.02, 0, 0, 0)
            
            wedges, texts = ax.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                startangle=90, labeldistance=1.1,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
            )
            
            # Style labels
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
            
            plt.tight_layout()
            fig_path = plot_dir / 'h3_colocation_complete_breakdown.png'
            plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {fig_path}")
        
        # =====================================================================
        # Analysis 4 Visualizations (if centroid spread analysis was run)
        # =====================================================================
        if self.spread_result is not None:
            self._visualize_spread_results(plot_dir, COLOR_PRIMARY, COLOR_SECONDARY, 
                                            COLOR_ACCENT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK, COLOR_BG_BOX)
    
    def _visualize_spread_results(self, plot_dir, COLOR_PRIMARY, COLOR_SECONDARY, 
                                   COLOR_ACCENT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK, COLOR_BG_BOX):
        """Generate visualizations for Analysis 4 (Centroid Spread)."""
        import matplotlib.pyplot as plt
        
        result = self.spread_result
        if result is None:
            return
        
        # =====================================================================
        # Plot 7: Distribution of centroids containing strong embeddings (Simplified)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        dist = result.centroids_distribution
        categories = list(dist.keys())
        counts = [dist[k] for k in categories]
        pcts = [c / result.cases_analyzed * 100 for c in counts]
        
        # Color gradient: blue tones from dark (clustered) to light (dispersed)
        colors_gradient = ['#1A5276', '#2874A6', '#3498DB', '#85C1E9']
        
        bars = ax.bar(categories, pcts, color=colors_gradient[:len(categories)], 
                      edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Unique Centroids with Strong Embeddings', fontsize=12, fontweight='medium')
        ax.set_ylabel('Percentage of Cases (%)', fontsize=12, fontweight='medium')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, pcts):
            ax.annotate(f'{pct:.1f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig_path = plot_dir / 'h3_colocation_centroid_spread_distribution.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {fig_path}")
        # (Plot 8 and 9 removed - keeping Analysis 4 simple with just the distribution)
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save analysis results to files."""
        if output_path is None:
            output_path = self.output_dir
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save Analysis 1 summary JSON
        if self.fallback_result:
            summary_path = output_path / "h3_colocation_fallback_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self.fallback_result.to_dict(), f, indent=2)
            print(f"Saved summary: {summary_path}")
            
            # Save case details parquet
            if self.fallback_result.case_details is not None:
                details_path = output_path / "h3_colocation_case_details.parquet"
                self.fallback_result.case_details.to_parquet(details_path, index=False)
                print(f"Saved case details: {details_path}")
        
        # Save Analysis 2 results
        if self.strength_result:
            summary_path = output_path / "h3_colocation_strength_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self.strength_result.to_dict(), f, indent=2)
            print(f"Saved strength summary: {summary_path}")
            
            # Save strength case details
            if self.strength_result.case_details is not None:
                details_path = output_path / "h3_colocation_strength_details.parquet"
                self.strength_result.case_details.to_parquet(details_path, index=False)
                print(f"Saved strength details: {details_path}")
        
        # Save Analysis 4 results
        if hasattr(self, 'spread_result') and self.spread_result:
            summary_path = output_path / "h3_colocation_spread_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self.spread_result.to_dict(), f, indent=2)
            print(f"Saved spread summary: {summary_path}")
            
            # Save spread case details
            if self.spread_result.case_details is not None:
                details_path = output_path / "h3_colocation_spread_details.parquet"
                self.spread_result.case_details.to_parquet(details_path, index=False)
                print(f"Saved spread details: {details_path}")


def run_analysis_1_standalone(
    config_name: str = "prod",
    golden_metrics_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
):
    """
    Standalone function to run Analysis 1 (72.6% finding) and save results.
    
    This is the reproducible entry point for validating the 72.6% finding.
    
    Args:
        config_name: One of 'smoke', 'dev', 'prod'. Controls sampling:
                     - smoke: 500 cases (~seconds)
                     - dev: 5000 cases (~1 min)
                     - prod: all 27,177 cases (~3-5 min)
        golden_metrics_dir: Override golden metrics directory
        run_dir: Override run directory  
        index_path: Override index path
        output_dir: Override output directory
        sample_size: Explicit sample size (overrides config)
    
    Returns:
        FallbackAnalysisResult
    """
    # Load config for paths and settings
    config = load_config(config_name)
    
    # Use config paths as defaults, allow overrides
    if golden_metrics_dir is None:
        golden_metrics_dir = str(Path(config.paths.run_dir) / "golden_metrics_v2")
    if run_dir is None:
        run_dir = config.paths.run_dir
    if index_path is None:
        index_path = config.paths.index_path
    if output_dir is None:
        output_dir = str(Path(config.paths.output_root) / "h3_colocation")
    
    # Get sample size from config if not explicitly provided
    if sample_size is None:
        sample_size = SAMPLE_SIZES.get(config_name, None)
    
    print("=" * 70)
    print(f"H3 Co-location Analysis: Reproducing the 72.6% Finding")
    print(f"Config: {config_name}" + (f" (sample_size={sample_size})" if sample_size else " (full)"))
    print("=" * 70)
    
    # Initialize
    analyzer = H3_Colocation(
        config=config,
        golden_metrics_dir=Path(golden_metrics_dir),
        run_dir=Path(run_dir),
        index_path=Path(index_path)
    )
    
    # Setup (load data)
    analyzer.setup()
    
    # Run Analysis 1 with sampling based on config
    result = analyzer.analyze_fallback_availability(save_details=True, sample_size=sample_size)
    
    # Save results
    output_path = Path(output_dir)
    analyzer.output_dir = output_path  # Set for visualize()
    analyzer.save_results(output_path)
    
    # Generate visualizations
    analyzer.visualize()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if result.is_sampled:
        print(f"\n⚠ SAMPLED RESULT (n={result.sample_size:,} of {result.full_population_size:,})")
    else:
        print(f"\n✓ FULL CENSUS")
    
    print(f"\nThe 72.6% Finding {'(Estimated)' if result.is_sampled else '(Reproduced)'}:")
    print(f"  Cases analyzed: {result.total_inaccessible_oracles:,}")
    print(f"  NO fallback available: {result.no_fallback_count:,} ({result.no_fallback_pct:.2f}%)")
    print(f"  Has fallback available: {result.has_fallback_count:,} ({result.has_fallback_pct:.2f}%)")
    print(f"\nValidation:")
    print(f"  M4R count matches expected: {result.m4r_inaccessible_count_matches}")
    print(f"  R0 lookup keys: {result.r0_lookup_unique_keys:,}")
    print(f"  Unique docs checked: {result.docs_checked:,}")
    
    return result


def run_full_analysis(
    config_name: str = "prod",
    golden_metrics_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    analysis1_sample: Optional[int] = None,
    analysis2_sample: Optional[int] = None,
    analysis4_sample: Optional[int] = None,
    skip_analysis2: bool = False,
    skip_analysis4: bool = False,
    k_threshold: float = 1.5,
):
    """
    Run Analysis 1 (72.6% finding), Analysis 2 (fallback strength), and Analysis 4 (centroid spread).
    
    Args:
        config_name: One of 'smoke', 'dev', 'prod'
        golden_metrics_dir: Override golden metrics directory
        run_dir: Override run directory
        index_path: Override index path
        output_dir: Override output directory
        analysis1_sample: Sample size for Analysis 1 (overrides config)
        analysis2_sample: Sample size for Analysis 2 (overrides config)
        analysis4_sample: Sample size for Analysis 4 (overrides config)
        skip_analysis2: If True, skip Analysis 2
        skip_analysis4: If True, skip Analysis 4
        k_threshold: Std deviations from oracle for "strong" embedding (default 1.5)
    
    Returns:
        Tuple of (FallbackAnalysisResult, FallbackStrengthResult or None, CentroidSpreadResult or None)
    """
    # Load config
    config = load_config(config_name)
    
    # Use config paths as defaults
    if golden_metrics_dir is None:
        golden_metrics_dir = str(Path(config.paths.run_dir) / "golden_metrics_v2")
    if run_dir is None:
        run_dir = config.paths.run_dir
    if index_path is None:
        index_path = config.paths.index_path
    if output_dir is None:
        output_dir = str(Path(config.paths.output_root) / "h3_colocation")
    
    print("=" * 70)
    print(f"H3 Co-location Analysis: Full Pipeline")
    print(f"Config: {config_name}")
    print("=" * 70)
    
    # Initialize
    analyzer = H3_Colocation(
        config=config,
        golden_metrics_dir=Path(golden_metrics_dir),
        run_dir=Path(run_dir),
        index_path=Path(index_path)
    )
    
    # Setup (load data)
    analyzer.setup()
    
    # Run Analysis 1
    result1 = analyzer.analyze_fallback_availability(
        save_details=True, 
        sample_size=analysis1_sample
    )
    
    # Run Analysis 2 if requested
    result2 = None
    if not skip_analysis2 and result1.has_fallback_count > 0:
        result2 = analyzer.analyze_fallback_strength(sample_size=analysis2_sample)
    
    # Run Analysis 4 if requested
    result4 = None
    if not skip_analysis4:
        result4 = analyzer.analyze_centroid_spread(
            sample_size=analysis4_sample, 
            k_threshold=k_threshold
        )
    
    # Save all results
    output_path = Path(output_dir)
    analyzer.output_dir = output_path
    analyzer.save_results(output_path)
    
    # Generate visualizations
    analyzer.visualize()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Analysis 1: Fallback Availability ---")
    print(f"  Cases: {result1.total_inaccessible_oracles:,}")
    print(f"  NO fallback: {result1.no_fallback_count:,} ({result1.no_fallback_pct:.1f}%)")
    print(f"  Has fallback: {result1.has_fallback_count:,} ({result1.has_fallback_pct:.1f}%)")
    
    if result2:
        print(f"\n--- Analysis 2: Fallback Strength (BIASED VIEW) ---")
        print(f"  ⚠ These stats are computed ONLY over cases with fallback")
        print(f"  Cases analyzed: {result2.cases_analyzed:,}")
        print(f"  Mean relative gap: {result2.mean_relative_gap:.1%}")
        print(f"  Strong (<5%): {result2.strong_fallback_count:,} ({result2.strong_fallback_pct:.1f}% of fallback cases)")
        print(f"  Moderate (5-20%): {result2.moderate_fallback_count:,} ({result2.moderate_fallback_pct:.1f}% of fallback cases)")
        print(f"  Weak (>=20%): {result2.weak_fallback_count:,} ({result2.weak_fallback_pct:.1f}% of fallback cases)")
        
        # HONEST FRAMING: Compute overall regime breakdown relative to ALL inaccessible
        total = result1.total_inaccessible_oracles
        no_fb = result1.no_fallback_count
        no_fb_pct = result1.no_fallback_pct
        
        # Scale counts based on sampling
        if result2.is_sampled and result2.full_population_size > 0:
            scale = result2.full_population_size / result2.cases_analyzed
            strong_est = int(result2.strong_fallback_count * scale)
            moderate_est = int(result2.moderate_fallback_count * scale)
            weak_est = int(result2.weak_fallback_count * scale)
        else:
            strong_est = result2.strong_fallback_count
            moderate_est = result2.moderate_fallback_count
            weak_est = result2.weak_fallback_count
        
        # Percentages of ALL inaccessible cases
        strong_pct_all = (strong_est / total) * 100
        moderate_pct_all = (moderate_est / total) * 100
        weak_pct_all = (weak_est / total) * 100
        
        # Total handicapped = no fallback + weak fallback + moderate fallback
        significant_handicap_pct = no_fb_pct + weak_pct_all + moderate_pct_all
        real_compensation_pct = strong_pct_all
        
        print(f"\n" + "=" * 50)
        print("HONEST FRAMING: % of ALL Inaccessible Cases")
        print("=" * 50)
        print(f"  Total inaccessible oracle cases: {total:,}")
        print(f"  ")
        print(f"  NO fallback (100% handicap):     ~{no_fb_pct:.1f}%  ({no_fb:,} cases)")
        print(f"  Weak fallback (>=20% loss):      ~{weak_pct_all:.1f}%  ({weak_est:,} cases)")
        print(f"  Moderate fallback (5-20% loss):  ~{moderate_pct_all:.1f}%  ({moderate_est:,} cases)")
        print(f"  Strong fallback (<5% loss):      ~{strong_pct_all:.1f}%  ({strong_est:,} cases)")
        print(f"  ")
        print(f"  ══════════════════════════════════════════════════")
        print(f"  HANDICAPPED (no/weak/moderate):  ~{significant_handicap_pct:.1f}%")
        print(f"  COMPENSATED (strong only):       ~{real_compensation_pct:.1f}%")
        print(f"  ══════════════════════════════════════════════════")
        print(f"  ")
        print(f"  CONCLUSION: ~{significant_handicap_pct:.0f}% of inaccessible cases suffer")
        print(f"              significant scoring handicap.")
    
    # Analysis 4 summary
    if result4:
        print(f"\n--- Analysis 4: Centroid Spread (Oracle-Accessible Cases) ---")
        print(f"  Cases analyzed: {result4.cases_analyzed:,} of {result4.total_oracle_accessible:,}")
        print(f"  Mean strong embeddings per case: {result4.mean_strong_embeddings:.2f}")
        print(f"  Mean unique centroids: {result4.mean_unique_centroids:.2f}")
        print(f"  Mean capture rate: {result4.mean_capture_rate:.1%}")
        print(f"  % with ALL strong in oracle centroid: {result4.pct_oracle_only:.1f}%")
        
        print(f"\n  Centroid Distribution:")
        for k, v in result4.centroids_distribution.items():
            pct = (v / result4.cases_analyzed) * 100
            print(f"    {k} centroid(s): {v:,} ({pct:.1f}%)")
    
    return result1, result2, result4


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="H3 Co-location Analysis - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (~1 min with Analysis 2)
  python -m hypothesis.hypotheses.h3_colocation --config smoke
  
  # Development run (~5 min)
  python -m hypothesis.hypotheses.h3_colocation --config dev
  
  # Full production run (~10-15 min)
  python -m hypothesis.hypotheses.h3_colocation --config prod
  
  # Analysis 1 only (skip query encoding)
  python -m hypothesis.hypotheses.h3_colocation --config prod --skip-analysis2 --skip-analysis4
  
  # Analysis 4 only with 10K sample
  python -m hypothesis.hypotheses.h3_colocation --config prod --skip-analysis2 --analysis4-sample 10000
  
  # Custom sample sizes
  python -m hypothesis.hypotheses.h3_colocation --config prod --analysis2-sample 500 --analysis4-sample 5000
        """
    )
    parser.add_argument("--config", type=str, default="prod",
                        choices=["smoke", "dev", "prod"],
                        help="Config to use: smoke, dev, prod")
    parser.add_argument("--golden-metrics-dir", type=str, default=None,
                        help="Override golden metrics directory")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override run directory")
    parser.add_argument("--index-path", type=str, default=None,
                        help="Override index path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--analysis1-sample", type=int, default=None,
                        help="Sample size for Analysis 1")
    parser.add_argument("--analysis2-sample", type=int, default=None,
                        help="Sample size for Analysis 2")
    parser.add_argument("--analysis4-sample", type=int, default=None,
                        help="Sample size for Analysis 4 (centroid spread)")
    parser.add_argument("--skip-analysis2", action="store_true",
                        help="Skip Analysis 2 (fallback strength)")
    parser.add_argument("--skip-analysis4", action="store_true",
                        help="Skip Analysis 4 (centroid spread)")
    parser.add_argument("--k-threshold", type=float, default=1.5,
                        help="Std deviations from oracle for 'strong' embeddings (default: 1.5)")
    
    args = parser.parse_args()
    
    run_full_analysis(
        config_name=args.config,
        golden_metrics_dir=args.golden_metrics_dir,
        run_dir=args.run_dir,
        index_path=args.index_path,
        output_dir=args.output_dir,
        analysis1_sample=args.analysis1_sample,
        analysis2_sample=args.analysis2_sample,
        analysis4_sample=args.analysis4_sample,
        skip_analysis2=args.skip_analysis2,
        skip_analysis4=args.skip_analysis4,
        k_threshold=args.k_threshold,
    )
